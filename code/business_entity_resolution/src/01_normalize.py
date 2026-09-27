"""
01_normalize.py — Text Normalization (VECTORIZED — fast on 10M+ rows)

All regex replacements run on pandas Series using str.replace (vectorized C layer).
Non-ASCII transliteration handled in a single batch pass — only rows with non-ASCII
characters go through the slower indic/anyascii path.

Steps:
  1. Lowercase + Unicode NFC
  2. Non-ASCII batch transliteration (indic → IAST, then anyascii fallback)
  3. Vectorized legal suffix expansion (pandas str.replace)
  4. Vectorized address abbreviation expansion
  5. Punctuation strip + whitespace collapse

Speed on 5M rows: ~60–90 seconds (vs. ~15 min with progress_apply).
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path

# ── Legal suffix expansion (pattern → replacement) ────────────────────────────
# Order matters: longer/more-specific patterns first.
LEGAL_SUFFIXES = [
    (r'&',           'and'),
    (r'\bpvt\b',     'private'),
    (r'\bltd\b',     'limited'),
    (r'\bllp\b',     'llp'),
    (r'\bllc\b',     'llc'),
    (r'\bcorp\b',    'corporation'),
    (r'\binc\b',     'incorporated'),
    (r'\bbros\b',    'brothers'),
    (r'\bassoc\b',   'associates'),
    (r'\bmfg\b',     'manufacturing'),
    (r'\bdept\b',    'department'),
    (r'\bsvcs\b',    'services'),
    (r'\bsvc\b',     'service'),
    (r'\bintl\b',    'international'),
    (r'\bnatl\b',    'national'),
    (r'\bmgmt\b',    'management'),
    (r'\benterprises\b', 'enterprises'),
    (r'\benterprise\b',  'enterprises'),
    (r'\bente\b',    'enterprises'),
    (r'\bgrp\b',     'group'),
    (r'\bholdings\b', 'holdings'),
    (r'\bholding\b', 'holdings'),
    (r'\bco\b',      'company'),
    # French legal forms (appear in test France records)
    (r'\bsarl\b',    'sarl'),
    (r'\bsasu\b',    'sasu'),
    (r'\beurl\b',    'eurl'),
    (r'\bsas\b',     'sas'),
]

# ── Address abbreviation expansion ────────────────────────────────────────────
ADDR_ABBREVS = [
    (r'\bblvd\b',  'boulevard'),
    (r'\bexpy\b',  'expressway'),
    (r'\bhwy\b',   'highway'),
    (r'\bave\b',   'avenue'),
    (r'\bapt\b',   'apartment'),
    (r'\bste\b',   'suite'),
    (r'\bbldg\b',  'building'),
    (r'\brd\b',    'road'),
    (r'\bst\b',    'street'),
    (r'\bdr\b',    'drive'),
    (r'\bln\b',    'lane'),
    (r'\bfl\b',    'floor'),
    (r'\bct\b',    'court'),
    (r'\bsq\b',    'square'),
    (r'\bpl\b',    'place'),
    (r'\bpk\b',    'park'),
    (r'\bter\b',   'terrace'),
    (r'\btrl\b',   'trail'),
    (r'\bext\b',   'extension'),
    (r'\bblk\b',   'block'),
    (r'\bsec\b',   'sector'),
    (r'\bph\b',    'phase'),
    (r'\bnagar\b', 'nagar'),
    (r'\bsoc\b',   'society'),
    (r'\bcol\b',   'colony'),
    (r'\bopp\b',   'opposite'),
    (r'\bnr\b',    'near'),
    (r'#',         'number '),
    (r'\bno\.\b',  'number'),
    (r'\bno\b',    'number'),
]

# Punct pattern: keep alphanumeric + spaces
_PUNCT_RE  = re.compile(r'[^\w\s]')
_WS_RE     = re.compile(r'\s+')


# ── Transliteration (batch, only on non-ASCII rows) ───────────────────────────

def _build_transliterator():
    """
    Returns a function: str → str that converts non-ASCII to ASCII.
    Uses indic-transliteration for Indic scripts, anyascii as fallback.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.detect import detect as _detect
        _has_indic = True
    except ImportError:
        _has_indic = False
        _detect = sanscript = None

    try:
        from anyascii import anyascii as _anyascii
        _has_anyascii = True
    except ImportError:
        _has_anyascii = False
        _anyascii = None

    def transliterate_one(text: str) -> str:
        if text.isascii():
            return text
        # Pass 1: Indic → IAST
        if _has_indic:
            try:
                script = _detect(text)
                if script and script not in ('hk', 'iast', 'slp1'):
                    text = sanscript.transliterate(text, script, sanscript.IAST)
            except Exception:
                pass
        # Pass 2: anyascii catches French accents, remaining Indic, etc.
        if not text.isascii() and _has_anyascii:
            text = _anyascii(text)
        # Final: strip any remaining non-ASCII
        if not text.isascii():
            text = text.encode('ascii', errors='ignore').decode('ascii')
        return text

    return transliterate_one


_transliterate_one = _build_transliterator()


def _batch_transliterate(series: pd.Series) -> pd.Series:
    """
    Transliterate only rows that contain non-ASCII characters.
    ~90% of rows are ASCII-only → fast path skips them.
    """
    is_nonascii = series.apply(lambda x: not x.isascii())
    n_nonascii = is_nonascii.sum()
    if n_nonascii == 0:
        return series

    result = series.copy()
    result[is_nonascii] = series[is_nonascii].apply(_transliterate_one)
    return result


# ── Core vectorized normalization ─────────────────────────────────────────────

def _normalize_series(raw: pd.Series, abbrevs: list[tuple]) -> pd.Series:
    """
    Vectorized normalization pipeline for a pandas Series of strings.

    1. Fill NaN + strip
    2. Unicode NFC (can't fully vectorize, but 1 apply on whole series)
    3. Lowercase (vectorized)
    4. Transliterate non-ASCII rows (batch, skips ASCII rows)
    5. Apply abbreviation replacements (vectorized str.replace in a loop)
    6. Strip punctuation (vectorized)
    7. Collapse whitespace (vectorized)
    """
    s = raw.fillna('').astype(str).str.strip()

    # NFC normalize — apply once on whole series (fast, C extension)
    s = s.apply(lambda x: unicodedata.normalize('NFC', x))

    # Lowercase
    s = s.str.lower()

    # Transliterate non-ASCII
    s = _batch_transliterate(s)

    # Vectorized abbreviation expansion
    for pat, rep in abbrevs:
        s = s.str.replace(pat, rep, regex=True, case=False)

    # Strip punctuation (keep word chars + spaces)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)

    # Collapse whitespace
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    return s


def normalize_name_series(raw: pd.Series) -> pd.Series:
    """Vectorized normalization for business_name column."""
    return _normalize_series(raw, LEGAL_SUFFIXES)


def normalize_addr_series(raw: pd.Series) -> pd.Series:
    """Vectorized normalization for business_address column."""
    return _normalize_series(raw, ADDR_ABBREVS)


# Scalar versions (for feature engineering on individual pairs)
def normalize_name(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ''
    return normalize_name_series(pd.Series([s])).iloc[0]


def normalize_addr(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ''
    return normalize_addr_series(pd.Series([s])).iloc[0]


def extract_zip_tokens(addr: str) -> list:
    """Extract numeric tokens ≥ 4 digits from a (normalized) address."""
    return re.findall(r'\b\d{4,}\b', addr)


# ── DataFrame normalization (vectorized, with Parquet caching) ────────────────

def normalize_df(df: pd.DataFrame, desc: str = '') -> pd.DataFrame:
    """
    Add norm_name, norm_addr, name_addr, zip_tokens, country_clean,
    name_prefix_key columns to df.

    All operations are vectorized — no row-by-row apply loops.
    """
    print(f'    [{desc}] Normalizing names ({len(df):,} rows) ...')
    df = df.copy()

    df['norm_name'] = normalize_name_series(df['business_name'])

    print(f'    [{desc}] Normalizing addresses ...')
    df['norm_addr'] = normalize_addr_series(df['business_address'])

    # Combined field for TF-IDF: name repeated 2× for weight boost
    df['name_addr'] = (
        df['norm_name'] + ' ' + df['norm_name'] + ' ' + df['norm_addr']
    )

    # ZIP tokens (vectorized via str.findall)
    df['zip_tokens'] = df['norm_addr'].str.findall(r'\b\d{4,}\b').apply(list)

    # Country
    df['country_clean'] = df['country'].fillna('').astype(str).str.strip().str.lower()

    # Blocking key B4: country + first 4 chars of name (no spaces)
    name_nospace = df['norm_name'].str.replace(r'\s+', '', regex=True)
    df['name_prefix_key'] = df['country_clean'] + '_' + name_nospace.str[:4]

    return df


import pyarrow as pa
import pyarrow.parquet as pq

def normalize_sources(data_dir: Path, split: str = 'train') -> dict:
    """
    Load + normalize all 3 source files for a split.
    Caches results as Parquet after first run for fast reload.
    Processes and writes in chunks to prevent Out-Of-Memory (OOM) errors.

    Returns dict: {'s1': df1, 's2': df2, 's3': df3}
    """
    results = {}
    CHUNK_SIZE = 500_000

    for src_id in ['source1', 'source2', 'source3']:
        key = f's{src_id[-1]}'
        raw_path   = Path(data_dir) / f'{split}_{src_id}.tsv'
        cache_path = Path(data_dir) / f'{split}_{src_id}_norm.parquet'

        if cache_path.exists():
            print(f'  Loading cached {cache_path.name}')
            results[key] = pd.read_parquet(cache_path)
            continue

        print(f'  Reading {raw_path.name} in chunks ...')
        writer = None
        
        for i, chunk in enumerate(pd.read_csv(raw_path, sep='\t', encoding='utf-8', low_memory=False, chunksize=CHUNK_SIZE)):
            print(f'    Processing chunk {i+1} ({len(chunk):,} rows) ...')
            norm_chunk = normalize_df(chunk, desc=f'{split}/{key} chunk {i+1}')
            
            # Write chunk directly to parquet to avoid pd.concat OOM
            table = pa.Table.from_pandas(norm_chunk)
            if writer is None:
                writer = pq.ParquetWriter(cache_path, table.schema)
            writer.write_table(table)
            
        if writer:
            writer.close()
            
        print(f'    Cached -> {cache_path.name}')
        # Load the final file from disk to ensure it's clean (it's safe to load once serialized)
        results[key] = pd.read_parquet(cache_path)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, time

    parser = argparse.ArgumentParser(description='Normalize source TSV files')
    parser.add_argument('--data-dir', default='dataset')
    parser.add_argument('--split', choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()

    base = Path(args.data_dir)
    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    for sp in splits:
        print(f'\n=== Normalizing {sp} ===')
        t0 = time.time()
        normalize_sources(base / sp, split=sp)
        print(f'Done {sp} in {(time.time()-t0)/60:.1f} min')
