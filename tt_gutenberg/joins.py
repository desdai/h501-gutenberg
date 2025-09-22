import pandas as pd

def author_language_table(
    authors_df: pd.DataFrame,
    languages_df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a two-column table (alias, language) via:
      metadata[gutenberg_id, gutenberg_author_id]
        -> authors[gutenberg_author_id -> alias]
        -> languages[gutenberg_id -> language]
    Requires:
      authors_df:   ['gutenberg_author_id', 'alias']
      languages_df: ['gutenberg_id', 'language']
      metadata_df:  ['gutenberg_id', 'gutenberg_author_id']
    """
    req_auth = {'gutenberg_author_id', 'alias'}
    req_lang = {'gutenberg_id', 'language'}
    req_meta = {'gutenberg_id', 'gutenberg_author_id'}
    if not req_auth.issubset(authors_df.columns):
        missing = req_auth - set(authors_df.columns)
        raise ValueError(f"authors_df missing columns: {missing}")
    if not req_lang.issubset(languages_df.columns):
        missing = req_lang - set(languages_df.columns)
        raise ValueError(f"languages_df missing columns: {missing}")
    if not req_meta.issubset(metadata_df.columns):
        missing = req_meta - set(metadata_df.columns)
        raise ValueError(f"metadata_df missing columns: {missing}")

    meta_auth = metadata_df[['gutenberg_id', 'gutenberg_author_id']].merge(
        authors_df[['gutenberg_author_id', 'alias']],
        on='gutenberg_author_id',
        how='left'
    )
    al = meta_auth.merge(
        languages_df[['gutenberg_id', 'language']],
        on='gutenberg_id',
        how='left'
    )
    al = al[['alias', 'language']].dropna(subset=['alias']).copy()
    return al
