import pandas as pd
import numpy as np

def clean_alias_column(authors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'alias' column: strip, coerce placeholders to NaN, drop missing.
    Requires columns: ['alias'].
    """
    out = authors_df.copy()
    if 'alias' not in out.columns:
        raise ValueError("authors_df must contain an 'alias' column.")
    out['alias'] = out['alias'].astype(str).str.strip()
    bad = {'', 'None', 'nan', 'NaN', '-', 'UNKNOWN', 'Unknown'}
    out.loc[out['alias'].isin(bad), 'alias'] = np.nan
    out = out.dropna(subset=['alias']).copy()
    return out