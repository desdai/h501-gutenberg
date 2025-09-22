# tt_gutenberg/authors.py

import pandas as pd
from typing import List
from .joins import author_language_table
from .clean import clean_alias_column

def list_authors(authors_df: pd.DataFrame,
                 languages_df: pd.DataFrame,
                 metadata_df: pd.DataFrame,
                 by_languages: bool = True,
                 alias: bool = True) -> List[str]:
    """
    Return a list of author aliases ordered by translation count.

    Parameters
    ----------
    authors_df : DataFrame with at least ['gutenberg_author_id', 'alias']
    languages_df : DataFrame with at least ['gutenberg_id', 'language']
    metadata_df : DataFrame with at least ['gutenberg_id', 'gutenberg_author_id']
    by_languages : bool
        If True, count distinct languages per alias. Else count total translations.
    alias : bool
        Must be True. Ensures we return aliases.

    Returns
    -------
    List[str] : aliases sorted by translation count.
    """
    if not alias:
        raise ValueError("This function is defined to return aliases only.")

    # clean aliases
    authors_df = clean_alias_column(authors_df)

    # join into alias-language pairs
    al = author_language_table(authors_df, languages_df, metadata_df)

    if by_languages:
        counts = al.groupby("alias")["language"].nunique().reset_index(name="n_langs")
        order_col = "n_langs"
    else:
        counts = al.groupby("alias")["language"].size().reset_index(name="n_trans")
        order_col = "n_trans"

    counts = counts.sort_values([order_col, "alias"], ascending=[False, True])
    return counts["alias"].tolist()
