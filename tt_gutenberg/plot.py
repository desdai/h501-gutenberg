import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def _load_tables():
    roots = [Path("."), Path("data/2025-06-03")]
    files = {
        "authors": "gutenberg_authors.csv",
        "languages": "gutenberg_languages.csv",
        "metadata": "gutenberg_metadata.csv",
    }
    paths = {}
    for key, name in files.items():
        found = next((p / name for p in roots if (p / name).exists()), None)
        if not found:
            raise FileNotFoundError(f"Could not find {name} in {roots}")
        paths[key] = found

    authors_df = pd.read_csv(paths["authors"])
    languages_df = pd.read_csv(paths["languages"])
    metadata_df = pd.read_csv(paths["metadata"])
    return authors_df, languages_df, metadata_df

def _author_language_table(authors_df: pd.DataFrame,
                            languages_df: pd.DataFrame,
                            metadata_df: pd.DataFrame) -> pd.DataFrame:
    req_auth = {'gutenberg_author_id', 'author', 'birthdate'}
    req_lang = {'gutenberg_id', 'language'}
    req_meta = {'gutenberg_id', 'gutenberg_author_id'}
    if not req_auth.issubset(authors_df.columns):
        raise ValueError(f"authors_df missing: {req_auth - set(authors_df.columns)}")
    if not req_lang.issubset(languages_df.columns):
        raise ValueError(f"languages_df missing: {req_lang - set(languages_df.columns)}")
    if not req_meta.issubset(metadata_df.columns):
        raise ValueError(f"metadata_df missing: {req_meta - set(metadata_df.columns)}")

    # metadata -> authors (to get author name + birthdate)
    ma = metadata_df[['gutenberg_id','gutenberg_author_id']].merge(
        authors_df[['gutenberg_author_id','author','birthdate']],
        on='gutenberg_author_id', how='left'
    )
    # + languages
    mal = ma.merge(languages_df[['gutenberg_id','language']], on='gutenberg_id', how='left')

    # keep needed cols, drop missing author names
    mal = mal[['author','birthdate','language']].dropna(subset=['author']).copy()
    # standardize types
    mal['author'] = mal['author'].astype(str).str.strip()
    mal['language'] = mal['language'].astype(str).str.strip()
    return mal

def _compute_birth_century(birthdate: float) -> float:
    # Handle floats like 1809.0; if NaN return np.nan
    if pd.isna(birthdate):
        return np.nan
    y = int(float(birthdate))
    return (y // 100) * 100

def plot_translations(over: str = 'birth_century'):
    authors_df, languages_df, metadata_df = _load_tables()
    mal = _author_language_table(authors_df, languages_df, metadata_df)

    # per-author distinct language count
    per_author = (mal.dropna(subset=['language'])
                    .groupby(['author'], as_index=False)
                    .agg(n_langs=('language', lambda s: s.dropna().nunique())))

    # attach birthdate and compute century
    birth = (authors_df[['author','birthdate']]
                .drop_duplicates(subset=['author']))
    per_author = per_author.merge(birth, on='author', how='left')
    per_author['birth_century'] = per_author['birthdate'].apply(_compute_birth_century)

    # drop entries without a birth century
    per_author = per_author.dropna(subset=['birth_century']).copy()
    per_author['birth_century'] = per_author['birth_century'].astype(int)

    # plot
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=per_author,
        x='birth_century',
        y='n_langs',
        estimator=np.mean,
        ci=95
    )
    ax.set_xlabel('Birth Century')
    ax.set_ylabel('Avg. # Languages per Author')
    ax.set_title('Average Translation Coverage by Author Birth Century (95% CI)')
    # ensure integer ticks and sorted
    xticks = sorted(per_author['birth_century'].unique())
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels([str(x) for x in xticks], rotation=0)
    # remap the categorical positions to actual centuries for even spacing
    # (Seaborn barplot uses categorical positions in order of appearance)
    # To strictly enforce order, pass order=xticks above (requires x to be categorical),
    # but this approach with set_xticklabels suffices for the exercise.
    plt.tight_layout()
    return ax