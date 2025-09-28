from keybert import KeyBERT
from collections import Counter
import pandas as pd
from tqdm import tqdm


def extract_keywords_df(
    df: pd.DataFrame,
    text_cols: tuple[str, ...] = ("title", "abstract"),
    kw_model: KeyBERT | None = None,
    top_k: int = 8,
    ngram_range: tuple[int, int] = (1, 3),
    stop_words: str = "english",
    keywords_col: str = "keywords",
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """
    Add a `keywords_col` list column to `df` and return corpus-level topics.

    Args:
        df (pd.DataFrame): DataFrame with text columns (e.g., title, abstract).
        text_cols (list[str]): Columns to concatenate as the document text.
        kw_model (KeyBERT, optional): Existing KeyBERT instance.
        top_k (int): Phrases to keep per row.
        ngram_range (tuple[int, int]): (min_n, max_n) for candidate phrases.
        stop_words (str or list, optional): Passed through to KeyBERT.
        keywords_col (str): Name of the new column to create.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with an added `keywords_col`.
            - list[tuple[str, int]]: List of (phrase, document_frequency) tuples.
    """
    if kw_model is None:
        kw_model = KeyBERT("paraphrase-MiniLM-L6-v2")

    docs = (
        df[list(text_cols)]
        .fillna("")                                 # NaNs → ""
        .apply(lambda row: " ".join(map(str, row)), axis=1)
        .tolist()
    )

    all_keywords: list[list[str]] = []
    for doc in tqdm(docs, desc="Extracting keywords"):
        kws = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=ngram_range,
            stop_words=stop_words,
            top_n=top_k,
        )
        all_keywords.append([phrase for phrase, _ in kws])

    df_out = df.copy()
    df_out[keywords_col] = all_keywords

    df_counter = Counter()
    for kw_list in all_keywords:
        df_counter.update(set(kw_list))  # document frequency

    topics = sorted(df_counter.items(), key=lambda x: (-x[1], x[0]))
    return df_out, topics
