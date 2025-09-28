import re
from typing import Literal
from nltk.corpus import stopwords
from fugashi import Tagger
import nltk

nltk.download("stopwords", quiet=True)

# Japanese tokenizer
tagger = Tagger()


def tokenize_corpus(
    corpus: dict[str, dict],
    method: Literal["words", "k_shingles"] = "words",
    lang: Literal["en", "jp"] = "en",
    k: int = 5
) -> dict[str, list[str]]:
    """
    Tokenizes the combined 'title' + 'abstract' text from each document in the corpus.

    Args:
        corpus (dict[str, dict]): A dictionary mapping document URIs to dicts with 'title' and 'abstract'.
        method (Literal["words", "k_shingles"]): The tokenization method to use.
        lang (Literal["en", "jp"]): The language of the text.
        k (int): The size of the shingles, if using k-shingles.

    Returns:
        dict[str, list[str]]: A mapping from document URIs to lists of tokens.
    """
    result = {}

    for uri, doc in corpus.items():
        text = (doc.get("title", "") + " " + doc.get("abstract", "")).strip()

        if method == "words":
            tokens = tokenize_words(text, lang)
        elif method == "k_shingles":
            tokens = tokenize_k_shingles(text, lang, k)
        else:
            raise ValueError(f"Unsupported method: {method}")

        result[uri] = tokens

    return result


def tokenize_words(text: str, lang: Literal["en", "jp"]) -> list[str]:
    if lang == "en":
        STOP = set(stopwords.words("english"))
        text = re.sub(r"[^A-Za-z0-9 ]+", " ", text.lower())
        text = re.sub(r"\b\w{1}\b", "", text)  # remove single characters
        text = re.sub(r"\s+", " ", text).strip()
        # Use set to remove duplicates, return as list
        return list(set(w for w in text.split() if w not in STOP))

    elif lang == "jp":
        # Use set to remove duplicates, return as list
        return list(set(word.surface for word in tagger(text)))

    else:
        raise ValueError(f"Unsupported language: {lang}")


def tokenize_k_shingles(text: str, lang: Literal["en", "jp"], k: int = 5) -> list[str]:
    if lang == "en":
        text = re.sub(r"[^A-Za-z0-9 ]+", " ", text.lower())
    elif lang == "jp":
        text = re.sub(r"[^\w\u3000-\u30FF\u4E00-\u9FFF]+", " ", text)
    else:
        raise ValueError(f"Unsupported language: {lang}")

    text = text.replace(" ", "_")

    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i+k])

    return list(shingles)
