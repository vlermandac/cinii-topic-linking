from parser import parse_and_extract_articles_langs_from_dirs
from preprocess import tokenize_corpus
from pathlib import Path
import json
import tomllib

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"


def generate_and_store_tokens(rdf_dirs: list[str], token_path: str):
    """
    Generate and store tokens for the given articles.

    Args:
        rdf_dirs (list[str]): List of directories containing RDF files.
        token_path (str): Path to the directory where tokens will be stored.
    """
    token_path = ROOT_DIR / Path(token_path)
    words_path = token_path / "words"
    k_shingles_path = token_path / "k_shingles"

    words_path.mkdir(parents=True, exist_ok=True)
    k_shingles_path.mkdir(parents=True, exist_ok=True)

    eng_articles, jpn_articles = parse_and_extract_articles_langs_from_dirs(
        rdf_dirs)

    print(f"English Articles: {len(eng_articles)}")
    print(f"Japanese Articles: {len(jpn_articles)}")

    en_articles_words = tokenize_corpus(
        eng_articles, method="words", lang="en")
    en_articles_k_shingles = tokenize_corpus(
        eng_articles, method="k_shingles", k=5, lang="en")

    jp_articles_words = tokenize_corpus(
        jpn_articles, method="words", lang="jp")
    jp_articles_k_shingles = tokenize_corpus(
        jpn_articles, method="k_shingles", k=5, lang="jp")

    with open(words_path / "en_words.json", "w", encoding="utf-8") as f:
        json.dump(en_articles_words, f, ensure_ascii=False, indent=2)

    with open(k_shingles_path / "en_k_shingles.json", "w", encoding="utf-8") as f:
        json.dump(en_articles_k_shingles, f, ensure_ascii=False, indent=2)

    with open(words_path / "jp_words.json", "w", encoding="utf-8") as f:
        json.dump(jp_articles_words, f, ensure_ascii=False, indent=2)

    with open(k_shingles_path / "jp_k_shingles.json", "w", encoding="utf-8") as f:
        json.dump(jp_articles_k_shingles, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Read the toml file
    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    rdf_dirs = [ROOT_DIR / Path(p) for p in config["paths"]["rdf_dirs"]]
    test_sample = [ROOT_DIR / Path(config["paths"]["test_sample"])]
    token_path = Path("data/tokens/")

    generate_and_store_tokens(rdf_dirs, token_path)
