import tomllib
from parser import parse_and_extract_articles_langs_from_dirs
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(str(Path().resolve() / "src"))


ROOT = Path().resolve()
config_path = "config.toml"

with open(config_path, "rb") as f:
    config = tomllib.load(f)

rdf_dirs = [ROOT / Path(d) for d in config["paths"]["rdf_dirs"]]

cache_dir = ROOT / "data" / "cache"

data_path = ROOT / "data" / "cinii_topics_benchmark" / "article_labels.csv"

en_articles, jp_articles = parse_and_extract_articles_langs_from_dirs(rdf_dirs)

df_en = pd.DataFrame.from_dict(
    en_articles, orient='index').reset_index(drop=True)
df_jp = pd.DataFrame.from_dict(
    jp_articles, orient='index').reset_index(drop=True)

cache_dir.mkdir(parents=True, exist_ok=True)

df_en.to_pickle(cache_dir / "en_articles.pkl")
df_jp.to_pickle(cache_dir / "jp_articles.pkl")
