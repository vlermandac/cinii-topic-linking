from pathlib import Path
import json
import tomllib
import pickle
from datasketch import MinHash

from indexes import (
    build_minhash,
    build_lsh_index,
    build_lshforest_index,
    build_lshensemble_index
)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)

lsh_params = config["lsh"]
num_perm = lsh_params["num_perm"]
threshold = lsh_params["threshold"]
forest_k = lsh_params["forest_k"]
ensemble_num_partitions = lsh_params["ensemble_num_partitions"]

tokens_path = config["paths"]["tokens"]
minhash_sig_path = config["paths"]["minhash_signatures"]


def create_and_store_minhash_signatures(
    tokens: dict[str, list[str]],
    minhash_path: Path
) -> dict[str, MinHash]:
    minhash_path.mkdir(parents=True, exist_ok=True)

    minhash_dict = {}
    for key, token_list in tokens.items():
        minhash = build_minhash(token_list, num_perm)
        minhash_dict[key] = minhash

    with open(minhash_path / "minhashes.pkl", "wb") as f:
        pickle.dump(minhash_dict, f)

    return minhash_dict


if __name__ == "__main__":
    # parse tokens from JSON file
    json_paths = [
        Path(tokens_path) / "k_shingles" / "en_k_shingles.json",
        Path(tokens_path) / "k_shingles" / "jp_k_shingles.json",
        Path(tokens_path) / "words" / "en_words.json",
        Path(tokens_path) / "words" / "jp_words.json"
    ]
    with open(json_paths[0], "r") as f:
        en_k_shingles = json.load(f)
    with open(json_paths[1], "r") as f:
        jp_k_shingles = json.load(f)
    with open(json_paths[2], "r") as f:
        en_words = json.load(f)
    with open(json_paths[3], "r") as f:
        jp_words = json.load(f)

    create_and_store_minhash_signatures(en_k_shingles, Path(minhash_sig_path))
    create_and_store_minhash_signatures(jp_k_shingles, Path(minhash_sig_path))
    create_and_store_minhash_signatures(en_words, Path(minhash_sig_path))
    create_and_store_minhash_signatures(jp_words, Path(minhash_sig_path))
