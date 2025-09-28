from datasketch import MinHash, MinHashLSH, MinHashLSHForest, MinHashLSHEnsemble


def build_minhash(tokens: list[str], num_perm: int = 128) -> MinHash:
    """
    Build a MinHash signature from a list of tokens.
    """
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m


def build_lsh_index(
    minhash_dict: dict[str, MinHash],
    threshold: float = 0.5,
    num_perm: int = 128
) -> MinHashLSH:
    """Builds a MinHashLSH index."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for key, mh in minhash_dict.items():
        lsh.insert(key, mh)
    return lsh


def build_lshforest_index(
    minhash_dict: dict[str, MinHash],
    num_perm: int = 128,
    l: int = 8
) -> MinHashLSHForest:
    """
    Builds a MinHashLSHForest index.
    Args:
      l: number of prefix trees (default 8)
    """
    forest = MinHashLSHForest(num_perm=num_perm, l=l)
    for key, mh in minhash_dict.items():
        forest.add(key, mh)
    forest.index()
    return forest


def build_lshensemble_index(
    minhash_dict: dict[str, MinHash],
    size_dict: dict[str, int],
    threshold: float = 0.5,
    num_perm: int = 128,
    num_partitions: int = 16,
    m: int = 8
) -> MinHashLSHEnsemble:
    """
    Builds a MinHashLSHEnsemble index.
    Args:
      num_partitions: number of partitions in the ensemble
      m: memory usage factor, improves accuracy at cost of memory
    """
    ensemble = MinHashLSHEnsemble(
        threshold=threshold,
        num_perm=num_perm,
        num_part=num_partitions,
        m=m
    )
    entries = [(key, mh, size_dict[key]) for key, mh in minhash_dict.items()]
    ensemble.index(entries)
    return ensemble
