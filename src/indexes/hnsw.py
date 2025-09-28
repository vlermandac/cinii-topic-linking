from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class HNSWWrapper:
    """
    Thin wrapper around a FAISS HNSW index that:
      - Stores the embedding model
      - Stores the full embeddings matrix
      - Provides a method to fetch an embedding by id/URI
      - Exposes query helpers (by URI or by raw text)
    """

    def __init__(
        self,
        texts: list[str],
        uris: list[str],
        emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_cosine: bool = True,
        batch_size: int = 256,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        use_gpu: bool = False,
    ) -> None:
        if len(texts) != len(uris):
            raise ValueError("texts and uris must have the same length")

        self.use_cosine = use_cosine
        self.uris: list[str] = list(uris)
        self.uri2row: dict[str, int] = {u: i for i, u in enumerate(self.uris)}

        # model and embeddings
        self.model = SentenceTransformer(emb_model_name)
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.use_cosine,
        ).astype(np.float32)

        dim = int(self.embeddings.shape[1])
        metric = faiss.METRIC_INNER_PRODUCT if self.use_cosine else faiss.METRIC_L2

        # HNSW index
        index = faiss.IndexHNSWFlat(dim, hnsw_m, metric)
        index.hnsw.efConstruction = int(ef_construction)
        index.hnsw.efSearch = int(ef_search)

        self._gpu_res = None
        if use_gpu:
            try:
                self._gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
            except Exception:
                self._gpu_res = None  # fallback to CPU

        index.add(self.embeddings)
        self.index: faiss.Index = index

    def set_ef_search(self, ef_search: int) -> None:
        """Increase ef_search for higher recall (slower search)."""
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = int(ef_search)

    def get_embedding(self, id_like: str | int) -> np.ndarray:
        """Return a copy of the embedding vector for a given URI or row id."""
        if isinstance(id_like, str):
            if id_like not in self.uri2row:
                raise KeyError(f"Unknown URI: {id_like}")
            row = self.uri2row[id_like]
        else:
            row = id_like
            if row < 0 or row >= len(self.uris):
                raise IndexError(f"Row id out of range: {row}")

        return self.embeddings[row].copy()

    def query_by_uri(
        self,
        query_uri: str,
        topk: int = 10,
        return_scores: bool = False,
        exclude_self: bool = True,
    ) -> list[str] | list[tuple[str, float]]:
        """Search using a document already in the corpus."""
        if query_uri not in self.uri2row:
            raise KeyError(f"Unknown URI: {query_uri}")

        qi = self.uri2row[query_uri]
        q = self.embeddings[qi: qi + 1]
        scores, idx = self.index.search(q, topk + (1 if exclude_self else 0))

        idx_list = idx[0].tolist()
        scs_list = scores[0].tolist()

        out: list = []
        for j, s in zip(idx_list, scs_list):
            if j < 0:
                continue
            if exclude_self and j == qi:
                continue
            u = self.uris[j]
            out.append((u, float(s)) if return_scores else u)
            if len(out) >= topk:
                break
        return out

    def query_by_text(
        self,
        text: str,
        topk: int = 10,
        return_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        """Search using a new text (not necessarily in the index)."""
        q_emb = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.use_cosine,
        ).astype(np.float32)

        scores, idx = self.index.search(q_emb, topk)
        idx_list = idx[0].tolist()
        scs_list = scores[0].tolist()

        out: list = []
        for j, s in zip(idx_list, scs_list):
            if j < 0:
                continue
            u = self.uris[j]
            out.append((u, float(s)) if return_scores else u)
        return out

    def save(self, index_path: str, npy_path: str | None = None) -> None:
        """Save index and optionally embeddings."""
        idx_cpu = self.index
        if hasattr(faiss, "index_gpu_to_cpu") and isinstance(self.index, faiss.GpuIndex):
            idx_cpu = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(idx_cpu, index_path)
        if npy_path:
            np.save(npy_path, self.embeddings)

    @staticmethod
    def load(
        index_path: str,
        model_name: str,
        uris: list[str],
        embeddings_npy: str | None = None,
        use_cosine: bool = True,
        use_gpu: bool = False,
    ) -> HNSWWrapper:
        """Load a saved index + optional embeddings."""
        wrapper = object.__new__(HNSWWrapper)  # bypass __init__
        wrapper.use_cosine = use_cosine
        wrapper.model = SentenceTransformer(model_name)
        wrapper.uris = list(uris)
        wrapper.uri2row = {u: i for i, u in enumerate(wrapper.uris)}

        idx = faiss.read_index(index_path)
        wrapper._gpu_res = None
        if use_gpu:
            try:
                wrapper._gpu_res = faiss.StandardGpuResources()
                idx = faiss.index_cpu_to_gpu(wrapper._gpu_res, 0, idx)
            except Exception:
                wrapper._gpu_res = None
        wrapper.index = idx

        if embeddings_npy:
            wrapper.embeddings = np.load(embeddings_npy).astype(np.float32)
        else:
            wrapper.embeddings = None  # type: ignore

        return wrapper
