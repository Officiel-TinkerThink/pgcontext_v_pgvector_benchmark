"""Workload vectors and the seeded replication used by every benchmark (identical to the original lab run).

data/workload_e5_1024.npz: `base` 1,069 x 1024 e5-large embeddings of a synthetic company corpus, `queries` 60 x 1024
question embeddings (see data/README.md). make_vectors() is the lab's generator verbatim: sample base rows with a
fixed seed, add gaussian noise (sigma 0.02 per dimension), keep the real rows first, L2-normalise.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "workload_e5_1024.npz"
NOISE_SIGMA = 0.02
SEED = 7
PROJECTION_SEED = 11


@lru_cache(maxsize=1)
def workload() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(DATA)
    return z["base"].astype(np.float32), z["queries"].astype(np.float32)


def synthetic_workload(n_base: int = 1069, n_queries: int = 60, dim: int = 1024, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Data-free alternative (--synthetic): isotropic random directions, queries near random base rows."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_base, dim)).astype(np.float32)
    picks = rng.integers(0, n_base, size=n_queries)
    queries = base[picks] + rng.normal(0, 0.5, size=(n_queries, dim)).astype(np.float32)
    return base, queries.astype(np.float32)


def make_vectors(base: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(base), size=n)
    v = base[idx] + rng.normal(0, NOISE_SIGMA, size=(n, base.shape[1])).astype(np.float32)
    v[: len(base)] = base[: min(n, len(base))]             # the real vectors are always present
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def exact_topk(vecs: np.ndarray, queries: np.ndarray, k: int = 10) -> list[list[int]]:
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    sims = q @ vecs.T
    return [list(np.argsort(-row)[:k]) for row in sims]


def project(vecs: np.ndarray, dim: int = 384) -> np.ndarray:
    """Seeded gaussian random projection (dimension probe: same rows, fewer dimensions), L2-normalised."""
    rng = np.random.default_rng(PROJECTION_SEED)
    r = rng.normal(size=(vecs.shape[1], dim)).astype(np.float32)
    p = vecs @ r
    return (p / np.linalg.norm(p, axis=1, keepdims=True)).astype(np.float32)
