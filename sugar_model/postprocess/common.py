"""Shared helpers: units, logging, mesh adjacency and connected components."""
from __future__ import annotations

import logging
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

log = logging.getLogger("cad_clean")

# how many output units one millimetre is
UNIT_PER_MM = {"mm": 1.0, "cm": 0.1, "m": 0.001, "in": 1.0 / 25.4}


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="[cad_clean %(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")


def face_normals_areas(V: np.ndarray, F: np.ndarray):
    """Unit face normals, face areas and centroids."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(area2, 1e-30)[:, None]
    return n, 0.5 * area2, (a + b + c) / 3.0


def face_adjacency(F: np.ndarray) -> np.ndarray:
    """(k,2) pairs of faces sharing an edge (manifold or not)."""
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    fid = np.repeat(np.arange(len(F)), 3)
    key = e[:, 0].astype(np.int64) * (int(F.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]
    return np.stack([fid[:-1][same], fid[1:][same]], axis=1)


def components(n: int, pairs: np.ndarray, mask: np.ndarray | None = None):
    """Connected components of n nodes linked by pairs (optionally only masked nodes).

    Returns labels (n,) with -1 for unmasked nodes.
    """
    if mask is None:
        mask = np.ones(n, bool)
    if len(pairs):
        keep = mask[pairs[:, 0]] & mask[pairs[:, 1]]
        p = pairs[keep]
    else:
        p = np.zeros((0, 2), int)
    g = sp.coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    lab = lab.astype(np.int64)
    lab[~mask] = -1
    # compact labels
    u, inv = np.unique(lab[mask], return_inverse=True)
    out = -np.ones(n, np.int64)
    out[mask] = inv
    return out


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rotation R (proper) minimising sum w ||R p - q||^2 for direction vectors."""
    if w is None:
        w = np.ones(len(P))
    H = (P * w[:, None]).T @ Q
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
