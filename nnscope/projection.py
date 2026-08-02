"""Temporally stable 2D projection of high-dimensional embeddings.

Naively running PCA on every frame produces a useless animation. PCA's basis
is sign- and rotation-ambiguous, so consecutive frames come back mirrored or
spun around and the scatter plot strobes. The real structure is buried under
basis churn.

``StableProjector`` fixes this by treating the basis as a quantity that
persists across frames: each new basis is rotated onto the previous one via
orthogonal Procrustes, then blended in with momentum. What you see moving on
screen is the embedding actually moving, not the projection re-deciding which
way is up.

Scale is deliberately *not* normalized per frame. Embeddings genuinely spread
out as a network learns to separate classes, and that spread is most of the
story; the viewport smooths itself on the client instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Projection:
    """Result of projecting one batch of embeddings."""

    coords: np.ndarray
    """(n, 2) float32 projected coordinates."""

    explained_variance: float
    """Fraction of total variance captured by the two retained components."""

    rotation: float
    """Radians of basis rotation applied this frame, after alignment. Small
    values mean the projection is settled; large values mean the embedding
    geometry is still being reorganized."""


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis spanning the same subspace, via QR.

    QR is only unique up to the signs of Q's columns, and LAPACK is free to
    return either. Pinning R's diagonal positive makes the factorization the
    unique Gram-Schmidt one, which varies continuously with its input --
    without this the smoothing step would reintroduce exactly the flipping
    the Procrustes alignment just removed.
    """
    q, r = np.linalg.qr(basis)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def _procrustes_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal R minimizing ||source @ R - target||_F.

    Both inputs are (d, k) with orthonormal columns; the returned R is (k, k).
    Reflections are allowed, since a mirrored basis is just as valid a PCA
    solution and forcing a proper rotation would reintroduce flipping.
    """
    u, _, vt = np.linalg.svd(source.T @ target)
    return u @ vt


class StableProjector:
    """Projects (n, d) embeddings to (n, 2) with a basis that persists.

    Args:
        momentum: How fast the retained basis tracks each new frame's PCA.
            0 freezes the initial basis; 1 discards smoothing and uses the
            aligned basis directly. The default is tuned so a scatter plot
            updated ~10x/second reads as fluid rather than jittery.
    """

    def __init__(self, momentum: float = 0.15) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.momentum = momentum
        self._basis: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self._basis is not None

    def reset(self) -> None:
        """Forget the retained basis. The next frame starts a new alignment."""
        self._basis = None

    def fit_transform(self, embeddings: np.ndarray) -> Projection:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected a 2D (n, d) array, got shape {x.shape}")
        if x.shape[0] == 0:
            return Projection(np.zeros((0, 2), np.float32), 0.0, 0.0)

        centered = x - x.mean(axis=0, keepdims=True)

        # Fewer than two dimensions of input needs no projection at all; pad
        # the missing axis so downstream consumers always see (n, 2).
        if centered.shape[1] <= 2:
            padded = np.zeros((centered.shape[0], 2), dtype=np.float64)
            padded[:, : centered.shape[1]] = centered
            return Projection(padded.astype(np.float32), 1.0, 0.0)

        basis, explained = self._principal_basis(centered)
        rotation = 0.0

        if self._basis is not None and self._basis.shape == basis.shape:
            aligned = basis @ _procrustes_rotation(basis, self._basis)
            rotation = self._angle_between(aligned, self._basis)
            blended = (1.0 - self.momentum) * self._basis + self.momentum * aligned
            basis = _orthonormalize(blended)

        self._basis = basis
        coords = centered @ basis
        return Projection(coords.astype(np.float32), explained, rotation)

    def _principal_basis(self, centered: np.ndarray) -> tuple[np.ndarray, float]:
        """Top-2 right singular vectors and the variance they explain."""
        try:
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            # SVD very occasionally fails to converge on degenerate batches.
            # Reusing the previous basis is far better than dropping the frame.
            if self._basis is not None:
                return self._basis, 0.0
            raise

        total = float((singular**2).sum())
        explained = float((singular[:2] ** 2).sum() / total) if total > 0 else 0.0
        return vt[:2].T, explained

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
        """Principal angle between two 2-planes, in radians."""
        cosines = np.linalg.svd(a.T @ b, compute_uv=False)
        return float(np.arccos(np.clip(cosines.min(), -1.0, 1.0)))
