from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(repr=False)
class InterventionData:
    """Observations from multiple experimental environments.

    X: shape (n, p), without an intercept column.
    Y: shape (n,).
    environments: shape (n,), one label per observation.
    Causals: 0-based predictor indices of the true direct causes of Y.
    """

    X: np.ndarray
    Y: np.ndarray
    environments: np.ndarray
    Causals: tuple[int, ...]

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.Y = np.asarray(self.Y, dtype=float)
        self.environments = np.asarray(self.environments)

        if self.X.ndim != 2:
            raise ValueError("X must be a two-dimensional array (n, p).")
        if self.Y.ndim == 2 and self.Y.shape[1] == 1:
            self.Y = self.Y[:, 0]
        if self.Y.ndim != 1:
            raise ValueError("Y must be a vector of shape (n,) or (n, 1).")
        if self.environments.ndim != 1:
            raise ValueError("environments must be a one-dimensional vector.")

        n, p = self.X.shape
        if n == 0:
            raise ValueError("The dataset must contain at least one observation.")
        if len(self.Y) != n or len(self.environments) != n:
            raise ValueError("X, Y and environments must have the same number of rows.")
        if not np.isfinite(self.X).all() or not np.isfinite(self.Y).all():
            raise ValueError("X and Y must contain only finite values.")
        if self.environments.dtype.kind not in "biufUS":
            raise TypeError("Environment labels must be numbers or strings of one consistent type.")
        if self.environments.dtype.kind == "f" and not np.isfinite(self.environments).all():
            raise ValueError("Environment labels cannot be NaN or infinite.")

        indices = tuple(self.Causals)
        if any(not isinstance(j, (int, np.integer)) or isinstance(j, (bool, np.bool_))
               for j in indices):
            raise TypeError("Causals must contain integer predictor indices.")
        self.Causals = tuple(int(j) for j in indices)
        if len(set(self.Causals)) != len(self.Causals):
            raise ValueError("Causals cannot contain duplicate indices.")
        if any(j < 0 or j >= p for j in self.Causals):
            raise ValueError("Every index in Causals must be between 0 and p - 1.")

    @classmethod
    def from_environment_blocks(
        cls,
        blocks: Mapping[object, tuple[np.ndarray, np.ndarray]],
        Causals: tuple[int, ...],
    ) -> "InterventionData":
        """Construct from {environment_label: (X_e, Y_e)}."""
        if not blocks:
            raise ValueError("At least one environment is required.")

        X_parts, Y_parts, labels = [], [], []
        for e, (X_e, Y_e) in blocks.items():
            X_e = np.asarray(X_e, dtype=float)
            Y_e = np.asarray(Y_e, dtype=float)
            if Y_e.ndim == 2 and Y_e.shape[1] == 1:
                Y_e = Y_e[:, 0]
            if X_e.ndim != 2 or Y_e.ndim != 1 or X_e.shape[0] != len(Y_e):
                raise ValueError(f"Mismatched X and Y shapes for environment {e!r}.")
            if len(Y_e) == 0:
                raise ValueError(f"Environment {e!r} cannot be empty.")
            X_parts.append(X_e)
            Y_parts.append(Y_e)
            labels.extend([e] * len(Y_e))

        return cls(
            X=np.vstack(X_parts),
            Y=np.concatenate(Y_parts),
            environments=np.asarray(labels),
            Causals=Causals,
        )

    def __repr__(self):
        return (f"InterventionData(n={self.n}, p={self.p}, "
                f"environments={self.E}, Causals={self.Causals})")

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def E(self) -> tuple:
        return tuple(np.unique(self.environments).tolist())

    @property
    def n_by_environment(self) -> dict:
        labels, counts = np.unique(self.environments, return_counts=True)
        return dict(zip(labels.tolist(), counts.tolist()))

    def _columns(self, S):
        if S is None:
            return list(range(self.p))
        S = tuple(S)
        if any(not isinstance(j, (int, np.integer)) or isinstance(j, (bool, np.bool_))
               for j in S):
            raise TypeError("S must contain integer predictor indices.")
        if len(set(S)) != len(S) or any(j < 0 or j >= self.p for j in S):
            raise ValueError("S must consist of distinct indices from 0 to p - 1.")
        return list(S)

    def _extract(self, mask, S=None, add_intercept=False):
        X = self.X[mask][:, self._columns(S)]
        Y = self.Y[mask]
        if add_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        return X, Y

    def get_all(self, S=None, add_intercept=False):
        """Return pooled observations, optionally selecting predictors S."""
        return self._extract(slice(None), S, add_intercept)

    def get_environment(self, e, S=None, add_intercept=False):
        """Return (X_e,S, Y_e) for the observations in environment e."""
        if e not in self.E:
            raise ValueError(f"Unknown environment {e!r}; available: {self.E}.")
        return self._extract(self.environments == e, S, add_intercept)

    def get_complement(self, e, S=None, add_intercept=False):
        """Return (X_-e,S, Y_-e), pooling all environments except e."""
        if e not in self.E:
            raise ValueError(f"Unknown environment {e!r}; available: {self.E}.")
        return self._extract(self.environments != e, S, add_intercept)
