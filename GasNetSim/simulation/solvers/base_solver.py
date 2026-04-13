from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    n_iter: int
    message: str


class Solver(ABC):
    """
    Abstract base class for nonlinear solvers.

    A solver takes a residual function F(x) and Jacobian J(x) and finds
    x such that F(x) = 0, starting from an initial guess x0.

    Solvers are decoupled from physics — they know nothing about gas networks,
    pressures, or flows. All physical logic lives in the Formulation.
    """

    @abstractmethod
    def solve(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        jacobian_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        tol: float,
        max_iter: int,
    ) -> SolveResult:
        """
        Find x such that residual_fn(x) ≈ 0.

        Parameters
        ----------
        residual_fn : callable
            F(x) → residual vector, same shape as x.
        jacobian_fn : callable
            J(x) → Jacobian matrix, shape (len(x), len(x)).
            When the formulation caches its last computation, calling
            jacobian_fn after residual_fn (or vice versa) for the same x
            is free.
        x0 : np.ndarray
            Initial guess.
        tol : float
            Convergence tolerance on max(|F(x)|).
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        SolveResult
        """
