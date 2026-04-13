from abc import ABC, abstractmethod
import numpy as np


class Problem(ABC):
    """
    Abstract base class that defines a nonlinear system F(x) = 0 for a solver.

    A Problem encapsulates all the physics — what the unknowns are, how to
    compute the residual and Jacobian, and how to write the solution back to
    the Network. The Solver knows nothing about gas networks; it only sees
    residual_fn and jacobian_fn.

    Subclass responsibilities
    -------------------------
    - Define the state vector x (e.g. junction pressures for PressureProblem,
      or [pressures, flows] for a future augmented formulation).
    - Implement residual(x) → F(x): the mismatch to be driven to zero.
    - Implement jacobian(x) → J(x): ∂F/∂x.
    - Implement initial_state() → x0: starting point for the solver.
    - Implement unpack(x): write the converged solution back to the Network.

    Performance note
    ----------------
    For Newton-Raphson the solver calls jacobian(x) and then residual(x) for
    the same x in each iteration. Subclasses should cache the result of their
    internal _compute(x) so that the second call is a free lookup.
    """

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Return the initial state vector x0."""

    @abstractmethod
    def residual(self, x: np.ndarray) -> np.ndarray:
        """Return F(x): the residual vector to be driven to zero."""

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return J(x): the Jacobian matrix ∂F/∂x."""

    @abstractmethod
    def unpack(self, x: np.ndarray) -> None:
        """Write the converged solution x back to the Network."""
