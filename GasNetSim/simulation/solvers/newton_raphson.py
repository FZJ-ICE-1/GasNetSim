import numpy as np
import logging

from .base_solver import Solver, SolveResult

logger = logging.getLogger(__name__)


class NewtonRaphsonSolver(Solver):
    """
    Hand-rolled Newton-Raphson solver with optional under-relaxation.

    At each iteration:
      1. Evaluate Jacobian J(x) and residual F(x)   (one _compute call)
      2. Solve the linear system: J dx = F
      3. Update: x ← x + dx / underrelaxation_factor

    Convergence is declared when max|F(x)| ≤ tol before a step is taken.
    """

    def __init__(self, underrelaxation_factor: float = 2.0):
        self.underrelaxation_factor = underrelaxation_factor

    def solve(self, residual_fn, jacobian_fn, x0, tol, max_iter):
        x = x0.copy()

        for n_iter in range(max_iter + 1):
            # Jacobian is evaluated first so the formulation can cache both
            # J and F from the single _compute(x) call, making residual_fn
            # a free cache hit.
            J = jacobian_fn(x)
            r = residual_fn(x)

            err = float(np.max(np.abs(r)))
            logger.debug(f"Newton iteration {n_iter}: max|F| = {err:.3e}")

            if err <= tol:
                return SolveResult(
                    x=x,
                    converged=True,
                    n_iter=n_iter,
                    message=f"Converged in {n_iter} iteration(s)",
                )

            if n_iter == max_iter:
                break

            dx = np.linalg.solve(J, r) / self.underrelaxation_factor
            x = x + dx

        return SolveResult(
            x=x,
            converged=False,
            n_iter=max_iter,
            message=f"Did not converge in {max_iter} iteration(s)",
        )
