import numpy as np
from scipy import optimize

from .base_solver import Solver, SolveResult


class ScipySolver(Solver):
    """
    Thin wrapper around scipy.optimize.root.

    Useful for benchmarking against the hand-rolled Newton-Raphson or for
    accessing scipy's globalization strategies (Krylov-Newton, Levenberg-
    Marquardt, etc.) without changing the formulation.

    Parameters
    ----------
    method : str
        One of scipy's root-finding methods: 'hybr' (default), 'lm',
        'broyden1', 'broyden2', 'anderson', 'linearmixing',
        'diagbroyden', 'excitingmixing', 'krylov', 'df-sane'.
        Methods 'hybr' and 'lm' use the Jacobian; others do not.
    options : dict, optional
        Passed directly to scipy.optimize.root as the `options` argument.
    **root_kwargs
        Any additional keyword arguments forwarded to scipy.optimize.root.

    Notes
    -----
    scipy.optimize.root does not distinguish between "max residual" and
    "function norm" tolerances consistently across methods. The `tol`
    argument here is passed as the `tol` keyword to root(), whose exact
    meaning is method-dependent — check scipy docs for the chosen method.
    """

    def __init__(self, method: str = "hybr", options: dict = None, **root_kwargs):
        self.method = method
        self.options = options or {}
        self.root_kwargs = root_kwargs

    def solve(self, residual_fn, jacobian_fn, x0, tol, max_iter):
        # Methods that use the Jacobian
        uses_jacobian = self.method in ("hybr", "lm")

        result = optimize.root(
            residual_fn,
            x0,
            jac=jacobian_fn if uses_jacobian else None,
            method=self.method,
            tol=tol,
            options={"maxfev": max_iter, **self.options},
            **self.root_kwargs,
        )

        return SolveResult(
            x=result.x,
            converged=result.success,
            n_iter=result.nfev,
            message=result.message,
        )
