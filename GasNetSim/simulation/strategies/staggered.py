import copy
import logging

import numpy as np

from .base_strategy import Strategy
from .direct import DirectStrategy

logger = logging.getLogger(__name__)


class StaggeredStrategy(Strategy):
    """
    Staggered (outer-Picard) coupling strategy.

    Alternates between two sub-steps until both pressure and composition
    converge:

    1. **Pressure step** — freeze composition, run a full direct Newton
       solve (``DirectStrategy`` with ``tracking_method="no_mixing"``).
    2. **Composition step** — freeze pressure, propagate compositions
       topologically (``network.composition_initialization``), then apply
       relaxed blending toward the new target.

    Parameters
    ----------
    solver : Solver, optional
        Inner Solver passed to the DirectStrategy pressure step.
        Defaults to NewtonRaphsonSolver.
    coupling_max_iter : int
        Maximum number of outer Picard iterations.
    coupling_tol : float
        Convergence tolerance on the composition change (max absolute
        difference across all nodal compositions).
    outer_pressure_tol : float, optional
        Convergence tolerance on the pressure change between outer
        iterations (Pa).  Defaults to ``max(10.0, tol * 1e5)``.
    composition_relaxation_factor : float
        Under-relaxation factor applied when blending compositions
        between outer iterations (1.0 = no relaxation).
    """

    def __init__(
        self,
        solver=None,
        coupling_max_iter=20,
        coupling_tol=1e-4,
        outer_pressure_tol=None,
        composition_relaxation_factor=0.5,
    ):
        self.solver = solver
        self.coupling_max_iter = coupling_max_iter
        self.coupling_tol = coupling_tol
        self.outer_pressure_tol = outer_pressure_tol
        self.composition_relaxation_factor = composition_relaxation_factor

    def solve(
        self,
        network,
        *,
        max_iter,
        tol,
        underrelaxation_factor,
        use_cuda,
        sparse_matrix,
        tracking_method,
        time_step,
    ):
        # no_mixing needs no composition coupling — delegate directly.
        if tracking_method == "no_mixing":
            return DirectStrategy(solver=self.solver).solve(
                network,
                max_iter=max_iter,
                tol=tol,
                underrelaxation_factor=underrelaxation_factor,
                use_cuda=use_cuda,
                sparse_matrix=sparse_matrix,
                tracking_method=tracking_method,
                time_step=time_step,
            )

        if self.coupling_max_iter < 1:
            raise ValueError("coupling_max_iter must be at least 1.")

        outer_pressure_tol = self.outer_pressure_tol
        if outer_pressure_tol is None:
            outer_pressure_tol = max(10.0, tol * 1e5)

        inner_strategy = DirectStrategy(solver=self.solver)
        common = dict(
            max_iter=max_iter,
            tol=tol,
            underrelaxation_factor=underrelaxation_factor,
            use_cuda=use_cuda,
            sparse_matrix=sparse_matrix,
            time_step=time_step,
        )

        # Save and restore network initialisation flags across outer iters.
        original_run_initialization = network.run_initialization
        original_pressure_prev = copy.deepcopy(network.pressure_prev)
        original_run_composition_initialization = (
            network.run_composition_initialization
        )

        previous_pressure = None

        try:
            for outer_iter in range(1, self.coupling_max_iter + 1):
                if outer_iter > 1:
                    network.run_initialization = False
                    network.pressure_prev = network.save_pressure_values()

                # --- Pressure step (composition frozen) ---
                network.run_composition_initialization = False
                inner_strategy.solve(
                    network, tracking_method="no_mixing", **common
                )

                current_pressure = np.array(
                    network.save_pressure_values(), copy=True
                )
                previous_composition = network._snapshot_nodal_eos_compositions()

                # --- Composition step (pressure frozen) ---
                network.run_composition_initialization = True
                if tracking_method == "batch_tracking" and network.pipelines:
                    cached_batch_information = {
                        i: (
                            p.batch_location_history.copy(),
                            p.composition_history.copy(),
                        )
                        for i, p in network.pipelines.items()
                    }
                else:
                    cached_batch_information = {}
                network.composition_initialization(
                    tracking_method=tracking_method,
                    time_step=time_step,
                    cached_batch_information=cached_batch_information,
                )
                target_composition = network._snapshot_nodal_eos_compositions()
                composition_delta = network._apply_relaxed_nodal_compositions(
                    previous_composition,
                    target_composition,
                    self.composition_relaxation_factor,
                )

                pressure_delta = (
                    np.inf
                    if previous_pressure is None
                    else float(np.max(np.abs(current_pressure - previous_pressure)))
                )

                logger.info(
                    "Staggered coupling iteration %d/%d: "
                    "pressure delta %.3f Pa, composition delta %.3e",
                    outer_iter,
                    self.coupling_max_iter,
                    pressure_delta,
                    composition_delta,
                )

                if (
                    previous_pressure is not None
                    and pressure_delta <= outer_pressure_tol
                    and composition_delta <= self.coupling_tol
                ):
                    logger.info(
                        "Staggered coupling converges in %d iteration(s).",
                        outer_iter,
                    )
                    return network

                previous_pressure = current_pressure

        finally:
            network.run_initialization = original_run_initialization
            network.pressure_prev = original_pressure_prev
            network.run_composition_initialization = (
                original_run_composition_initialization
            )

        raise RuntimeError(
            f"Staggered coupling not converged in {self.coupling_max_iter} "
            "iteration(s)!"
        )
