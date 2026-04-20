import logging

from GasNetSim.simulation.formulations import PressureProblem
from GasNetSim.simulation.solvers import NewtonRaphsonSolver

from .base_strategy import Strategy

logger = logging.getLogger(__name__)


class DirectStrategy(Strategy):
    """
    Single-pass Newton strategy: solves pressure and composition together
    in one Newton loop.

    PressureProblem handles composition tracking inside each residual
    evaluation, so pressure and composition are updated every iteration.

    Parameters
    ----------
    problem : Problem, optional
        Custom Problem instance.  If None, a PressureProblem is built
        from the network and the solve-time configuration.
    solver : Solver, optional
        Custom Solver instance.  If None, NewtonRaphsonSolver is used.
    """

    def __init__(self, problem=None, solver=None):
        self.problem = problem
        self.solver = solver

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
        problem = self.problem
        solver = self.solver

        if problem is None:
            problem = PressureProblem(
                network,
                use_cuda=use_cuda,
                sparse_matrix=sparse_matrix,
                tracking_method=tracking_method,
                time_step=time_step,
            )
        if solver is None:
            solver = NewtonRaphsonSolver(
                underrelaxation_factor=underrelaxation_factor
            )

        x0 = problem.initial_state()
        result = solver.solve(
            residual_fn=problem.residual,
            jacobian_fn=problem.jacobian,
            x0=x0,
            tol=tol,
            max_iter=max_iter,
        )

        if not result.converged:
            raise RuntimeError(
                f"Simulation not converged in {result.n_iter} iteration(s)!"
            )

        logger.info("Simulation converges in %d iteration(s).", result.n_iter)
        problem.unpack(result.x)

        for i_connection, connection in network.connections.items():
            logger.debug("Pipeline index: %s", i_connection)
            logger.debug("Pipeline flow rate: %s", connection.flow_rate)
            logger.debug(
                "Gas mixture composition: %s",
                connection.gas_mixture.composition,
            )

        return network
