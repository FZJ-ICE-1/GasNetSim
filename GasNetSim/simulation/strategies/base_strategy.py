from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    Abstract base class for simulation strategies.

    A strategy controls how the pressure and composition sub-problems are
    coupled and iterated to reach a converged solution.  It owns the
    top-level algorithm (direct Newton, staggered Picard, …) while
    delegating numerical details to a Problem and a Solver.
    """

    @abstractmethod
    def solve(
        self,
        network,
        *,
        max_iter: int,
        tol: float,
        underrelaxation_factor: float,
        use_cuda: bool,
        sparse_matrix: bool,
        tracking_method: str,
        time_step: int,
    ):
        """
        Run the simulation strategy on *network* in-place and return it.
        """
