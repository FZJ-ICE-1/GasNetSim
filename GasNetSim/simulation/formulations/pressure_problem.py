import numpy as np
import logging

from .base_problem import Problem
from .composition_tracker import CompositionTracker
from GasNetSim.components.utils.cuda_support import list_to_array

logger = logging.getLogger(__name__)


class PressureProblem(Problem):
    """
    Steady-state gas network problem with nodal pressures as unknowns.

    State vector x
    --------------
    Junction node pressures (Pa), in simulation-index order, excluding
    non-junction (reference / shortpipe-inlet) nodes whose pressures are fixed.

    Residual F(x)
    -------------
    Nodal flow balance at each junction node:
        F_i = f_target_i - sum_j Q_ij(p)
    where Q_ij is the flow from node i to node j computed from the current
    pressure state.

    Jacobian J(x)
    -------------
    ∂F/∂p assembled by Pipeline/Resistance/Compressor via
    Network.jacobian_matrix().

    Caching
    -------
    Each call to residual(x) or jacobian(x) triggers _compute(x), which does
    the full evaluation: property updates → jacobian_matrix() → composition
    tracking → residual assembly. Results are cached by x so that the second
    call (same x) is a free lookup. This matters for Newton-Raphson which
    calls jacobian then residual for the same x every iteration.
    """

    def __init__(
        self,
        network,
        use_cuda: bool = False,
        sparse_matrix: bool = False,
        tracking_method: str = "simple_mixing",
        time_step: int = 3600,
        run_composition_tracking: bool = True,
    ):
        self.network = network
        self.use_cuda = use_cuda
        self.sparse_matrix = sparse_matrix
        self.tracking_method = tracking_method
        self.time_step = time_step
        # When False, composition is frozen during Newton iterations and
        # initial_state skips the topological composition seeding. Used by
        # the staggered coupler, which owns composition updates.
        self.run_composition_tracking = run_composition_tracking

        # Derived index lists (computed once from the fixed network topology)
        non_junction_ids = set(network.non_junction_nodes)
        n = network.get_simulation_node_count()
        self._junction_sim_indices = [
            i for i in range(n)
            if network.simulation_node_index_to_node_id(i) not in non_junction_ids
        ]
        self._non_junction_ids = non_junction_ids

        # Internal state updated during iterations
        self._f_target = None
        self._t = None

        # Batch tracking snapshot — taken once so CompositionTracker can
        # restore pipeline history before each Newton iteration.
        if tracking_method == "batch_tracking" and network.pipelines is not None:
            cached_batch_information = {
                i: (p.batch_location_history.copy(), p.composition_history.copy())
                for i, p in network.pipelines.items()
            }
        else:
            cached_batch_information = {}

        self._composition_tracker = CompositionTracker(
            network=network,
            tracking_method=tracking_method,
            time_step=time_step,
            cached_batch_information=cached_batch_information,
        )

        # Cache for last _compute(x) result
        self._cache_key: bytes = b""
        self._cache_r: np.ndarray = None
        self._cache_J = None

    # ------------------------------------------------------------------
    # Problem interface
    # ------------------------------------------------------------------

    def initial_state(self) -> np.ndarray:
        """
        Initialise network parameters and return junction pressures as x0.
        """
        network = self.network
        init_f, init_p, init_t = network.newton_raphson_initialization()

        self._t = list_to_array(init_t, use_cuda=self.use_cuda)
        self._f_target = list_to_array(init_f, use_cuda=self.use_cuda)

        p_full = list_to_array(init_p, use_cuda=self.use_cuda)
        network.update_node_parameters(
            pressure=p_full, flow=self._f_target, temperature=self._t
        )
        if network.pipelines is not None:
            network.update_pipeline_parameters()
        if network.resistances is not None:
            network.update_resistance_parameters()
        if network.compressors is not None:
            network.update_compressor_parameters()

        if self.run_composition_tracking:
            network.composition_initialization(
                tracking_method=self.tracking_method,
                time_step=self.time_step,
                cached_batch_information=self._composition_tracker.cached_batch_information,
            )

        return self._full_p_to_x(p_full)

    def residual(self, x: np.ndarray) -> np.ndarray:
        r, _ = self._compute(x)
        return r

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        _, J = self._compute(x)
        return J

    def unpack(self, x: np.ndarray) -> None:
        """Write converged junction pressures back and update derived quantities."""
        network = self.network

        # Update all node pressures
        for node_id in network.nodes:
            if node_id not in self._non_junction_ids:
                sim_idx = network.node_id_to_simulation_node_index(node_id)
                k = self._junction_sim_indices.index(sim_idx)
                network.nodes[node_id].pressure = x[k]

        # Recompute flow matrix to get supply/demand at non-junction nodes
        _, f_mat = network.jacobian_matrix(
            use_cuda=self.use_cuda, sparse_matrix=self.sparse_matrix
        )
        if self.use_cuda:
            import cupy as cp
            nodal_flow = cp.sum(f_mat, axis=1)
        else:
            nodal_flow = np.sum(f_mat, axis=1)

        for node_id in self._non_junction_ids:
            sim_idx = network.node_id_to_simulation_node_index(node_id)
            network.nodes[node_id].volumetric_flow = nodal_flow[sim_idx]
            node = network.nodes[node_id]
            if node.flow_type == "volumetric":
                node.convert_volumetric_to_energy_flow()
            elif node.flow_type == "energy":
                node.convert_energy_to_volumetric_flow()

        # Finalise gas composition (promote tmp → permanent)
        for node in network.nodes.values():
            node.gas_mixture.eos_composition = node.gas_mixture.eos_composition_tmp
            node.gas_mixture.convert_eos_composition_to_dictionary()

        network.update_connection_flow_rate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute(self, x: np.ndarray):
        """
        Core evaluation: update network state for junction pressures x, then
        return (residual, jacobian). Results are cached by x.tobytes() so
        that the second call with the same x is free.
        """
        cache_key = x.tobytes()
        if cache_key == self._cache_key:
            return self._cache_r, self._cache_J

        network = self.network

        # 1. Write junction pressures into network nodes and update all
        #    gas-mixture properties (compressibility, viscosity, etc.)
        p_full = self._x_to_full_p(x)
        network.update_node_parameters(
            pressure=p_full, flow=self._f_target, temperature=self._t
        )
        if network.pipelines is not None:
            network.update_pipeline_parameters()
        if network.resistances is not None:
            network.update_resistance_parameters()
        if network.compressors is not None:
            network.update_compressor_parameters()

        for connection in network.connections.values():
            connection.inlet = network.nodes[connection.inlet_index]
            connection.outlet = network.nodes[connection.outlet_index]

        # 2. Assemble Jacobian and flow matrix
        j_mat, f_mat = network.jacobian_matrix(
            use_cuda=self.use_cuda, sparse_matrix=self.sparse_matrix
        )

        # 3. Composition tracking (single-pass, cached topological order)
        network.update_connection_flow_rate()
        if self.run_composition_tracking:
            self._composition_tracker.update()

        # 4. Compute nodal flow balance residual
        if self.use_cuda:
            import cupy as cp
            nodal_flow = cp.sum(f_mat, axis=1)
        else:
            nodal_flow = np.sum(f_mat, axis=1)

        # Update f_target with latest flow-type conversions
        for node in network.nodes.values():
            if node.flow_type == "volumetric":
                node.convert_volumetric_to_energy_flow()
            elif node.flow_type == "energy":
                node.convert_energy_to_volumetric_flow()

        self._f_target = list_to_array(
            [
                node.volumetric_flow if node.volumetric_flow is not None else 0.0
                for node in network.nodes.values()
            ],
            use_cuda=self.use_cuda,
        )

        delta_flow = self._f_target - nodal_flow
        residual = list_to_array(
            [
                delta_flow[i]
                for i in range(len(delta_flow))
                if network.simulation_node_index_to_node_id(i) not in self._non_junction_ids
            ],
            use_cuda=self.use_cuda,
        )

        self._cache_key = cache_key
        self._cache_r = residual
        self._cache_J = j_mat

        return residual, j_mat

    def _full_p_to_x(self, p_full) -> np.ndarray:
        """Extract junction pressures from the full pressure vector."""
        return np.array([p_full[i] for i in self._junction_sim_indices])

    def _x_to_full_p(self, x: np.ndarray) -> np.ndarray:
        """Build the full pressure vector from junction pressures x."""
        network = self.network
        n = network.get_simulation_node_count()
        p_full = np.zeros(n)
        for k, sim_idx in enumerate(self._junction_sim_indices):
            p_full[sim_idx] = x[k]
        # Fixed pressures for non-junction nodes
        for node_id in self._non_junction_ids:
            sim_idx = network.node_id_to_simulation_node_index(node_id)
            p_full[sim_idx] = network.nodes[node_id].pressure
        return p_full
