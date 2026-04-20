import logging

import networkx as nx
import numpy as np

from GasNetSim.components.pipeline import Pipeline
from GasNetSim.components.utils.utils import (
    batch_tracking,
    gas_composition_tracking,
)

logger = logging.getLogger(__name__)

_N_SPECIES = 21  # number of EOS composition entries


def _allclose_with_nan(a, b, rtol=1e-3, atol=1e-4):
    nan_equal = np.isnan(a) & np.isnan(b)
    close_equal = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    return bool(np.all(nan_equal | close_equal))


class CompositionTracker:
    """
    Tracks gas composition through the network during Newton-Raphson iterations.

    Responsibilities
    ----------------
    - Build and cache the topological edge order (based on flow directions).
    - Propagate composition through the network in a single pass per call,
      eliminating the while-loop in the original calculate_nodal_inflow_states.
    - Return the nodal composition matrix and update node.gas_mixture.eos_composition_tmp.

    Caching
    -------
    The topological order is rebuilt only when the sign of flow_velocity changes
    on any edge (i.e. a pipe reversal). For most Newton iterations the cached
    order is reused, saving one full NetworkX graph construction per iteration.

    Single-pass correctness
    -----------------------
    With edges processed in topological order, every edge upstream of a node is
    processed before any edge downstream of that node. After the last incoming
    edge to a node is processed, the node's composition is immediately finalised
    and available to downstream edges — so one pass is always sufficient.

    For batch_tracking the outflow composition depends on the stored batch
    history, not just the upstream node composition, so the same single-pass
    applies: each edge's batch state is advanced exactly once per call.
    """

    def __init__(self, network, tracking_method: str, time_step: int,
                 cached_batch_information: dict):
        self.network = network
        self.tracking_method = tracking_method
        self.time_step = time_step
        self.cached_batch_information = cached_batch_information  # may be {}

        # Cached topological edge order and the flow-direction fingerprint used
        # to detect when a pipe reversal requires a rebuild.
        self._cached_edge_order: list | None = None
        self._cached_flow_signs: dict = {}  # conn_id → -1 | 0 | 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> np.ndarray:
        """
        Propagate composition through the network in one topological pass.

        Resets node.gas_mixture.eos_composition_tmp to the current permanent
        composition before propagating, then overwrites it with the new
        flow-weighted result for each node that has inflow.

        Returns
        -------
        nodal_composition_matrix : np.ndarray, shape (n_species, n_nodes)
            Column i is the flow-weighted inlet composition at simulation node i.
            Columns with no inflow are NaN.
        """
        network = self.network

        # Reset tmp to the current permanent composition so that source nodes
        # (no incoming flow) keep their own composition.
        for node in network.nodes.values():
            node.gas_mixture.eos_composition_tmp = node.gas_mixture.eos_composition.copy()

        # Restore batch history snapshot before each pass so the batch state
        # isn't advanced multiple times within one Newton iteration.
        if self.tracking_method == "batch_tracking" and network.pipelines is not None:
            for i, pipeline in network.pipelines.items():
                pipeline.batch_location_history = self.cached_batch_information[i][0][:]
                pipeline.composition_history = self.cached_batch_information[i][1][:]

        edge_order = self._get_edge_order()

        if self.tracking_method == "simple_mixing":
            nodal_composition = self._propagate_simple_mixing(edge_order)
        elif self.tracking_method == "no_mixing":
            nodal_composition = self._propagate_no_mixing(edge_order)
        elif self.tracking_method == "batch_tracking":
            nodal_composition = self._propagate_batch_tracking(edge_order)
        else:
            raise ValueError(f"Unknown tracking method: {self.tracking_method!r}")

        # Write result into node tmp compositions.
        # Skip non-junction nodes (reference / shortpipe-inlet) — their
        # composition is defined by the supply source and must not be
        # overwritten by any transient inflow during Newton iterations.
        non_junction = network.non_junction_nodes
        for sim_idx in range(nodal_composition.shape[1]):
            col = nodal_composition[:, sim_idx]
            if not np.any(np.isnan(col)):
                node_id = network.simulation_node_index_to_node_id(sim_idx)
                if node_id in non_junction:
                    continue
                network.nodes[node_id].gas_mixture.eos_composition_tmp = col

        return nodal_composition

    # ------------------------------------------------------------------
    # Topological order (cached)
    # ------------------------------------------------------------------

    def _flow_sign(self, connection) -> int:
        v = connection.flow_velocity
        if v is None:
            return 0
        return 1 if v >= 0 else -1

    def _flow_directions_changed(self) -> bool:
        for conn_id, connection in self.network.connections.items():
            if self._cached_flow_signs.get(conn_id) != self._flow_sign(connection):
                return True
        return False

    def _get_edge_order(self) -> list:
        if self._cached_edge_order is None or self._flow_directions_changed():
            self._rebuild_edge_order()
        return self._cached_edge_order

    def _rebuild_edge_order(self):
        connections = self.network.connections
        G = nx.MultiDiGraph()
        edge_index = {}

        for conn_id, connection in connections.items():
            sign = self._flow_sign(connection)
            if sign >= 0:
                u, v = connection.inlet_index, connection.outlet_index
            else:
                u, v = connection.outlet_index, connection.inlet_index
            G.add_edge(u, v, key=conn_id)
            edge_index[(u, v, conn_id)] = conn_id

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError(
                "Flow directions form a cycle — cannot determine topological order. "
                "Check for bidirectional flow loops."
            )

        edge_order = []
        for node in nx.topological_sort(G):
            for successor in G.successors(node):
                for key in G.get_edge_data(node, successor):
                    edge_order.append(edge_index[(node, successor, key)])

        self._cached_edge_order = edge_order
        self._cached_flow_signs = {
            conn_id: self._flow_sign(c)
            for conn_id, c in connections.items()
        }
        logger.debug("Rebuilt topological edge order (%d edges).", len(edge_order))

    # ------------------------------------------------------------------
    # Propagation strategies (all single-pass)
    # ------------------------------------------------------------------

    def _propagate_simple_mixing(self, edge_order: list) -> np.ndarray:
        """
        Flow-weighted mixing: outflow composition = upstream node composition.

        One pass is sufficient because the upstream node's tmp composition is
        finalised before any downstream edge is processed (topological order).
        """
        network = self.network
        n_nodes = network.get_simulation_node_count()
        non_junction = network.non_junction_nodes

        composition_sum = np.zeros((_N_SPECIES, n_nodes))
        flow_sum = np.zeros(n_nodes)

        for conn_id in edge_order:
            connection = network.connections[conn_id]
            flow_rate = connection.flow_rate
            if not flow_rate:
                continue

            if self._flow_sign(connection) >= 0:
                upstream_id = connection.inlet_index
                downstream_id = connection.outlet_index
            else:
                upstream_id = connection.outlet_index
                downstream_id = connection.inlet_index

            outflow_comp = (
                network.nodes[upstream_id].gas_mixture.eos_composition_tmp
            )
            abs_flow = abs(flow_rate)
            ds_idx = network.node_id_to_simulation_node_index(downstream_id)

            composition_sum[:, ds_idx] += outflow_comp * abs_flow
            flow_sum[ds_idx] += abs_flow

            # Immediately finalise downstream node so edges further downstream
            # see the correct composition — but never overwrite a reference /
            # shortpipe-inlet node, whose composition is fixed by the supply.
            if downstream_id not in non_junction:
                network.nodes[downstream_id].gas_mixture.eos_composition_tmp = (
                    composition_sum[:, ds_idx] / flow_sum[ds_idx]
                )

        with np.errstate(divide="ignore", invalid="ignore"):
            nodal_composition = np.where(
                flow_sum > 0,
                composition_sum / flow_sum,
                np.nan,
            )
        return nodal_composition

    def _propagate_no_mixing(self, edge_order: list) -> np.ndarray:
        """
        No mixing: for Pipelines, the edge's outflow is the downstream node's
        current composition (pipe does not mix contents). For non-Pipeline
        connections (ShortPipe / Compressor / Resistance) the edge always
        carries the upstream composition — matching the inline semantics
        where ShortPipe/Compressor cache outflow_composition = inlet composition
        at construction.

        At a junction receiving any non-Pipeline inflow carrying a different
        composition, the node's fixed-point composition is the flow-weighted
        average of those external contributions. Single-pass does not reach
        that fixed point because Pipeline edges self-reference the node's own
        composition, so we iterate until the nodal composition matrix stops
        changing — mirroring the `while to_update` loop in the original
        `calculate_nodal_inflow_states`.
        """
        network = self.network
        n_nodes = network.get_simulation_node_count()
        non_junction = network.non_junction_nodes

        prev_nodal = np.zeros((_N_SPECIES, n_nodes))
        max_inner_iter = 100
        nodal_composition = prev_nodal

        for _ in range(max_inner_iter):
            composition_sum = np.zeros((_N_SPECIES, n_nodes))
            flow_sum = np.zeros(n_nodes)

            for conn_id in edge_order:
                connection = network.connections[conn_id]
                flow_rate = connection.flow_rate
                if not flow_rate:
                    continue

                if self._flow_sign(connection) >= 0:
                    upstream_id = connection.inlet_index
                    downstream_id = connection.outlet_index
                else:
                    upstream_id = connection.outlet_index
                    downstream_id = connection.inlet_index

                if isinstance(connection, Pipeline):
                    # Pipeline under no_mixing: outflow is the downstream
                    # node's current composition (pipe leaves contents alone).
                    outflow_comp = (
                        network.nodes[downstream_id].gas_mixture.eos_composition_tmp
                    )
                else:
                    # ShortPipe / Compressor / Resistance: outflow carries the
                    # upstream composition, regardless of tracking method.
                    outflow_comp = (
                        network.nodes[upstream_id].gas_mixture.eos_composition_tmp
                    )

                abs_flow = abs(flow_rate)
                ds_idx = network.node_id_to_simulation_node_index(downstream_id)
                composition_sum[:, ds_idx] += outflow_comp * abs_flow
                flow_sum[ds_idx] += abs_flow

            with np.errstate(divide="ignore", invalid="ignore"):
                nodal_composition = np.where(
                    flow_sum > 0,
                    composition_sum / flow_sum,
                    np.nan,
                )

            # Write back to junction nodes so the next inner iteration sees
            # the updated composition via eos_composition_tmp.
            for sim_idx in range(n_nodes):
                col = nodal_composition[:, sim_idx]
                if np.any(np.isnan(col)):
                    continue
                node_id = network.simulation_node_index_to_node_id(sim_idx)
                if node_id in non_junction:
                    continue
                network.nodes[node_id].gas_mixture.eos_composition_tmp = col

            if _allclose_with_nan(nodal_composition, prev_nodal):
                break
            prev_nodal = nodal_composition

        return nodal_composition

    def _propagate_batch_tracking(self, edge_order: list) -> np.ndarray:
        """
        Batch tracking: advance each pipe's batch state and accumulate
        outflow compositions into downstream nodes.
        """
        network = self.network
        n_nodes = network.get_simulation_node_count()
        non_junction = network.non_junction_nodes
        composition_sum = np.zeros((_N_SPECIES, n_nodes))
        flow_sum = np.zeros(n_nodes)

        for conn_id in edge_order:
            connection = network.connections[conn_id]
            # ShortPipe / Compressor / Resistance are pass-through: they
            # contribute the upstream composition to the downstream node.
            if not isinstance(connection, Pipeline):
                flow_rate = connection.flow_rate
                if not flow_rate:
                    continue
                if self._flow_sign(connection) >= 0:
                    upstream_id = connection.inlet_index
                    downstream_id = connection.outlet_index
                else:
                    upstream_id = connection.outlet_index
                    downstream_id = connection.inlet_index
                outflow_comp = (
                    network.nodes[upstream_id].gas_mixture.eos_composition_tmp
                )
                abs_flow = abs(flow_rate)
                ds_idx = network.node_id_to_simulation_node_index(downstream_id)
                composition_sum[:, ds_idx] += outflow_comp * abs_flow
                flow_sum[ds_idx] += abs_flow
                if flow_sum[ds_idx] > 0 and downstream_id not in non_junction:
                    network.nodes[downstream_id].gas_mixture.eos_composition_tmp = (
                        composition_sum[:, ds_idx] / flow_sum[ds_idx]
                    )
                continue

            flow_rate = connection.flow_rate
            sign = self._flow_sign(connection)
            velocity = connection.flow_velocity or 0.0

            if sign >= 0:
                upstream_id = connection.inlet_index
                downstream_id = connection.outlet_index
                inflow_comp = network.nodes[upstream_id].gas_mixture.eos_composition_tmp
                outflow_comp = network.nodes[downstream_id].gas_mixture.eos_composition_tmp
            else:
                upstream_id = connection.outlet_index
                downstream_id = connection.inlet_index
                inflow_comp = network.nodes[upstream_id].gas_mixture.eos_composition_tmp
                outflow_comp = network.nodes[downstream_id].gas_mixture.eos_composition_tmp

            (
                connection.batch_location_history,
                connection.composition_history,
                outflow_comp,
            ) = batch_tracking(
                self.time_step,
                velocity,
                connection.length,
                inflow_comp,
                outflow_comp,
                connection.batch_location_history,
                connection.composition_history,
            )
            connection.outflow_composition = outflow_comp

            if not flow_rate:
                continue

            abs_flow = abs(flow_rate)
            ds_idx = network.node_id_to_simulation_node_index(downstream_id)
            composition_sum[:, ds_idx] += outflow_comp * abs_flow
            flow_sum[ds_idx] += abs_flow

            # Immediately update so downstream edges see the fresh composition
            # — but never overwrite a reference / shortpipe-inlet node, whose
            # composition is fixed by the supply.
            if flow_sum[ds_idx] > 0 and downstream_id not in non_junction:
                network.nodes[downstream_id].gas_mixture.eos_composition_tmp = (
                    composition_sum[:, ds_idx] / flow_sum[ds_idx]
                )

        with np.errstate(divide="ignore", invalid="ignore"):
            nodal_composition = np.where(
                flow_sum > 0,
                composition_sum / flow_sum,
                np.nan,
            )
        return nodal_composition
