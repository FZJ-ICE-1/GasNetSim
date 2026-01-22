"""
Dynamic Network - Pi-model scaffold based on GasNetSim Network.

Author: Yetkin Civan Serin
"""

import math
import numpy as np
from scipy.constants import g as GRAVITY
from scipy.constants import R as R_UNIV

from GasNetSim.components.network import Network


class DynamicNetwork(Network):
    """
    Dynamic network subclass that keeps Network inputs but adds Pi-model state.
    """

    def __init__(
        self,
        nodes: dict,
        pipelines=None,
        compressors=None,
        resistances=None,
        linear_resistances=None,
        shortpipes=None,
        run_initialization=True,
        pressure_prev=None,
        base_composition=None,
    ):
        super().__init__(
            nodes=nodes,
            pipelines=pipelines,
            compressors=compressors,
            resistances=resistances,
            linear_resistances=linear_resistances,
            shortpipes=shortpipes,
            run_initialization=run_initialization,
            pressure_prev=pressure_prev,
            base_composition=base_composition,
        )

        # Physical nodes from base Network; virtual nodes are added for supplies
        self.n_physical_nodes = len(self.nodes)
        self._supply_nodes = [n for n in self.nodes.values() if self._is_supply_node(n)]
        self._demand_nodes = [n for n in self.nodes.values() if self._is_demand_node(n)]
        self.n_virtual_nodes = len(self._supply_nodes)
        self.n_nodes = self.n_physical_nodes + self.n_virtual_nodes

        # Use pipeline list for Pi-model arrays
        self.pipes = list(self.pipelines.values()) if self.pipelines else []
        n_pipes = len(self.pipes)

        # Global index = virtual nodes [0..n_virtual-1] + physical nodes (offset)
        self._assign_global_indices()
        self.virtual_connections = self._build_virtual_connections()

        # History terms for Pi-model time stepping
        self.mC_prev = np.zeros((n_pipes, 2))
        self.mL_prev = np.zeros(n_pipes)

        # Precompute constants for each pipe
        self._precompute_constants()

    @staticmethod
    def _is_supply_node(node) -> bool:
        node_type = getattr(node, "node_type", None)
        if node_type is None:
            return False
        return str(node_type).lower() in {"reference", "supply", "slack"}

    @staticmethod
    def _is_demand_node(node) -> bool:
        node_type = getattr(node, "node_type", None)
        if node_type is None:
            return False
        if str(node_type).lower() in {"reference", "supply", "slack"}:
            return False
        return str(node_type).lower() == "demand"

    def _assign_global_indices(self):
        """
        Assign global indices (virtual + physical) to nodes for dynamic solving.
        """
        for v_idx, node in enumerate(self._supply_nodes):
            # Virtual nodes map to supply nodes for pressure BCs
            if hasattr(node, "virtual_index"):
                node.virtual_index = v_idx
            if hasattr(node, "global_index"):
                node.global_index = self.node_id_to_simulation_node_index(node.index) + self.n_virtual_nodes

        for node in self.nodes.values():
            if node in self._supply_nodes:
                continue
            if hasattr(node, "global_index"):
                node.global_index = self.node_id_to_simulation_node_index(node.index) + self.n_virtual_nodes

    def _build_virtual_connections(self):
        # (virtual_idx, physical_global_idx) pairs for Y-matrix stamping
        connections = []
        for node in self._supply_nodes:
            v_idx = getattr(node, "virtual_index", None)
            if v_idx is None:
                v_idx = len(connections)
                if hasattr(node, "virtual_index"):
                    node.virtual_index = v_idx
            p_idx = self.node_id_to_simulation_node_index(node.index) + self.n_virtual_nodes
            connections.append((v_idx, p_idx))
        return connections

    def _precompute_constants(self):
        # GLc, GCc, Ggc use base geometry; multiplied by dt/T during assembly
        n_pipes = len(self.pipes)
        self.GLc = np.empty(n_pipes)
        self.GCc = np.empty(n_pipes)
        self.Ggc = np.empty(n_pipes)
        self.i_idx = np.empty(n_pipes, dtype=int)
        self.j_idx = np.empty(n_pipes, dtype=int)

        for k, pipe in enumerate(self.pipes):
            inlet_id = getattr(pipe, "inlet_index", None)
            outlet_id = getattr(pipe, "outlet_index", None)

            self.i_idx[k] = (
                self.node_id_to_simulation_node_index(inlet_id) + self.n_virtual_nodes
            )
            self.j_idx[k] = (
                self.node_id_to_simulation_node_index(outlet_id) + self.n_virtual_nodes
            )

            area = getattr(pipe, "A", None)
            if area is None:
                area = math.pi * (pipe.diameter / 2.0) ** 2

            self.GLc[k] = area / pipe.length
            self.GCc[k] = area * pipe.length
            self.Ggc[k] = area * GRAVITY * math.sin(getattr(pipe, "theta", 0.0))

    def reset_states(self):
        self.mC_prev[:] = 0.0
        self.mL_prev[:] = 0.0

    def build_Y(self, dt, Z, T_nodes, M):
        """
        Build nodal admittance matrix for the Pi-model.
        """
        Y = np.zeros((self.n_nodes, self.n_nodes))

        # Tie each virtual node to its physical supply node
        for v_idx, p_idx in self.virtual_connections:
            Y[v_idx, p_idx] = 1.0
            Y[p_idx, v_idx] = 1.0

        for k, pipe in enumerate(self.pipes):
            i = self.i_idx[k]
            j = self.j_idx[k]

            G_L = self.GLc[k] * dt
            G_C_i = (M * self.GCc[k]) / (Z * R_UNIV * T_nodes[i] * dt)
            G_C_j = (M * self.GCc[k]) / (Z * R_UNIV * T_nodes[j] * dt)
            g_theta_i = (M * self.Ggc[k] * dt) / (2.0 * Z * R_UNIV * T_nodes[i])
            g_theta_j = (M * self.Ggc[k] * dt) / (2.0 * Z * R_UNIV * T_nodes[j])

            Y[i, i] += G_L + G_C_i - g_theta_i
            Y[j, j] += G_L + G_C_j + g_theta_j
            Y[i, j] += -G_L - g_theta_i
            Y[j, i] += -G_L + g_theta_j

        return Y

    def update_I(self, P_old, dt, Z, T_nodes, M):
        """
        Compute equivalent current injection vector for the Pi-model.
        """
        I_eq = np.zeros(self.n_nodes)
        p_min = 1e5

        for k, pipe in enumerate(self.pipes):
            i = self.i_idx[k]
            j = self.j_idx[k]

            G_C_i = (M * self.GCc[k]) / (Z * R_UNIV * T_nodes[i] * dt)
            G_C_j = (M * self.GCc[k]) / (Z * R_UNIV * T_nodes[j] * dt)

            mC_eq_i = G_C_i * P_old[i] - self.mC_prev[k, 0]
            mC_eq_j = G_C_j * P_old[j] - self.mC_prev[k, 1]

            p_avg = P_old[i] + P_old[j]
            if p_avg < p_min:
                p_avg = p_min

            area = getattr(pipe, "A", None)
            if area is None:
                area = math.pi * (pipe.diameter / 2.0) ** 2

            pipe_d = getattr(pipe, "D", None)
            if pipe_d is None:
                pipe_d = pipe.diameter

            # Explicit friction term using previous inductor flow
            friction_coeff = (pipe.f * Z * R_UNIV * T_nodes[i] * dt) / (
                M * area * pipe_d
            )
            friction_term = (
                friction_coeff * self.mL_prev[k] * abs(self.mL_prev[k]) / p_avg
            )
            mL_eq = self.mL_prev[k] - friction_term

            I_eq[i] += mC_eq_i - mL_eq
            I_eq[j] += mC_eq_j + mL_eq

        return I_eq

    def update_states(self, P_old, P_new, dt, Z, T_nodes, M):
        """
        Update Pi-model history terms after solving for new pressures.
        """
        p_min = 1e5
        G_L = self.GLc * dt

        for k, pipe in enumerate(self.pipes):
            i = self.i_idx[k]
            j = self.j_idx[k]

            p_avg = P_old[i] + P_old[j]
            if p_avg < p_min:
                p_avg = p_min

            area = getattr(pipe, "A", None)
            if area is None:
                area = math.pi * (pipe.diameter / 2.0) ** 2

            pipe_d = getattr(pipe, "D", None)
            if pipe_d is None:
                pipe_d = pipe.diameter

            # Explicit friction term uses previous inductor flow
            friction_coeff = (pipe.f * Z * R_UNIV * T_nodes[i] * dt) / (
                M * area * pipe_d
            )
            friction_term = (
                friction_coeff * self.mL_prev[k] * abs(self.mL_prev[k]) / p_avg
            )
            mL_eq = self.mL_prev[k] - friction_term

            G_C_i = (M * self.GCc[k]) / (Z * R_UNIV * T_nodes[i] * dt)
            G_C_j = (M * self.GCc[k]) / (Z * R_UNIV * T_nodes[j] * dt)
            mC_eq_i = G_C_i * P_old[i] - self.mC_prev[k, 0]
            mC_eq_j = G_C_j * P_old[j] - self.mC_prev[k, 1]

            self.mC_prev[k, 0] = -G_C_i * P_new[i] + mC_eq_i
            self.mC_prev[k, 1] = -G_C_j * P_new[j] + mC_eq_j
            self.mL_prev[k] = G_L[k] * (P_new[i] - P_new[j]) + mL_eq

    def get_supply_pressures(self):
        return np.array([node.pressure for node in self._supply_nodes])

    def get_temperature_vector(self, default_T=288.15):
        T = np.full(self.n_nodes, default_T)

        for node in self.nodes.values():
            sim_idx = self.node_id_to_simulation_node_index(node.index)
            T[sim_idx + self.n_virtual_nodes] = (
                node.temperature if node.temperature is not None else default_T
            )

        for v_idx, p_idx in self.virtual_connections:
            T[v_idx] = T[p_idx]

        return T

    def get_demand_nodes(self):
        return self._demand_nodes

    def get_supply_nodes(self):
        return self._supply_nodes

    def node_id_to_global_index(self, node_id):
        return self.node_id_to_simulation_node_index(node_id) + self.n_virtual_nodes

    def global_index_to_node_id(self, global_idx):
        if global_idx < self.n_virtual_nodes:
            raise ValueError(
                f"Index {global_idx} is a virtual node, not a physical node"
            )
        sim_idx = global_idx - self.n_virtual_nodes
        return self.simulation_node_index_to_node_id(sim_idx)

    def get_physical_node_pressures(self, P_full):
        return P_full[self.n_virtual_nodes :]
