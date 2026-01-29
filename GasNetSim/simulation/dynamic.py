"""
Dynamic transient simulation for GasNetSim Pi-model networks.

Author: Yetkin Civan Serin
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

try:
    from ..components.dynamic.dynamic_network import DynamicNetwork
    from .timeseries import save_time_series_results_to_file
except ImportError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from GasNetSim.components.dynamic.dynamic_network import DynamicNetwork
    from GasNetSim.simulation.timeseries import save_time_series_results_to_file


def _build_initial_pressure_vector(network: DynamicNetwork, P0: Optional[np.ndarray]):
    # Initialize physical-node pressures from input or node state
    n_phys = network.n_physical_nodes
    p_init = np.zeros(n_phys)

    if P0 is not None:
        if len(P0) != n_phys:
            raise ValueError("P0 length must match number of physical nodes.")
        return np.array(P0, dtype=float)

    for sim_idx in range(n_phys):
        node_id = network.simulation_node_index_to_node_id(sim_idx)
        node = network.nodes[node_id]
        p_init[sim_idx] = node.pressure if node.pressure is not None else 0.0

    return p_init


def _resolve_supply_pressures(network: DynamicNetwork, supply_pressures):
    # Normalize supply pressures to an array sized by virtual nodes
    if supply_pressures is None:
        return np.array(network.get_supply_pressures(), dtype=float)
    if np.isscalar(supply_pressures):
        return np.full(network.n_virtual_nodes, float(supply_pressures))
    if len(supply_pressures) != network.n_virtual_nodes:
        raise ValueError("Supply pressure length must match number of supply nodes.")
    return np.array(supply_pressures, dtype=float)


def _resolve_demands(
    network: DynamicNetwork,
    demands: Optional[Dict[int, np.ndarray]],
    n_steps: int,
) -> List[Tuple[int, np.ndarray]]:
    # Use explicit demand profiles if provided; otherwise pull from nodes
    demand_list = []

    if demands is not None:
        for node_id, profile in demands.items():
            global_idx = network.node_id_to_global_index(node_id)
            if len(profile) != n_steps:
                raise ValueError(f"Demand profile length mismatch for node {node_id}.")
            demand_list.append((global_idx, np.array(profile, dtype=float)))
        return demand_list

    for node in network.get_demand_nodes():
        profile = None
        if hasattr(node, "demand_profile") and node.demand_profile is not None:
            profile = np.array(node.demand_profile, dtype=float)
        else:
            if hasattr(node, "create_constant_profile"):
                profile = node.create_constant_profile(n_steps)
        if profile is None:
            profile = np.zeros(n_steps)
        if len(profile) != n_steps:
            raise ValueError(f"Demand profile length mismatch for node {node.index}.")
        global_idx = network.node_id_to_global_index(node.index)
        demand_list.append((global_idx, profile))

    return demand_list


def _apply_demand_events(
    network: DynamicNetwork,
    demand_list: List[Tuple[int, np.ndarray]],
    events: Optional[List[Dict]],
    dt: float,
    n_steps: int,
) -> List[Tuple[int, np.ndarray]]:
    # Apply step events to demand profiles (mult/add/set from a given time/step)
    if not events:
        return demand_list

    demand_map = {idx: profile.copy() for idx, profile in demand_list}

    for event in events:
        # Identify target node (node id or global index)
        if "node" in event:
            global_idx = network.node_id_to_global_index(event["node"])
        elif "global_idx" in event:
            global_idx = int(event["global_idx"])
        else:
            raise ValueError("Event must include 'node' or 'global_idx'.")

        # Resolve event time to step index
        if "step" in event:
            step = int(event["step"])
        elif "time" in event:
            step = int(float(event["time"]) / dt)
        else:
            raise ValueError("Event must include 'step' or 'time'.")

        step = max(0, min(step, n_steps - 1))

        profile = demand_map.get(global_idx)
        if profile is None:
            profile = np.zeros(n_steps, dtype=float)

        # Apply operation from step onward
        if "mult" in event:
            profile[step:] *= float(event["mult"])
        elif "add" in event:
            profile[step:] += float(event["add"])
        elif "set" in event:
            profile[step:] = float(event["set"])
        else:
            raise ValueError("Event must include 'mult', 'add', or 'set'.")

        demand_map[global_idx] = profile

    return list(demand_map.items())


def simulate_transient(
    network: DynamicNetwork,
    dt: float,
    t_end: float,
    Z: float,
    M: float,
    P0: Optional[np.ndarray] = None,
    T_nodes: Optional[np.ndarray] = None,
    supply_pressures: Optional[np.ndarray] = None,
    demands: Optional[Dict[int, np.ndarray]] = None,
    events: Optional[List[Dict]] = None,
    save_to_file: bool = False,
    output_format: str = "excel",
    output_filename: str = "dynamic_results",
):
    """
    Run a transient Pi-model simulation on a DynamicNetwork.

    events: list of dicts with keys:
      - node (node_id) or global_idx
      - time (seconds) or step (int)
      - mult / add / set (operation on demand profile from that time onward)
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_end <= 0:
        raise ValueError("t_end must be positive.")

    n_steps = int(t_end / dt)
    time = np.linspace(0, t_end, n_steps + 1)

    # Build temperatures for all (virtual + physical) nodes
    if T_nodes is None:
        T_nodes = network.get_temperature_vector()
    if len(T_nodes) != network.n_nodes:
        raise ValueError("T_nodes length must match total node count.")

    # Pressure history stores physical nodes only
    P_hist = np.zeros((network.n_physical_nodes, n_steps + 1))
    P_hist[:, 0] = _build_initial_pressure_vector(network, P0)

    supply_p = _resolve_supply_pressures(network, supply_pressures)
    demand_list = _resolve_demands(network, demands, n_steps)
    demand_list = _apply_demand_events(network, demand_list, events, dt, n_steps)

    # Assemble constant Y once; update I and states each step
    network.reset_states()
    Y = network.build_Y(dt, Z, T_nodes, M)
    Y_sparse = csr_matrix(Y)

    for n in range(n_steps):
        # Build full pressure vector with virtual nodes
        P_old = np.zeros(network.n_nodes)

        for v_idx, p_idx in network.virtual_connections:
            P_old[v_idx] = supply_p[v_idx]
            P_old[p_idx] = P_hist[p_idx - network.n_virtual_nodes, n]

        for sim_idx in range(network.n_physical_nodes):
            global_idx = sim_idx + network.n_virtual_nodes
            P_old[global_idx] = P_hist[sim_idx, n]

        I_eq = network.update_I(P_old, dt, Z, T_nodes, M)

        for global_idx, profile in demand_list:
            I_eq[global_idx] -= profile[n]

        for v_idx, _ in network.virtual_connections:
            I_eq[v_idx] = supply_p[v_idx]

        # Solve for updated pressures
        solution = spsolve(Y_sparse, I_eq)

        P_new = np.zeros(network.n_nodes)
        for v_idx, _ in network.virtual_connections:
            P_new[v_idx] = supply_p[v_idx]
        P_new[network.n_virtual_nodes :] = solution[network.n_virtual_nodes :]

        # Update history terms for the next step
        network.update_states(P_old, P_new, dt, Z, T_nodes, M)

        P_hist[:, n + 1] = P_new[network.n_virtual_nodes :]

        # Write back node pressures
        for sim_idx in range(network.n_physical_nodes):
            node_id = network.simulation_node_index_to_node_id(sim_idx)
            node = network.nodes[node_id]
            if hasattr(node, "set_pressure"):
                node.set_pressure(P_new[sim_idx + network.n_virtual_nodes])
            else:
                node.pressure = P_new[sim_idx + network.n_virtual_nodes]

    if save_to_file:
        results = {"nodal_pressure": P_hist.T.tolist()}
        time_steps = list(range(n_steps + 1))
        save_time_series_results_to_file(
            results, time_steps, output_format, output_filename
        )

    return time, P_hist
