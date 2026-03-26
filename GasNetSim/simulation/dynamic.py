"""
Dynamic transient simulation for GasNetSim Pi-model networks.

Author: Yetkin Civan Serin
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple
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


def _build_full_pressure_vector(
    network: DynamicNetwork,
    p_physical: np.ndarray,
    supply_p: np.ndarray,
) -> np.ndarray:
    # Assemble full solver vector [virtual supplies & physical nodes]
    p_full = np.zeros(network.n_nodes, dtype=float)
    p_full[network.n_virtual_nodes :] = p_physical
    for v_idx, _ in network.virtual_connections:
        p_full[v_idx] = supply_p[v_idx]
    return p_full


def _write_back_physical_pressures(
    network: DynamicNetwork,
    p_physical: np.ndarray,
) -> None:
    # Push solved physical-node pressures back to node objects
    for sim_idx in range(network.n_physical_nodes):
        node_id = network.simulation_node_index_to_node_id(sim_idx)
        node = network.nodes[node_id]
        p_val = float(p_physical[sim_idx])
        if hasattr(node, "set_pressure"):
            node.set_pressure(p_val)
        else:
            node.pressure = p_val


def _snapshot_history_states(network: DynamicNetwork) -> Tuple[np.ndarray, np.ndarray]:
    # Save history terms so rejected adaptive trials can roll back
    return network.mC_prev.copy(), network.mL_prev.copy()


def _restore_history_states(
    network: DynamicNetwork, snapshot: Tuple[np.ndarray, np.ndarray]
) -> None:
    # Restore history terms after a rejected trial
    m_c, m_l = snapshot
    network.mC_prev[:, :] = m_c
    network.mL_prev[:] = m_l


def _advance_single_step(
    network: DynamicNetwork,
    p_old_phys: np.ndarray,
    dt: float,
    Z: float,
    T_nodes: np.ndarray,
    M: float,
    supply_p: np.ndarray,
    demand_values: Dict[int, float],
    friction_treatment: str = "explicit",
    y_sparse=None,
) -> np.ndarray:
    # One Pi-model step used by adaptive full/half-step trials
    p_old = _build_full_pressure_vector(network, p_old_phys, supply_p)
    i_eq = network.update_I(
        p_old, dt, Z, T_nodes, M, friction_treatment=friction_treatment
    )

    for global_idx, demand_value in demand_values.items():
        i_eq[global_idx] -= float(demand_value)

    for v_idx, _ in network.virtual_connections:
        i_eq[v_idx] = supply_p[v_idx]

    if y_sparse is None:
        y_sparse = csr_matrix(network.build_Y(dt, Z, T_nodes, M))
    solution = spsolve(y_sparse, i_eq)
    p_new_phys = solution[network.n_virtual_nodes :].copy()
    p_new = _build_full_pressure_vector(network, p_new_phys, supply_p)

    network.update_states(
        p_old,
        p_new,
        dt,
        Z,
        T_nodes,
        M,
        friction_treatment=friction_treatment,
    )

    return p_new_phys


def _build_time_demand_functions(
    network: DynamicNetwork,
    demands: Dict[int, np.ndarray],
    demand_dt: float,
) -> Tuple[Dict[int, Callable[[float], float]], int]:
    # Convert node demand profile arrays into callable d(t) profiles
    functions: Dict[int, Callable[[float], float]] = {}
    max_profile_len = 1

    if demand_dt <= 0:
        raise ValueError("demand_dt must be positive.")

    for node_id, profile in demands.items():
        global_idx = network.node_id_to_global_index(node_id)
        arr = np.asarray(profile, dtype=float).reshape(-1)
        if arr.size == 0:
            functions[global_idx] = lambda _t: 0.0
            continue

        if arr.size == 1:
            value = float(arr[0])
            functions[global_idx] = (lambda v: (lambda _t: v))(value)
            continue

        local_arr = arr.copy()
        max_profile_len = max(max_profile_len, int(local_arr.size))

        def profile_fn(t, values=local_arr, dt=demand_dt):
            idx = int(np.floor(max(float(t), 0.0) / dt))
            idx = min(idx, len(values) - 1)
            return float(values[idx])

        functions[global_idx] = profile_fn

    return functions, max_profile_len


def _normalize_time_events(
    network: DynamicNetwork,
    events: Optional[List[Dict]],
) -> Tuple[Dict[int, List[Tuple[float, str, float]]], np.ndarray]:
    # Pre-sort events per node and keep a global list of event times
    if not events:
        return {}, np.array([], dtype=float)

    by_node: Dict[int, List[Tuple[float, str, float]]] = {}
    event_times: List[float] = []

    for event in events:
        if "node" not in event or "time" not in event:
            raise ValueError("Events must include 'node' and 'time'.")
        global_idx = network.node_id_to_global_index(event["node"])
        t_event = float(event["time"])

        if "mult" in event:
            op, operand = "mult", float(event["mult"])
        elif "add" in event:
            op, operand = "add", float(event["add"])
        elif "set" in event:
            op, operand = "set", float(event["set"])
        else:
            raise ValueError("Events must include 'mult', 'add', or 'set'.")

        by_node.setdefault(global_idx, []).append((t_event, op, operand))
        event_times.append(t_event)

    for global_idx in by_node:
        by_node[global_idx].sort(key=lambda x: x[0])

    if not event_times:
        return by_node, np.array([], dtype=float)
    unique_times = np.array(sorted(set(event_times)), dtype=float)
    return by_node, unique_times


def _demand_values_at_time(
    t: float,
    base_functions: Dict[int, Callable[[float], float]],
    events_by_node: Dict[int, List[Tuple[float, str, float]]],
) -> Dict[int, float]:
    # Evaluate base demands and apply all events active at time t
    values: Dict[int, float] = {}
    for global_idx, fn in base_functions.items():
        value = float(fn(t))
        for t_event, op, operand in events_by_node.get(global_idx, []):
            if t < t_event:
                break
            if op == "mult":
                value *= operand
            elif op == "add":
                value += operand
            elif op == "set":
                value = operand
        values[global_idx] = value
    return values


def _next_time_after(sorted_times: np.ndarray, t: float, eps: float) -> float:
    # Return the next strictly-future boundary/event time
    if sorted_times.size == 0:
        return math.inf
    idx = int(np.searchsorted(sorted_times, t + eps, side="right"))
    if idx >= sorted_times.size:
        return math.inf
    return float(sorted_times[idx])


def _next_demand_break_time(
    t: float,
    demand_dt: Optional[float],
    max_profile_len: int,
) -> float:
    # Demand profiles are piecewise constant; gives next jump time
    if demand_dt is None or demand_dt <= 0 or max_profile_len <= 1:
        return math.inf
    next_idx = int(np.floor(max(float(t), 0.0) / demand_dt)) + 1
    if next_idx >= max_profile_len:
        return math.inf
    return float(next_idx * demand_dt)


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
    friction_treatment: str = "explicit",
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

    friction_treatment:
      - "explicit": legacy explicit friction update
      - "semi_implicit": sign-preserving implicit friction damping update
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
        P_old = _build_full_pressure_vector(network, P_hist[:, n], supply_p)

        I_eq = network.update_I(
            P_old, dt, Z, T_nodes, M, friction_treatment=friction_treatment
        )

        for global_idx, profile in demand_list:
            I_eq[global_idx] -= profile[n]

        for v_idx, _ in network.virtual_connections:
            I_eq[v_idx] = supply_p[v_idx]

        # Solve for updated pressures
        solution = spsolve(Y_sparse, I_eq)
        p_new_phys = solution[network.n_virtual_nodes :].copy()
        P_new = _build_full_pressure_vector(network, p_new_phys, supply_p)

        # Update history terms for the next step
        network.update_states(
            P_old,
            P_new,
            dt,
            Z,
            T_nodes,
            M,
            friction_treatment=friction_treatment,
        )

        P_hist[:, n + 1] = p_new_phys
        _write_back_physical_pressures(network, p_new_phys)

    if save_to_file:
        results = {"nodal_pressure": P_hist.T.tolist()}
        time_steps = list(range(n_steps + 1))
        save_time_series_results_to_file(
            results, time_steps, output_format, output_filename
        )

    return time, P_hist


def simulate_transient_adaptive(
    network: DynamicNetwork,
    dt_init: float,
    t_end: float,
    Z: float,
    M: float,
    P0: Optional[np.ndarray] = None,
    T_nodes: Optional[np.ndarray] = None,
    supply_pressures: Optional[np.ndarray] = None,
    demands: Optional[Dict[int, np.ndarray]] = None,
    demand_dt: Optional[float] = None,
    events: Optional[List[Dict]] = None,
    friction_treatment: str = "semi_implicit",
    rtol: float = 1e-4,
    atol: float = 1e3,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 2.0,
    output_times: Sequence[float] = (),
    max_steps: int = 200000,
):
    """
    Run an adaptive transient Pi-model simulation using step-doubling control.

    Notes:
      - Keeps the existing Pi-model kernel (`update_I`/`update_states`) unchanged.
      - Uses one full step vs two half steps to estimate local truncation error.
      - Applies accept/reject and dt scaling similar to adaptive ODE workflows.
      - Accepted steps are clipped to hit `output_times` exactly.
      - If multi-value `demands` are provided, `demand_dt` defines profile spacing.
      - Returns `(time, p_hist, diagnostics)` for benchmarking.
    """
    if dt_init <= 0:
        raise ValueError("dt_init must be positive.")
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    if dt_min is None:
        dt_min = dt_init * 1e-3
    if dt_max is None:
        dt_max = dt_init * 10.0
    if dt_min <= 0 or dt_max <= 0:
        raise ValueError("dt_min and dt_max must be positive.")
    if dt_max < dt_min:
        raise ValueError("dt_max must be >= dt_min.")

    if T_nodes is None:
        T_nodes = network.get_temperature_vector()
    T_nodes = np.array(T_nodes, dtype=float)
    if len(T_nodes) != network.n_nodes:
        raise ValueError("T_nodes length must match total node count.")

    if demands is None:
        raise ValueError("demands must be provided for adaptive benchmark runs.")

    supply_p = _resolve_supply_pressures(network, supply_pressures)
    p_curr = _build_initial_pressure_vector(network, P0)

    step_dt = demand_dt if demand_dt is not None else dt_init
    base_demand_functions, max_profile_len = _build_time_demand_functions(
        network=network,
        demands=demands,
        demand_dt=step_dt,
    )
    events_by_node, event_times = _normalize_time_events(network, events)

    time_out = np.asarray(output_times, dtype=float)
    if time_out.ndim != 1 or time_out.size < 2:
        raise ValueError("output_times must be a 1D sequence with at least 2 points.")
    if not np.all(np.diff(time_out) > 0):
        raise ValueError("output_times must be strictly increasing.")
    if abs(float(time_out[0])) > 1e-12:
        raise ValueError("output_times must start at 0.0.")
    if abs(float(time_out[-1]) - float(t_end)) > 1e-10:
        raise ValueError("output_times must end at t_end.")

    p_hist = np.zeros((network.n_physical_nodes, len(time_out)))
    p_hist[:, 0] = p_curr
    next_output_idx = 1

    network.reset_states()

    t = 0.0
    dt = float(dt_init)
    accepted_steps = 0
    rejected_steps = 0
    eps_time = 1e-10

    while t < t_end - eps_time:
        if (accepted_steps + rejected_steps) >= max_steps:
            raise RuntimeError(
                f"Adaptive integration exceeded max_steps={max_steps} before t_end."
            )

        # Determine the next time to step to based on events, output times, and demand profile breaks
        next_event_time = _next_time_after(event_times, t, eps_time)
        next_output_time = _next_time_after(time_out, t, eps_time)
        next_demand_time = _next_demand_break_time(t, step_dt, max_profile_len)

        boundary_time = min(float(t_end), next_event_time, next_output_time, next_demand_time)
        max_step_to_boundary = max(0.0, boundary_time - t)

        if max_step_to_boundary <= eps_time:
            t = min(boundary_time, t_end)
            while next_output_idx < len(time_out) and t >= time_out[next_output_idx] - eps_time:
                p_hist[:, next_output_idx] = p_curr
                next_output_idx += 1
            continue

        dt_lower = min(dt_min, max_step_to_boundary)
        dt_try = min(dt, dt_max, max_step_to_boundary)
        if dt_try < dt_lower:
            dt_try = dt_lower

        state_start = _snapshot_history_states(network)
        state_half = None
        p_half_2 = None
        err = math.inf

        try:
            # Error estimate: one full step vs two half-steps
            y_full = csr_matrix(network.build_Y(dt_try, Z, T_nodes, M))
            demand_t = _demand_values_at_time(t, base_demand_functions, events_by_node)
            p_full = _advance_single_step(
                network=network,
                p_old_phys=p_curr,
                dt=dt_try,
                Z=Z,
                T_nodes=T_nodes,
                M=M,
                supply_p=supply_p,
                demand_values=demand_t,
                friction_treatment=friction_treatment,
                y_sparse=y_full,
            )

            half_dt = dt_try * 0.5
            y_half = csr_matrix(network.build_Y(half_dt, Z, T_nodes, M))

            _restore_history_states(network, state_start)
            p_half = _advance_single_step(
                network=network,
                p_old_phys=p_curr,
                dt=half_dt,
                Z=Z,
                T_nodes=T_nodes,
                M=M,
                supply_p=supply_p,
                demand_values=demand_t,
                friction_treatment=friction_treatment,
                y_sparse=y_half,
            )

            demand_t_half = _demand_values_at_time(
                t + half_dt, base_demand_functions, events_by_node
            )
            p_half_2 = _advance_single_step(
                network=network,
                p_old_phys=p_half,
                dt=half_dt,
                Z=Z,
                T_nodes=T_nodes,
                M=M,
                supply_p=supply_p,
                demand_values=demand_t_half,
                friction_treatment=friction_treatment,
                y_sparse=y_half,
            )
            state_half = _snapshot_history_states(network)

            scale = atol + rtol * np.maximum(np.abs(p_half_2), np.abs(p_full))
            scale = np.maximum(scale, 1e-12)
            err = float(np.max(np.abs(p_half_2 - p_full) / scale))

        except Exception:
            err = math.inf

        at_min_step = dt_try <= dt_lower * (1.0 + 1e-12)
        can_accept = (
            p_half_2 is not None
            and state_half is not None
            and np.isfinite(err)
            and (err <= 1.0 or at_min_step)
        )

        if can_accept:
            # Commit the more accurate two-half-step state
            _restore_history_states(network, state_half)
            p_curr = p_half_2
            t += dt_try
            accepted_steps += 1

            _write_back_physical_pressures(network, p_curr)

            while next_output_idx < len(time_out) and t >= time_out[next_output_idx] - eps_time:
                p_hist[:, next_output_idx] = p_curr
                next_output_idx += 1

            err_eff = max(err, 1e-12)
            factor = safety * (1.0 / err_eff) ** 0.5
            factor = min(max(factor, min_factor), max_factor)
            dt = min(dt_max, max(dt_min, dt_try * factor))
            continue

        rejected_steps += 1
        # Roll back to pre-trial history and reduce step size
        _restore_history_states(network, state_start)

        if at_min_step:
            raise RuntimeError(
                f"Adaptive step rejected at minimum step size dt={dt_try:.6g} s, t={t:.6g} s."
            )

        err_eff = max(err, 1e-12) if np.isfinite(err) else 1e12
        factor = safety * (1.0 / err_eff) ** 0.5
        factor = min(max(factor, min_factor), 0.5)
        dt = max(dt_lower, dt_try * factor)

    if next_output_idx < len(time_out):
        p_hist[:, next_output_idx:] = p_curr[:, None]
    time = time_out

    diagnostics = {
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "final_dt_s": float(dt),
    }
    return time, p_hist, diagnostics
