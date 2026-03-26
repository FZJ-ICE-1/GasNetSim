"""
Experimental SUNDIALS-based transient simulation backend.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.constants import R as R_UNIV

try:
    from ..components.dynamic.dynamic_network import DynamicNetwork
except ImportError:
    from GasNetSim.components.dynamic.dynamic_network import DynamicNetwork


def _require_sksundae():
    try:
        import sksundae as sun  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SUNDIALS Python bindings are not installed. "
            "Install scikit-SUNDAE first, e.g. `pip install scikit-sundae`."
        ) from exc
    return sun


def _node_pressure_from_state(
    sim_idx: int,
    p_var: np.ndarray,
    sim_to_var_idx: Dict[int, int],
    fixed_pressures: Dict[int, float],
) -> float:
    # Free nodes come from CVODE state; supply-coupled nodes stay fixed
    if sim_idx in sim_to_var_idx:
        return float(p_var[sim_to_var_idx[sim_idx]])
    return float(fixed_pressures[sim_idx])


def _build_base_demand_functions(
    network: DynamicNetwork,
    demands: Optional[Dict[int, np.ndarray]],
    demand_dt: Optional[float],
) -> Dict[int, Callable[[float], float]]:
    # Build continuous-time demand callbacks so CVODE can sample at internal times
    functions: Dict[int, Callable[[float], float]] = {}

    if demands is None:
        for node in network.get_demand_nodes():
            global_idx = network.node_id_to_global_index(node.index)
            if hasattr(node, "mass_flow"):
                m_dot = float(node.mass_flow)
            elif node.volumetric_flow is not None:
                m_dot = float(node.volumetric_flow)
            else:
                m_dot = 0.0
            functions[global_idx] = (lambda value: (lambda _t: value))(m_dot)
        return functions

    for node_id, profile in demands.items():
        global_idx = network.node_id_to_global_index(node_id)
        arr = np.asarray(profile, dtype=float)
        if arr.ndim == 0 or arr.size == 1:
            value = float(arr.reshape(-1)[0])
            functions[global_idx] = (lambda v: (lambda _t: v))(value)
        else:
            if demand_dt is None or demand_dt <= 0:
                raise ValueError(
                    "demand_dt must be provided and positive when demands are time-series arrays."
                )
            local_arr = arr.copy()

            def profile_fn(t, values=local_arr, dt=demand_dt):
                idx = int(np.floor(max(t, 0.0) / dt))
                idx = min(idx, len(values) - 1)
                return float(values[idx])

            functions[global_idx] = profile_fn

    return functions


def _normalize_events(
    network: DynamicNetwork,
    events: Optional[List[Dict]],
    demand_dt: Optional[float],
) -> Dict[int, List[Tuple[float, str, float]]]:
    # Convert event definitions to sorted per-node time operations
    if not events:
        return {}

    by_node: Dict[int, List[Tuple[float, str, float]]] = {}
    for event in events:
        if "node" in event:
            global_idx = network.node_id_to_global_index(event["node"])
        elif "global_idx" in event:
            global_idx = int(event["global_idx"])
        else:
            continue

        if "time" in event:
            t_event = float(event["time"])
        elif "step" in event:
            if demand_dt is None or demand_dt <= 0:
                raise ValueError(
                    "demand_dt must be provided when events specify 'step' for SUNDIALS mode."
                )
            t_event = float(event["step"]) * demand_dt
        else:
            continue

        if "mult" in event:
            op, value = "mult", float(event["mult"])
        elif "add" in event:
            op, value = "add", float(event["add"])
        elif "set" in event:
            op, value = "set", float(event["set"])
        else:
            continue

        by_node.setdefault(global_idx, []).append((t_event, op, value))

    for global_idx in by_node:
        by_node[global_idx].sort(key=lambda x: x[0])
    return by_node


def _demand_at_time(
    t: float,
    base_functions: Dict[int, Callable[[float], float]],
    events_by_node: Dict[int, List[Tuple[float, str, float]]],
) -> Dict[int, float]:
    # Compose base demands with all events active at time t
    demand_values: Dict[int, float] = {}
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
        demand_values[global_idx] = value
    return demand_values


def _collect_discontinuity_times(
    t_end: float,
    demands: Optional[Dict[int, np.ndarray]],
    demand_dt: Optional[float],
    events_by_node: Dict[int, List[Tuple[float, str, float]]],
) -> np.ndarray:
    # Gather demand-profile jump times and explicit event times.
    times: List[float] = []

    if demands is not None and demand_dt is not None and demand_dt > 0:
        max_len = 1
        for profile in demands.values():
            arr = np.asarray(profile, dtype=float).reshape(-1)
            max_len = max(max_len, int(arr.size))
        for k in range(1, max_len):
            t_k = float(k) * float(demand_dt)
            if 0.0 < t_k < float(t_end):
                times.append(t_k)

    for node_events in events_by_node.values():
        for t_event, _op, _operand in node_events:
            if 0.0 < float(t_event) < float(t_end):
                times.append(float(t_event))

    if not times:
        return np.array([], dtype=float)
    return np.array(sorted(set(times)), dtype=float)


def _normalize_atol(
    atol,
    y0: np.ndarray,
    n_free: int,
) -> np.ndarray | float:
    # CVODE tolerances should be scaled to state magnitudes for mixed-unit states.
    if np.isscalar(atol):
        atol_val = float(atol)
        if atol_val <= 0:
            raise ValueError("atol must be positive.")
        atol_vec = np.full_like(y0, atol_val, dtype=float)
        if y0.size > 0:
            p_scale = float(np.nanmedian(np.abs(y0[:n_free]))) if n_free > 0 else 1.0
            m_scale = (
                float(np.nanmedian(np.abs(y0[n_free:])))
                if y0.size > n_free
                else 1.0
            )
            p_scale = max(1.0, p_scale)
            m_scale = max(1.0, m_scale)
            # Keep user scalar as a floor, but avoid unrealistically tiny absolute
            # tolerances on large pressure states.
            atol_vec[:n_free] = max(atol_val, 1e-6 * p_scale)
            atol_vec[n_free:] = max(atol_val, 1e-6 * m_scale)
        return atol_vec

    atol_arr = np.asarray(atol, dtype=float).reshape(-1)
    if atol_arr.size != y0.size:
        raise ValueError(
            f"atol vector length must match state size ({y0.size}), got {atol_arr.size}."
        )
    if not np.all(atol_arr > 0):
        raise ValueError("all atol values must be positive.")
    return atol_arr


def _make_cvode_solver(
    sun,
    rhsfn,
    rtol: float,
    atol_eff,
    n_free: int,
):
    # Prefer nonnegative constraints on pressure states when supported.
    kwargs = {"rtol": rtol, "atol": atol_eff}
    use_constraints = n_free > 0
    if use_constraints:
        kwargs["constraints_idx"] = np.arange(n_free, dtype=int)
        kwargs["constraints_type"] = np.ones(n_free, dtype=int)  # y[i] >= 0

    try:
        return sun.cvode.CVODE(rhsfn, method="BDF", **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("constraints_idx", None)
        kwargs.pop("constraints_type", None)
        try:
            return sun.cvode.CVODE(rhsfn, method="BDF", **kwargs)
        except (TypeError, ValueError):
            return sun.cvode.CVODE(rhsfn, lmm="BDF", **kwargs)


def simulate_transient_sundials(
    network: DynamicNetwork,
    t_end: float,
    Z: float,
    M: float,
    P0: Optional[np.ndarray] = None,
    T_nodes: Optional[np.ndarray] = None,
    supply_pressures: Optional[np.ndarray] = None,
    demands: Optional[Dict[int, np.ndarray]] = None,
    demand_dt: Optional[float] = None,
    events: Optional[List[Dict]] = None,
    output_times: Optional[Sequence[float]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """
    Simulate transients with SUNDIALS CVODE using a reduced ODE formulation.

    Notes:
      - This is an adaptive-time experimental backend.
      - Supply-node pressures are treated as fixed boundary conditions.
      - State vector = [free physical-node pressures, pipe mass-flow-like states].
    """
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    if rtol <= 0:
        raise ValueError("rtol must be positive.")

    sun = _require_sksundae()

    n_virtual = network.n_virtual_nodes
    n_phys = network.n_physical_nodes
    n_pipes = len(network.pipes)
    if n_pipes == 0:
        raise ValueError("SUNDIALS transient simulation requires at least one pipeline.")

    if T_nodes is None:
        T = network.get_temperature_vector()
    else:
        T = np.array(T_nodes, dtype=float)
    if len(T) != network.n_nodes:
        raise ValueError("T_nodes length must match total node count.")

    if supply_pressures is None:
        supply_p = np.array(network.get_supply_pressures(), dtype=float)
    elif np.isscalar(supply_pressures):
        supply_p = np.full(network.n_virtual_nodes, float(supply_pressures))
    else:
        supply_p = np.array(supply_pressures, dtype=float)
    if len(supply_p) != network.n_virtual_nodes:
        raise ValueError("supply_pressures length must match number of supply nodes.")

    if P0 is None:
        p_phys0 = np.zeros(n_phys)
        for sim_idx in range(n_phys):
            node_id = network.simulation_node_index_to_node_id(sim_idx)
            node = network.nodes[node_id]
            p_phys0[sim_idx] = float(node.pressure if node.pressure is not None else 0.0)
    else:
        p_phys0 = np.array(P0, dtype=float)
    if len(p_phys0) != n_phys:
        raise ValueError("P0 length must match number of physical nodes.")

    supply_sim_indices = sorted(
        list({int(p_idx - n_virtual) for _, p_idx in network.virtual_connections})
    )
    # Eliminate fixed-pressure states from ODE unknowns to reduce system size
    free_sim_indices = [i for i in range(n_phys) if i not in supply_sim_indices]
    sim_to_var_idx = {sim_idx: k for k, sim_idx in enumerate(free_sim_indices)}
    n_free = len(free_sim_indices)

    fixed_pressures = {sim_idx: float(p_phys0[sim_idx]) for sim_idx in supply_sim_indices}
    for v_idx, p_idx in network.virtual_connections:
        sim_idx = int(p_idx - n_virtual)
        fixed_pressures[sim_idx] = float(supply_p[v_idx])

    y0 = np.zeros(n_free + n_pipes)
    for sim_idx in free_sim_indices:
        y0[sim_to_var_idx[sim_idx]] = p_phys0[sim_idx]
    # Initialize flow-like states from pressure differences (quasi-static guess)
    for k in range(n_pipes):
        i_sim = int(network.i_idx[k] - n_virtual)
        j_sim = int(network.j_idx[k] - n_virtual)
        p_i = p_phys0[i_sim] if i_sim in sim_to_var_idx else fixed_pressures[i_sim]
        p_j = p_phys0[j_sim] if j_sim in sim_to_var_idx else fixed_pressures[j_sim]
        y0[n_free + k] = network.GLc[k] * (p_i - p_j)

    base_demand_fns = _build_base_demand_functions(network, demands, demand_dt=demand_dt)
    events_by_node = _normalize_events(network, events, demand_dt=demand_dt)

    node_global_idx = {sim_idx: sim_idx + n_virtual for sim_idx in range(n_phys)}

    c_node = np.zeros(n_free)
    incident_signs: List[List[Tuple[int, float]]] = [[] for _ in range(n_free)]
    fric0 = np.zeros(n_pipes)
    grav0 = np.zeros(n_pipes)
    i_sim_arr = np.zeros(n_pipes, dtype=int)
    j_sim_arr = np.zeros(n_pipes, dtype=int)

    for k, pipe in enumerate(network.pipes):
        i_global = int(network.i_idx[k])
        j_global = int(network.j_idx[k])
        i_sim = int(i_global - n_virtual)
        j_sim = int(j_global - n_virtual)
        i_sim_arr[k] = i_sim
        j_sim_arr[k] = j_sim

        c_i = (M * network.GCc[k]) / (Z * R_UNIV * T[i_global])
        c_j = (M * network.GCc[k]) / (Z * R_UNIV * T[j_global])

        if i_sim in sim_to_var_idx:
            vi = sim_to_var_idx[i_sim]
            c_node[vi] += c_i
            incident_signs[vi].append((k, -1.0))
        if j_sim in sim_to_var_idx:
            vj = sim_to_var_idx[j_sim]
            c_node[vj] += c_j
            incident_signs[vj].append((k, +1.0))

        area = getattr(pipe, "A", None)
        if area is None:
            area = np.pi * (pipe.diameter / 2.0) ** 2
        pipe_d = getattr(pipe, "D", None)
        if pipe_d is None:
            pipe_d = pipe.diameter
        f_val = float(pipe.f) if pipe.f is not None else 0.01
        fric0[k] = (f_val * Z * R_UNIV * T[i_global]) / (M * area * pipe_d)

        # Gravity coupling term (matches Pi-model g_theta structure with p_i + p_j)
        t_avg = 0.5 * (float(T[i_global]) + float(T[j_global]))
        t_avg = max(t_avg, 1e-12)
        grav0[k] = (M * network.Ggc[k]) / (2.0 * Z * R_UNIV * t_avg)

    c_node = np.maximum(c_node, 1e-12)
    glc = network.GLc.copy()

    def rhsfn(t, y, yp):
        # y = [free pressures, pipe flow-like states]
        p_var = y[:n_free]
        m_pipe = y[n_free:]

        demand_now = _demand_at_time(t, base_demand_fns, events_by_node)

        # Flow-state derivatives (pipe momentum-like dynamics)
        for k in range(n_pipes):
            i_sim = i_sim_arr[k]
            j_sim = j_sim_arr[k]
            p_i = _node_pressure_from_state(i_sim, p_var, sim_to_var_idx, fixed_pressures)
            p_j = _node_pressure_from_state(j_sim, p_var, sim_to_var_idx, fixed_pressures)
            p_avg = max(p_i + p_j, 1e5)
            m = m_pipe[k]
            g_term = grav0[k] * (p_i + p_j)
            yp[n_free + k] = (
                glc[k] * (p_i - p_j)
                - g_term
                - fric0[k] * m * abs(m) / p_avg
            )

        # Pressure derivatives (nodal mass-balance-like dynamics)
        for vi, sim_idx in enumerate(free_sim_indices):
            net_flow = 0.0
            for k, sign in incident_signs[vi]:
                net_flow += sign * m_pipe[k]
            # Nodal pressure rate from net flow minus demand
            dmdt_node = net_flow - demand_now.get(node_global_idx[sim_idx], 0.0)
            yp[vi] = dmdt_node / c_node[vi]

    if output_times is None:
        tspan = np.array([0.0, float(t_end)], dtype=float)
    else:
        tspan = np.array(output_times, dtype=float)
        if len(tspan) < 2:
            raise ValueError("output_times must contain at least two points.")
        if not np.all(np.diff(tspan) > 0):
            raise ValueError("output_times must be strictly increasing.")
        if abs(float(tspan[0])) > 1e-12:
            raise ValueError("output_times must start at 0.0.")
        if abs(float(tspan[-1]) - float(t_end)) > 1e-10:
            raise ValueError("output_times must end at t_end.")

    atol_eff = _normalize_atol(atol=atol, y0=y0, n_free=n_free)
    break_times = _collect_discontinuity_times(
        t_end=t_end,
        demands=demands,
        demand_dt=demand_dt,
        events_by_node=events_by_node,
    )

    # Segment integration at known discontinuities (demand profile jumps/events).
    seg_points = np.concatenate(([0.0], break_times, [float(t_end)]))
    seg_points = np.unique(seg_points)

    t_sol = tspan.copy()
    p_hist = np.zeros((n_phys, len(t_sol)), dtype=float)
    filled = np.zeros(len(t_sol), dtype=bool)
    t_to_idx = {round(float(t), 12): i for i, t in enumerate(t_sol)}
    y_curr = y0.copy()

    # Fill initial output from y0.
    idx0 = t_to_idx.get(round(0.0, 12))
    if idx0 is not None:
        for sim_idx in range(n_phys):
            if sim_idx in sim_to_var_idx:
                p_hist[sim_idx, idx0] = y0[sim_to_var_idx[sim_idx]]
            else:
                p_hist[sim_idx, idx0] = fixed_pressures[sim_idx]
        filled[idx0] = True

    for s in range(len(seg_points) - 1):
        t0 = float(seg_points[s])
        t1 = float(seg_points[s + 1])
        if t1 <= t0:
            continue

        mask = (tspan > t0 + 1e-12) & (tspan <= t1 + 1e-12)
        local_targets = tspan[mask]
        if local_targets.size == 0:
            tspan_seg = np.array([t0, t1], dtype=float)
        else:
            tspan_seg = np.concatenate(([t0], local_targets))
            if abs(float(tspan_seg[-1]) - t1) > 1e-10:
                tspan_seg = np.append(tspan_seg, t1)

        solver = _make_cvode_solver(
            sun=sun,
            rhsfn=rhsfn,
            rtol=rtol,
            atol_eff=atol_eff,
            n_free=n_free,
        )
        soln_seg = solver.solve(tspan_seg, y_curr)
        if hasattr(soln_seg, "success") and not bool(soln_seg.success):
            msg = getattr(soln_seg, "message", "unknown SUNDIALS failure")
            raise RuntimeError(
                f"SUNDIALS segment failed on [{t0:.6g}, {t1:.6g}] s: {msg}"
            )
        t_seg = np.array(soln_seg.t, dtype=float)
        y_seg = np.array(soln_seg.y, dtype=float)
        if y_seg.ndim != 2 or y_seg.shape[0] == 0:
            raise RuntimeError(
                f"SUNDIALS segment returned empty state history on [{t0:.6g}, {t1:.6g}] s."
            )
        if not np.all(np.isfinite(y_seg)):
            raise RuntimeError(
                f"SUNDIALS segment produced non-finite states on [{t0:.6g}, {t1:.6g}] s."
            )
        y_curr = y_seg[-1, :].copy()

        for row_idx, t_val in enumerate(t_seg):
            idx = t_to_idx.get(round(float(t_val), 12))
            if idx is None:
                continue
            for sim_idx in range(n_phys):
                if sim_idx in sim_to_var_idx:
                    p_hist[sim_idx, idx] = y_seg[row_idx, sim_to_var_idx[sim_idx]]
                else:
                    p_hist[sim_idx, idx] = fixed_pressures[sim_idx]
            filled[idx] = True

    # If any outputs were missed due solver internals, carry forward previous value.
    for i in range(1, len(t_sol)):
        if not filled[i]:
            p_hist[:, i] = p_hist[:, i - 1]
            filled[i] = True

    # Write final pressures back to network nodes
    for sim_idx in range(n_phys):
        node_id = network.simulation_node_index_to_node_id(sim_idx)
        node = network.nodes[node_id]
        p_val = float(p_hist[sim_idx, -1])
        if hasattr(node, "set_pressure"):
            node.set_pressure(p_val)
        else:
            node.pressure = p_val

    return t_sol, p_hist
