"""
Utilities to build and calibrate dynamic-network cases from CSV inputs.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.constants import R as R_UNIV

try:
    from .create_network import create_network_from_folder
except ImportError:
    from GasNetSim.components.utils.create_network import create_network_from_folder


def fill_missing_pressures(network, default_pressure: Optional[float] = None) -> float:
    """
    Fill missing node pressures with the first available supply pressure.

    If no supply pressure exists, uses ``default_pressure`` if provided,
    otherwise falls back to ``50e5`` Pa.
    """
    supply_pressures = [
        node.pressure for node in network.get_supply_nodes() if node.pressure is not None
    ]
    if supply_pressures:
        default_pressure = float(supply_pressures[0])
    if default_pressure is None:
        default_pressure = 50e5

    for node in network.nodes.values():
        if node.pressure is None:
            if hasattr(node, "set_pressure"):
                node.set_pressure(default_pressure)
            else:
                node.pressure = default_pressure

    return float(default_pressure)


def fill_missing_temperatures(network, default_temperature: float = 288.15):
    """Fill missing node temperatures with ``default_temperature``."""
    for node in network.nodes.values():
        if node.temperature is None:
            node.temperature = default_temperature


def _to_kg_per_mol(molar_mass: float) -> float:
    mm = float(molar_mass)
    # GasMixture often reports molar mass in g/mol.
    if mm > 1.0:
        mm /= 1000.0
    return mm


def estimate_effective_z_m_from_network(
    network,
    default_Z: float = 0.88,
    default_M: float = 0.0168,
) -> Tuple[float, float]:
    """
    Estimate effective constants from a solved steady-state network.

    Uses |q|*L-weighted averages over pipelines, with node-average fallback.
    """
    weighted_sum_z = 0.0
    weighted_sum_m = 0.0
    weighted_total = 0.0

    if network.pipelines is not None:
        for pipe in network.pipelines.values():
            q = getattr(pipe, "flow_rate", None)
            length = float(getattr(pipe, "length", 1.0))
            gas_mixture = getattr(pipe, "gas_mixture", None)
            if q is None or gas_mixture is None:
                continue

            z_val = getattr(gas_mixture, "compressibility", None)
            m_val = getattr(gas_mixture, "molar_mass", None)
            if z_val is None or m_val is None:
                continue

            z_val = float(z_val)
            m_val = _to_kg_per_mol(m_val)
            weight = abs(float(q)) * length

            if (
                np.isfinite(z_val)
                and np.isfinite(m_val)
                and z_val > 0
                and m_val > 0
                and weight > 0
            ):
                weighted_sum_z += weight * z_val
                weighted_sum_m += weight * m_val
                weighted_total += weight

    if weighted_total > 0:
        return weighted_sum_z / weighted_total, weighted_sum_m / weighted_total

    # Fallback: arithmetic average from nodal gas mixtures
    z_vals = []
    m_vals = []
    for node in network.nodes.values():
        gas_mixture = getattr(node, "gas_mixture", None)
        if gas_mixture is None:
            continue

        z_val = getattr(gas_mixture, "compressibility", None)
        m_val = getattr(gas_mixture, "molar_mass", None)
        if z_val is None or m_val is None:
            continue

        z_val = float(z_val)
        m_val = _to_kg_per_mol(m_val)
        if np.isfinite(z_val) and np.isfinite(m_val) and z_val > 0 and m_val > 0:
            z_vals.append(z_val)
            m_vals.append(m_val)

    if z_vals and m_vals:
        return float(np.mean(z_vals)), float(np.mean(m_vals))

    return float(default_Z), float(default_M)


def build_constant_demands_from_csv_volumetric(
    network,
    n_steps: int,
    M: float,
    T_std_K: float = 288.15,
    P_std_Pa: float = 101325.0,
    Z_std: float = 1.0,
):
    """
    Build constant demand profiles [kg/s] from CSV volumetric demands [sm3/s].

    Uses a chosen standard-basis density:
    ``rho_std = P_std * M / (Z_std * R * T_std)``.
    """
    rho_std = float(P_std_Pa) * float(M) / (float(Z_std) * R_UNIV * float(T_std_K))
    demands = {}
    total_q_std = 0.0
    total_m = 0.0

    for node in network.get_demand_nodes():
        q_std = (
            float(node.volumetric_flow)
            if getattr(node, "volumetric_flow", None) is not None
            else 0.0
        )
        m_dot = q_std * rho_std
        demands[node.index] = np.ones(n_steps) * m_dot
        total_q_std += q_std
        total_m += m_dot

    return demands, rho_std, total_q_std, total_m


def configure_constant_pipe_hydraulics(
    network,
    friction_factor: float = 0.01,
    efficiency: float = 1.0,
):
    """Configure constant friction-factor/efficiency on all pipelines."""
    if network.pipelines is None:
        return
    for pipe in network.pipelines.values():
        pipe.friction_factor_method = "constant"
        pipe.constant_friction_factor = float(friction_factor)
        pipe.efficiency = float(efficiency)


def get_static_initial_pressures_from_folder(
    path_to_folder: Path,
    segment_pipes: bool = False,
    segment_length_m: Optional[float] = None,
    segments_per_pipe: Optional[int] = None,
    length_overrides: Optional[Dict[int, float]] = None,
    temperature_interpolator: Optional[
        Callable[[Optional[float], Optional[float], float], Optional[float]]
    ] = None,
    default_Z: float = 0.88,
    default_M: float = 0.0168,
    friction_factor: float = 0.01,
    efficiency: float = 1.0,
    simulation_tol: float = 1e-4,
    return_node_pressures: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Solve steady state and return physical-node pressures with calibrated Z/M.
    """
    static_net = create_network_from_folder(
        path_to_folder,
        dynamic=False,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )
    configure_constant_pipe_hydraulics(
        static_net, friction_factor=friction_factor, efficiency=efficiency
    )
    try:
        static_net.simulation(tol=simulation_tol)
    except Exception as exc:
        print(f"Static init failed: {exc}. Falling back to supply-based P0.")
        if return_node_pressures:
            return None, None, None, None
        return None, None, None

    n_nodes = static_net.n_nodes if hasattr(static_net, "n_nodes") else len(static_net.nodes)
    P0 = np.zeros(n_nodes)
    p0_by_node = {}
    for sim_idx in range(len(static_net.nodes)):
        node_id = static_net.simulation_node_index_to_node_id(sim_idx)
        p_val = static_net.nodes[node_id].pressure
        P0[sim_idx] = p_val
        try:
            p0_by_node[int(node_id)] = float(p_val)
        except Exception:
            continue

    z_eff, m_eff = estimate_effective_z_m_from_network(
        static_net, default_Z=default_Z, default_M=default_M
    )
    if return_node_pressures:
        return P0, z_eff, m_eff, p0_by_node
    return P0, z_eff, m_eff


def build_dynamic_network_from_folder(
    path_to_folder: Path,
    segment_pipes: bool = False,
    segment_length_m: Optional[float] = None,
    segments_per_pipe: Optional[int] = None,
    length_overrides: Optional[Dict[int, float]] = None,
    temperature_interpolator: Optional[
        Callable[[Optional[float], Optional[float], float], Optional[float]]
    ] = None,
    default_pressure: Optional[float] = None,
    default_temperature: float = 288.15,
    friction_factor: float = 0.01,
    efficiency: float = 1.0,
):
    """
    Build a dynamic network from CSV folder and apply common Pi-model defaults.
    """
    net = create_network_from_folder(
        path_to_folder,
        dynamic=True,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )
    fill_missing_pressures(net, default_pressure=default_pressure)
    fill_missing_temperatures(net, default_temperature=default_temperature)
    configure_constant_pipe_hydraulics(
        net, friction_factor=friction_factor, efficiency=efficiency
    )
    return net
