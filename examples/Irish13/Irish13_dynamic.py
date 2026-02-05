"""
Dynamic Pi-model run for Irish13 using CSV inputs.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from GasNetSim.components.utils.create_network import create_network_from_folder
from GasNetSim.simulation.dynamic import simulate_transient

# Table 3 (Ekhtiari et al., 2019) reference pressures in bar gauge
IRISH13_REFERENCE_PRESSURES = {
    "saint": np.array(
        [
            70.00,
            70.00,
            70.00,
            69.66,
            68.32,
            66.26,
            66.07,
            66.04,
            66.02,
            65.96,
            67.89,
            67.85,
            67.45,
        ]
    ),
    "matlab": np.array(
        [
            70.00,
            70.00,
            70.00,
            69.41,
            68.27,
            66.58,
            66.53,
            66.51,
            66.50,
            66.19,
            67.89,
            67.89,
            67.89,
        ]
    ),
    "novel": np.array(
        [
            70.00,
            70.00,
            70.00,
            69.10,
            67.93,
            66.48,
            65.93,
            66.23,
            66.24,
            66.00,
            67.72,
            67.43,
            67.79,
        ]
    ),
}

# Helper functions to preprocess the network data and extract demand profiles from CSV inputs

# Mark nodes with flow_type == volumetric as demand nodes
def _infer_demand_nodes(network):
    demand_ids = []
    for node in network.nodes.values():
        node_type = str(node.node_type).lower() if node.node_type is not None else ""
        if node_type in {"reference", "supply", "slack", "junction"}:
            continue
        flow_type = (
            str(node.flow_type).lower() if node.flow_type is not None else ""
        )
        if flow_type == "volumetric":
            node.node_type = "demand"
            demand_ids.append(node.index)
    return demand_ids

def _fill_missing_pressures(network, default_pressure=None):
    supply_pressures = [
        node.pressure
        for node in network.get_supply_nodes()
        if node.pressure is not None
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

    return default_pressure


def _fill_missing_temperatures(network, default_temperature=288.15):
    for node in network.nodes.values():
        if node.temperature is None:
            node.temperature = default_temperature


def _get_static_initial_pressures(
    data_dir,
    segment_pipes=False,
    segment_length_m=None,
    segments_per_pipe=None,
    length_overrides=None,
    temperature_interpolator=None,
):
    static_net = create_network_from_folder(
        data_dir,
        dynamic=False,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )
    for pipe in static_net.pipelines.values():
        pipe.friction_factor_method = "constant"
        pipe.constant_friction_factor = 0.01
        pipe.efficiency = 1.0
    try:
        static_net.simulation(tol=1e-4)
    except Exception as e:
        print(f"Static init failed: {e}. Falling back to supply-based P0.")
        return None

    P0 = np.zeros(static_net.n_nodes if hasattr(static_net, "n_nodes") else len(static_net.nodes))
    for sim_idx in range(len(static_net.nodes)):
        node_id = static_net.simulation_node_index_to_node_id(sim_idx)
        P0[sim_idx] = static_net.nodes[node_id].pressure
    return P0


def _build_network(
    data_dir,
    segment_pipes=False,
    segment_length_m=None,
    segments_per_pipe=None,
    length_overrides=None,
    temperature_interpolator=None,
):
    net = create_network_from_folder(
        data_dir,
        dynamic=True,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )

    _fill_missing_pressures(net)
    _fill_missing_temperatures(net)

    # Constant friction factor for all pipes (Pi-model assumption)
    for pipe in net.pipelines.values():
        pipe.friction_factor_method = "constant"
        pipe.constant_friction_factor = 0.01
        pipe.efficiency = 1.0

    return net


def main():
    data_dir = Path(__file__).resolve().parent
    segment_pipes = True
    segment_length_m = 20_000.0 
    segments_per_pipe = None
    length_overrides = None
    temperature_interpolator = None
    compare_to_reference = False
    compare_to_static = True

    net_base = _build_network(
        data_dir,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )
    net_pert = _build_network(
        data_dir,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )

    dt = 25.0
    t_end = 100000.0
    n_steps = int(t_end / dt)

    demand_ids = _infer_demand_nodes(net_base)

    demands = {}
    for node_id in demand_ids:
        node = net_base.nodes[node_id]
        base = node.mass_flow
        demands[node_id] = np.ones(n_steps) * base

    use_static_init = True
    P0 = None
    if use_static_init:
        P0 = _get_static_initial_pressures(
            data_dir,
            segment_pipes=segment_pipes,
            segment_length_m=segment_length_m,
            segments_per_pipe=segments_per_pipe,
            length_overrides=length_overrides,
            temperature_interpolator=temperature_interpolator,
        )
        if P0 is not None and len(P0) != net_base.n_physical_nodes:
            print(
                "Static init length mismatch; falling back to supply-based P0."
            )
            P0 = None

    # Baseline run (no perturbations)
    time_base, P_hist_base = simulate_transient(
        net_base,
        dt=dt,
        t_end=t_end,
        Z=0.89,
        M=0.016,
        demands=demands,
        P0=P0,
        events=[],
    )

    events = [
        {"node": 10, "time": 3600.0, "mult": 1.2},
        {"node": 6, "time": 7200.0, "add": 2.0},
        {"node": 9, "time": 10800.0, "mult": 0.85},
        {"node": 12, "time": 14400.0, "add": -1.5},
        {"node": 8, "time": 21600.0, "mult": 1.1},
    ]

    time, P_hist = simulate_transient(
        net_pert,
        dt=dt,
        t_end=t_end,
        Z=0.89,
        M=0.016,
        demands=demands,
        P0=P0,
        events=events,
    )

    print("Simulation finished.")

    # Plot transient pressures (bar gauge) for original Irish13 nodes only
    plt.figure(figsize=(12, 6))
    ref_node_ids = list(range(1, len(IRISH13_REFERENCE_PRESSURES["novel"]) + 1))
    ref_sim_idx = [net_pert.node_id_to_simulation_node_index(nid) for nid in ref_node_ids]
    for node_id, sim_idx in zip(ref_node_ids, ref_sim_idx):
        plt.plot(
            time / 3600,
            P_hist[sim_idx] / 1e5 - 1.01325,
            label=f"Node {node_id}",
        )
    plt.xlabel("Time [hours]")
    plt.ylabel("Pressure [bar gauge]")
    plt.title("Irish 13-Node Network: Transient Pressure Profiles")
    plt.legend(fontsize="small", ncol=3, loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compare steady-state to baseline (bar gauge)
    ref_node_ids = list(range(1, len(IRISH13_REFERENCE_PRESSURES["novel"]) + 1))
    ref_sim_idx = [net_pert.node_id_to_simulation_node_index(nid) for nid in ref_node_ids]
    P_ss_base_all = P_hist_base[:, -1] / 1e5 - 1.01325
    P_ss_pert_all = P_hist[:, -1] / 1e5 - 1.01325
    P_ss_base = P_ss_base_all[ref_sim_idx]
    P_ss_pert = P_ss_pert_all[ref_sim_idx]
    diff = P_ss_pert - P_ss_base

    print("\nSteady-State Pressure Comparison (Perturbed vs Baseline)")
    print("-" * 86)
    print(
        f"{'Node':<8} {'Baseline [barg]':<18} {'Perturbed [barg]':<20} {'Diff [bar]':<12}"
    )
    print("-" * 86)
    for i, (base, pert, d) in enumerate(zip(P_ss_base, P_ss_pert, diff), start=1):
        marker = "  *" if abs(d) > 0.3 else ""
        print(f"{i:<8} {base:<18.2f} {pert:<20.2f} {d:<+12.2f}{marker}")
    print("-" * 86)
    print(f"{'Max absolute diff:':<44} {np.max(np.abs(diff)):.2f} bar")
    print(f"{'Mean absolute diff:':<44} {np.mean(np.abs(diff)):.2f} bar")
    print(f"{'RMS diff:':<44} {np.sqrt(np.mean(diff**2)):.2f} bar")

    # Baseline vs steady-state solver comparison
    if P0 is None:
        print("\nStatic baseline unavailable (static init failed).")
    else:
        P_ss_static_all = P0 / 1e5 - 1.01325
        P_ss_static = P_ss_static_all[ref_sim_idx]
        diff_base_static = P_ss_base - P_ss_static

        print("\nSteady-State Pressure Comparison (Baseline vs Steady-State Solver)")
        print("-" * 86)
        print(
            f"{'Node':<8} {'Static [barg]':<18} {'Baseline [barg]':<20} {'Diff [bar]':<12}"
        )
        print("-" * 86)
        for i, (stat, base, d) in enumerate(
            zip(P_ss_static, P_ss_base, diff_base_static), start=1
        ):
            marker = "  *" if abs(d) > 0.3 else ""
            print(f"{i:<8} {stat:<18.2f} {base:<20.2f} {d:<+12.2f}{marker}")
        print("-" * 86)
        print(f"{'Max absolute diff:':<44} {np.max(np.abs(diff_base_static)):.2f} bar")
        print(f"{'Mean absolute diff:':<44} {np.mean(np.abs(diff_base_static)):.2f} bar")
        print(f"{'RMS diff:':<44} {np.sqrt(np.mean(diff_base_static**2)):.2f} bar")

    if compare_to_static:
        if P0 is None:
            print("\nStatic baseline unavailable (static init failed).")
        else:
            P_ss_static_all = P0 / 1e5 - 1.01325
            P_ss_static = P_ss_static_all[ref_sim_idx]
            diff_static = P_ss_pert - P_ss_static

            print("\nSteady-State Pressure Comparison (Perturbed vs Steady-State Solver)")
            print("-" * 86)
            print(
                f"{'Node':<8} {'Static [barg]':<18} {'Perturbed [barg]':<20} {'Diff [bar]':<12}"
            )
            print("-" * 86)
            for i, (stat, pert, d) in enumerate(
                zip(P_ss_static, P_ss_pert, diff_static), start=1
            ):
                marker = "  *" if abs(d) > 0.3 else ""
                print(f"{i:<8} {stat:<18.2f} {pert:<20.2f} {d:<+12.2f}{marker}")
            print("-" * 86)
            print(f"{'Max absolute diff:':<44} {np.max(np.abs(diff_static)):.2f} bar")
            print(f"{'Mean absolute diff:':<44} {np.mean(np.abs(diff_static)):.2f} bar")
            print(f"{'RMS diff:':<44} {np.sqrt(np.mean(diff_static**2)):.2f} bar")

    if compare_to_reference:
        reference_method = "novel"  # "saint", "matlab", or "novel"
        ref = IRISH13_REFERENCE_PRESSURES[reference_method]
        diff_ref = P_ss_pert - ref
        print("\nSteady-State vs Reference (not valid if perturbed)")
        print("-" * 70)
        print(
            f"{'Node':<8} {'Simulated [barg]':<18} {'Reference [barg]':<18} {'Diff [bar]':<12}"
        )
        print("-" * 70)
        for i, (sim, ref_val, d) in enumerate(zip(P_ss_pert, ref, diff_ref), start=1):
            marker = "  *" if abs(d) > 0.3 else ""
            print(f"{i:<8} {sim:<18.2f} {ref_val:<18.2f} {d:<+12.2f}{marker}")
        print("-" * 70)


if __name__ == "__main__":
    main()
