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

def _infer_demand_nodes(network):
    demand_ids = []
    for node in network.nodes.values():
        node_type = str(node.node_type).lower() if node.node_type is not None else ""
        if node_type in {"reference", "supply", "slack"}:
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


def _fill_missing_temperatures(network, default_temperature=300.0):
    for node in network.nodes.values():
        if node.temperature is None:
            node.temperature = default_temperature


def _get_static_initial_pressures(data_dir):
    static_net = create_network_from_folder(data_dir, dynamic=False)
    for pipe in static_net.pipelines.values():
        pipe.friction_factor_method = "constant"
        pipe.constant_friction_factor = 0.01
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


def main():
    data_dir = Path(__file__).resolve().parent
    net = create_network_from_folder(
        data_dir,
        dynamic=True,
    )

    _fill_missing_pressures(net)
    _fill_missing_temperatures(net)

    # Force constant friction factor for all pipes (Pi-model assumption)
    for pipe in net.pipelines.values():
        pipe.friction_factor_method = "constant"
        pipe.constant_friction_factor = 0.01

    dt = 20.0
    t_end = 100000.0
    n_steps = int(t_end / dt)

    # Infer demand nodes from CSV flow_type == volumetric
    demand_ids = _infer_demand_nodes(net)

    demands = {}
    for node_id in demand_ids:
        node = net.nodes[node_id]
        base = node.mass_flow
        demands[node_id] = np.ones(n_steps) * base

    use_static_init = True
    P0 = None
    if use_static_init:
        P0 = _get_static_initial_pressures(data_dir)

    events = [
        {"node": 10, "time": 3600.0, "mult": 1.2},
        {"node": 6, "time": 7200.0, "add": 2.0},
    ]

    time, P_hist = simulate_transient(
        net,
        dt=dt,
        t_end=t_end,
        Z=1.0,
        M=0.016,
        demands=demands,
        P0=P0,
        events=events,
    )

    print("Simulation finished.")

    # Plot transient pressures (bar gauge) for all physical nodes
    plt.figure(figsize=(12, 6))
    for i in range(net.n_physical_nodes):
        plt.plot(time / 3600, P_hist[i] / 1e5 - 1.01325, label=f"Node {i + 1}")
    plt.xlabel("Time [hours]")
    plt.ylabel("Pressure [bar gauge]")
    plt.title("Irish 13-Node Network: Transient Pressure Profiles")
    plt.legend(fontsize="small", ncol=3, loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compare steady-state to reference pressures (bar gauge)
    reference_method = "novel"  # "saint", "matlab", or "novel"
    ref = IRISH13_REFERENCE_PRESSURES[reference_method]
    P_ss = P_hist[:, -1] / 1e5 - 1.01325
    diff = P_ss - ref

    print("\nSteady-State Pressure Comparison")
    print("-" * 70)
    print(f"{'Node':<8} {'Simulated [barg]':<18} {'Reference [barg]':<18} {'Diff [bar]':<12}")
    print("-" * 70)
    for i, (sim, ref, d) in enumerate(
        zip(P_ss, ref, diff), start=1
    ):
        marker = "  *" if abs(d) > 0.3 else ""
        print(f"{i:<8} {sim:<18.2f} {ref:<18.2f} {d:<+12.2f}{marker}")
    print("-" * 70)
    print(f"{'Max absolute diff:':<44} {np.max(np.abs(diff)):.2f} bar")
    print(f"{'Mean absolute diff:':<44} {np.mean(np.abs(diff)):.2f} bar")
    print(f"{'RMS diff:':<44} {np.sqrt(np.mean(diff**2)):.2f} bar")


if __name__ == "__main__":
    main()
