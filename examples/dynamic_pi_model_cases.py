"""
Run Pi-model dynamic validation cases.
"""

import numpy as np
import matplotlib.pyplot as plt

from GasNetSim.components.dynamic.dynamic_node import DynamicNode
from GasNetSim.components.dynamic.dynamic_pipeline import DynamicPipeline
from GasNetSim.components.dynamic.dynamic_network import DynamicNetwork
from GasNetSim.simulation.dynamic import simulate_transient


def build_segmented_network(
    length_total,
    diameter,
    friction,
    n_segments=None,
    segment_length_m=None,
    supply_at_outlet=False,
):
    if n_segments is None:
        if segment_length_m is None:
            raise ValueError("Provide n_segments or segment_length_m.")
        if segment_length_m <= 0:
            raise ValueError("segment_length_m must be > 0.")
        n_segments = int(np.ceil(length_total / segment_length_m))
    if int(n_segments) < 1:
        raise ValueError("n_segments must be >= 1.")
    n_segments = int(n_segments)
    n_nodes = n_segments + 1
    nodes = {}

    for i in range(1, n_nodes + 1):
        node_type = "junction"
        if i == 1 and not supply_at_outlet:
            node_type = "reference"
        elif i == n_nodes and supply_at_outlet:
            node_type = "reference"
        elif i == 1 and supply_at_outlet:
            node_type = "demand"
        elif i == n_nodes and not supply_at_outlet:
            node_type = "demand"

        nodes[i] = DynamicNode(
            node_index=i,
            pressure_pa=50e5,
            node_type=node_type,
        )

    segment_length = length_total / n_segments
    pipes = {
        i: DynamicPipeline(
            pipeline_index=i,
            inlet=nodes[i],
            outlet=nodes[i + 1],
            diameter=diameter,
            length=segment_length,
            friction_factor_method="constant",
            constant_friction_factor=friction,
        )
        for i in range(1, n_nodes)
    }

    return DynamicNetwork(nodes=nodes, pipelines=pipes)


def case_single_segment():
    print("\nSingle Segment Pipeline Test")
    diameter = 0.6
    friction = 0.01
    length_total = 50e3
    n_segments = 10

    net = build_segmented_network(
        n_segments=n_segments,
        length_total=length_total,
        diameter=diameter,
        friction=friction,
        supply_at_outlet=False,
    )

    t_end = 3000.0
    supply_p = 50e5

    plt.figure(figsize=(8, 5))
    for dt in [1, 10, 30]:
        n_steps = int(t_end / dt)
        demand_profile = np.ones(n_steps) * 20.0
        demand_profile[int(1500 / dt) :] = 10.0

        time_sim, P_hist = simulate_transient(
            net,
            dt=dt,
            t_end=t_end,
            Z=1.0,
            M=0.016,
            P0=np.full(n_segments + 1, 50e5),
            supply_pressures=supply_p,
            demands={n_segments + 1: demand_profile},
        )
        plt.plot(time_sim, P_hist[-1] / 1e6, label=f"dt = {dt} s")

    plt.axvline(x=1500, color="r", linestyle="--", alpha=0.5, label="Demand change")
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [MPa]")
    plt.title("Single Segment Pipeline - Convergence Test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def case_forward_flow():
    print("\nCase 1: Forward Flow with Demand Change")
    diameter = 0.6
    friction = 0.01
    length_total = 50e3
    segment_length_m = 10e3
    n_segments = int(np.ceil(length_total / segment_length_m))
    t_end = 3000.0
    supply_p = 50e5

    net = build_segmented_network(
        segment_length_m=segment_length_m,
        length_total=length_total,
        diameter=diameter,
        friction=friction,
        supply_at_outlet=False,
    )

    plt.figure(figsize=(12, 5))
    P_hist_case1_final = None
    P0_at_1500 = None
    P_supply_at_1500 = None

    for dt in [1, 10, 30]:
        n_steps = int(t_end / dt)
        demand_profile = np.ones(n_steps) * 20.0
        events = [{"node": n_segments + 1, "time": 1500.0, "mult": 0.5}]

        time_sim, P_hist = simulate_transient(
            net,
            dt=dt,
            t_end=t_end,
            Z=1.0,
            M=0.016,
            P0=np.ones(n_segments + 1) * 50e5,
            supply_pressures=supply_p,
            demands={n_segments + 1: demand_profile},
            events=events,
        )
        plt.plot(time_sim, P_hist[-1] / 1e6, label=f"dt = {dt} s")

        if dt == 30:
            P_hist_case1_final = P_hist.copy()
            P0_at_1500 = P_hist[:, int(1500 / dt)].copy()
            P_supply_at_1500 = P0_at_1500[-1]

    plt.axvline(x=1500, color="r", linestyle="--", alpha=0.5, label="Demand change")
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [MPa]")
    plt.title("Case 1: Forward Flow with Demand Change")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if P_hist_case1_final is not None:
        print(f"Outlet pressure at t=1500s: {P0_at_1500[-1] / 1e6:.4f} MPa")
        print(
            f"Final outlet pressure (t=3000s): {P_hist_case1_final[-1, -1] / 1e6:.4f} MPa"
        )

    return P0_at_1500, P_supply_at_1500


def case_reverse_flow():
    print("\nCase 2: Reverse Flow (Demand at Inlet)")
    diameter = 0.6
    friction = 0.01
    length_total = 50e3
    n_segments = 10
    t_end = 3000.0
    supply_p = 50e5

    net = build_segmented_network(
        n_segments=n_segments,
        length_total=length_total,
        diameter=diameter,
        friction=friction,
        supply_at_outlet=True,
    )

    plt.figure(figsize=(12, 5))
    for dt in [1, 10, 30]:
        n_steps = int(t_end / dt)
        demand_profile = np.ones(n_steps) * 20.0

        time_sim, P_hist = simulate_transient(
            net,
            dt=dt,
            t_end=t_end,
            Z=1.0,
            M=0.016,
            P0=np.ones(n_segments + 1) * 50e5,
            supply_pressures=supply_p,
            demands={1: demand_profile},
        )
        plt.plot(time_sim, P_hist[0] / 1e6, label=f"dt = {dt} s")

    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [MPa]")
    plt.title("Case 2: Reverse Flow (Demand at Inlet, Supply at Outlet)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def case_injection(P0_at_1500, supply_at_1500):
    print("\nCase 3: Injection (Negative Demand) at Inlet")
    diameter = 0.6
    friction = 0.01
    length_total = 50e3
    n_segments = 10
    t_end = 3000.0

    net = build_segmented_network(
        n_segments=n_segments,
        length_total=length_total,
        diameter=diameter,
        friction=friction,
        supply_at_outlet=True,
    )

    plt.figure(figsize=(12, 5))
    for dt in [1, 10, 30]:
        n_steps = int(t_end / dt)
        demand_profile = np.ones(n_steps) * (-20.0)

        time_sim, P_hist = simulate_transient(
            net,
            dt=dt,
            t_end=t_end,
            Z=1.0,
            M=0.016,
            P0=np.ones(n_segments + 1) * supply_at_1500,
            supply_pressures=supply_at_1500,
            demands={1: demand_profile},
        )
        plt.plot(time_sim, P_hist[0] / 1e6, label=f"dt = {dt} s")

    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [MPa]")
    plt.title("Case 3: Injection (Negative Demand) at Inlet")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def build_triangle_network(reverse_pipe_13=False):
    nodes = {
        1: DynamicNode(node_index=1, pressure_pa=50e5, node_type="reference", altitude=100.0),
        2: DynamicNode(node_index=2, pressure_pa=47e5, node_type="demand", altitude=40.0),
        3: DynamicNode(node_index=3, pressure_pa=47e5, node_type="demand", altitude=50.0),
    }

    L_12 = 200e3
    L_23 = 150e3
    L_13 = 180e3
    diameter = 0.5
    friction = 0.01

    pipes = {
        1: DynamicPipeline(
            pipeline_index=1,
            inlet=nodes[1],
            outlet=nodes[2],
            diameter=diameter,
            length=L_12,
            friction_factor_method="constant",
            constant_friction_factor=friction,
        ),
        2: DynamicPipeline(
            pipeline_index=2,
            inlet=nodes[2],
            outlet=nodes[3],
            diameter=diameter,
            length=L_23,
            friction_factor_method="constant",
            constant_friction_factor=friction,
        ),
    }

    if reverse_pipe_13:
        inlet, outlet = nodes[3], nodes[1]
    else:
        inlet, outlet = nodes[1], nodes[3]

    pipes[3] = DynamicPipeline(
        pipeline_index=3,
        inlet=inlet,
        outlet=outlet,
        diameter=diameter,
        length=L_13,
        friction_factor_method="constant",
        constant_friction_factor=friction,
    )

    return DynamicNetwork(nodes=nodes, pipelines=pipes)


def case_triangle_orientation():
    print("\nTriangle Network (Orientation Independence):")
    dt = 10.0
    t_end = 180000.0
    n_steps = int(t_end / dt)

    P0 = np.array([50e5, 47e5, 47e5])
    supply_p = 50e5

    sm3_to_kg = 0.6816
    demand_node2 = np.ones(n_steps) * 20.0 * sm3_to_kg
    demand_node3 = np.ones(n_steps) * 30.0 * sm3_to_kg

    results = {}
    for name, reverse in [("Network 1", False), ("Network 2", True)]:
        net = build_triangle_network(reverse_pipe_13=reverse)
        time_tri, P_hist = simulate_transient(
            net,
            dt=dt,
            t_end=t_end,
            Z=1.0,
            M=0.016,
            P0=P0,
            supply_pressures=supply_p,
            demands={2: demand_node2, 3: demand_node3},
        )
        results[name] = (time_tri, P_hist)

    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, style in [("Network 1", "-"), ("Network 2", "--")]:
        time_data, P_data = results[name]
        ax2.plot(time_data, P_data[1] / 1e6, style, lw=2, label=f"{name}")
        ax3.plot(time_data, P_data[2] / 1e6, style, lw=2, label=f"{name}")

    ax2.set_ylabel("Pressure [MPa]")
    ax2.set_title("Pressure at Node 2")
    ax2.legend()
    ax2.grid(True)

    ax3.set_ylabel("Pressure [MPa]")
    ax3.set_xlabel("Time [s]")
    ax3.set_title("Pressure at Node 3")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print("Steady-state pressure comparison:")
    print("-" * 50)
    print(f"{'Node':<10} {'Network 1 [MPa]':<18} {'Network 2 [MPa]':<18} {'Diff [Pa]':<12}")
    print("-" * 50)
    for node_idx, node_name in enumerate(["Node 1", "Node 2", "Node 3"]):
        P1 = results["Network 1"][1][node_idx, -1]
        P2 = results["Network 2"][1][node_idx, -1]
        diff = abs(P1 - P2)
        print(f"{node_name:<10} {P1/1e6:<18.6f} {P2/1e6:<18.6f} {diff:<12.2f}")


if __name__ == "__main__":
    case_single_segment()
    P0_at_1500, supply_at_1500 = case_forward_flow()
    case_reverse_flow()
    if P0_at_1500 is not None and supply_at_1500 is not None:
        case_injection(P0_at_1500, supply_at_1500)
    case_triangle_orientation()
