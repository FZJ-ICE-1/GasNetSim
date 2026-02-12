"""
Dynamic Pi-model run for Irish13 using CSV inputs.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from GasNetSim.components.utils.create_dynamic_network import (
    build_constant_demands_from_csv_volumetric,
    build_dynamic_network_from_folder,
    get_static_initial_pressures_from_folder,
)
from GasNetSim.simulation.dynamic import (
    simulate_transient,
)

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

def main():
    data_dir = Path(__file__).resolve().parent
    segment_pipes = True
    segment_length_m = 20_000.0
    segments_per_pipe = None
    length_overrides = None
    temperature_interpolator = None
    compare_to_reference = False
    compare_to_static = True

    net_base = build_dynamic_network_from_folder(
        data_dir,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )
    net_pert = build_dynamic_network_from_folder(
        data_dir,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )

    # Dynamic Pi-model constants (calibrated from steady-state by default)
    Z = 0.88
    M = 0.0168
    use_calibrated_zm_from_static = True

    # Use the same standard base as steady-state code convention (15 degC, 1 atm)
    T_std_K = 288.15
    P_std_Pa = 101325.0
    Z_std = 1.0

    dt = 25.0
    t_end = 100000.0
    n_steps = int(t_end / dt)

    use_static_init = True
    P0 = None
    if use_static_init:
        P0, z_eff, m_eff = get_static_initial_pressures_from_folder(
            data_dir,
            segment_pipes=segment_pipes,
            segment_length_m=segment_length_m,
            segments_per_pipe=segments_per_pipe,
            length_overrides=length_overrides,
            temperature_interpolator=temperature_interpolator,
            default_Z=Z,
            default_M=M,
        )
        if P0 is not None and len(P0) != net_base.n_physical_nodes:
            print(
                "Static init length mismatch; falling back to supply-based P0."
            )
            P0 = None
        elif use_calibrated_zm_from_static and z_eff is not None and m_eff is not None:
            Z = float(z_eff)
            M = float(m_eff)
            print(
                f"Calibrated transient constants from steady-state: Z={Z:.4f}, M={M:.6f} kg/mol"
            )

    print(f"Using transient constants: Z={Z:.4f}, M={M:.6f} kg/mol")

    demands, rho_std, total_q_std, total_m = build_constant_demands_from_csv_volumetric(
        net_base,
        n_steps=n_steps,
        M=M,
        T_std_K=T_std_K,
        P_std_Pa=P_std_Pa,
        Z_std=Z_std,
    )
    print(
        "Demand basis: CSV volumetric flow_sm3_per_s "
        f"-> mass via rho_std={rho_std:.6f} kg/sm3 "
        f"(T_std={T_std_K:.2f} K, P_std={P_std_Pa:.1f} Pa, Z_std={Z_std:.2f})"
    )
    print(
        f"Total demand: {total_q_std:.3f} sm3/s -> {total_m:.3f} kg/s"
    )

    # Baseline run (no perturbations)
    _, P_hist_base = simulate_transient(
        net_base,
        dt=dt,
        t_end=t_end,
        Z=Z,
        M=M,
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
        Z=Z,
        M=M,
        demands=demands,
        P0=P0,
        events=events,
    )

    print("Simulation finished.")
    ref_node_ids = list(range(1, len(IRISH13_REFERENCE_PRESSURES["novel"]) + 1))
    ref_sim_idx = [net_pert.node_id_to_simulation_node_index(nid) for nid in ref_node_ids]

    # Plot transient pressures (bar gauge) for original Irish13 nodes only
    plt.figure(figsize=(12, 6))
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
    P_ss_static = None
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
        if P_ss_static is not None:
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
