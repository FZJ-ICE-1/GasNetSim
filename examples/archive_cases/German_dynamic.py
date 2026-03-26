"""
Dynamic benchmark wrapper for archive cases:
  - extracted_subnetwork
  - northwest
  - nrw
  - south
"""

import argparse
import csv
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import R as R_UNIV

from GasNetSim.components.utils.create_dynamic_network import (
    build_dynamic_network_from_folder,
    get_static_initial_pressures_from_folder,
)
from GasNetSim.simulation.dynamic_benchmark import (
    BackendResult,
    DynamicBenchmarkConfig,
    load_profile_demands_from_csv,
    prepare_profile_demands,
    run_dynamic_benchmark,
)


def _case_dirs(base_dir: Path):
    return {
        "extracted_subnetwork": base_dir / "extracted_subnetwork",
        "northwest": base_dir / "northwest",
        "nrw": base_dir / "nrw",
        "south": base_dir / "south",
    }


def _bar_gauge(p_pa: np.ndarray) -> np.ndarray:
    return np.asarray(p_pa, dtype=float) / 1e5 - 1.01325


def _compute_mw_profile_scale(
    data_dir: Path,
    cv_kwh_per_sm3: float,
    std_temp_k: float,
    std_pressure_pa: float,
    std_z: float,
    m_fallback: float,
) -> tuple[float, float, float]:
    """
    Convert profile values in MW to mass flow [kg/s] via:
      MW -> Sm3/s -> kg/s

    Uses rho_std = P_std * M / (Z_std * R * T_std), with M estimated from
    steady-state gas properties when available.
    """
    if cv_kwh_per_sm3 <= 0:
        raise ValueError("cv_kwh_per_sm3 must be positive.")
    if std_temp_k <= 0:
        raise ValueError("std_temp_k must be positive.")
    if std_pressure_pa <= 0:
        raise ValueError("std_pressure_pa must be positive.")
    if std_z <= 0:
        raise ValueError("std_z must be positive.")
    if m_fallback <= 0:
        raise ValueError("m_fallback must be positive.")

    m_used = float(m_fallback)
    try:
        _, _, m_eff = get_static_initial_pressures_from_folder(
            path_to_folder=data_dir,
            segment_pipes=False,
            segment_length_m=None,
            segments_per_pipe=None,
            length_overrides=None,
            temperature_interpolator=None,
            default_Z=0.88,
            default_M=float(m_fallback),
            return_node_pressures=False,
        )
        if m_eff is not None and np.isfinite(m_eff) and float(m_eff) > 0:
            m_used = float(m_eff)
    except Exception:
        # Fallback to m_fallback if static property estimation is unavailable
        pass

    rho_std = float(std_pressure_pa) * m_used / (float(std_z) * R_UNIV * float(std_temp_k))
    mw_to_sm3ps = 1000.0 / (float(cv_kwh_per_sm3) * 3600.0)
    value_scale = mw_to_sm3ps * rho_std
    return float(value_scale), float(rho_std), float(m_used)


def _build_static_init_case_with_daystart_profile(
    source_case_dir: Path,
    out_case_dir: Path,
    daystart_profile_mw: Dict[int, float],
    cv_kwh_per_sm3: float,
) -> None:
    """
    Create a temporary static-init case where nodes.csv demand flows are set
    to the selected day-start profile values.
    """
    if cv_kwh_per_sm3 <= 0:
        raise ValueError("cv_kwh_per_sm3 must be positive.")

    out_case_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in source_case_dir.glob("*.csv"):
        shutil.copy2(csv_path, out_case_dir / csv_path.name)

    nodes_path = out_case_dir / "nodes.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.csv not found in {out_case_dir}")

    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Empty nodes header in {nodes_path}")
        rows = list(reader)

    mw_to_sm3ps = 1000.0 / (float(cv_kwh_per_sm3) * 3600.0)
    for row in rows:
        node_type = str(row.get("node_type") or "").strip().lower()
        if node_type in {"reference", "supply", "slack"}:
            continue
        try:
            node_id = int(row["node_index"])
        except Exception:
            continue
        if node_id not in daystart_profile_mw:
            continue
        mw_val = float(daystart_profile_mw[node_id])
        q_std = mw_val * mw_to_sm3ps
        if "flow_sm3_per_s" in row:
            row["flow_sm3_per_s"] = f"{q_std:.12g}"
        if "flow_type" in row:
            row["flow_type"] = "volumetric"

    with nodes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _find_backend_result_exact(
    results: List[BackendResult],
    backend_name: str,
) -> Optional[BackendResult]:
    for r in results:
        if (
            r.name == backend_name
            and r.ok
            and r.time_s is not None
            and r.p_base is not None
            and r.p_pert is not None
        ):
            return r
    return None


def _node_ids_to_sim_indices(network, node_ids: List[int]) -> List[int]:
    sim_indices: List[int] = []
    for nid in node_ids:
        try:
            sim_indices.append(int(network.node_id_to_simulation_node_index(int(nid))))
        except Exception:
            continue
    return sim_indices


def _plot_single_case(
    rec: Dict[str, object],
) -> None:
    backend_order = ["Fixed-Exp", "Fixed-Semi", "SUNDIALS", "Adaptive-Exp"]
    cfg: DynamicBenchmarkConfig = rec["cfg"]
    results: List[BackendResult] = rec["results"]
    meta: Dict[str, object] = rec["meta"]
    network = rec["network"]
    perturbed_node_ids: List[int] = rec["perturbed_node_ids"]
    case_label = "German Network"
    day_label = ""
    case_name_text = str(cfg.case_name)
    day_token = "(Day "
    day_start = case_name_text.find(day_token)
    if day_start >= 0:
        day_end = case_name_text.find(")", day_start)
        if day_end > day_start:
            day_label = case_name_text[day_start + 1 : day_end]
    header_label = f"{case_label} - {day_label}" if day_label else case_label

    if cfg.plot_node_ids is not None and len(cfg.plot_node_ids) > 0:
        node_ids = sorted({int(nid) for nid in cfg.plot_node_ids})
    else:
        node_ids = sorted({int(nid) for nid in perturbed_node_ids})
    sim_indices = _node_ids_to_sim_indices(network, node_ids)
    p0 = meta.get("p0_ref_pa")

    def _plot_single_case_panel(value_kind: str) -> None:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10.0, 6.6),
            squeeze=False,
            sharex=False,
            sharey=False,
        )

        for idx, backend_name in enumerate(backend_order):
            row_idx = idx // 2
            col_idx = idx % 2
            ax = axes[row_idx, col_idx]
            selected = _find_backend_result_exact(results, backend_name)

            if selected is None:
                ax.text(
                    0.5,
                    0.5,
                    "FAILED",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax.set_title(backend_name)
                if col_idx == 0:
                    ax.set_ylabel("dP [bar g]" if value_kind == "dp" else "P [bar abs]")
                if row_idx == 1:
                    ax.set_xlabel("Time [h]")
                ax.grid(True, alpha=0.5)
                continue

            if value_kind == "dp":
                if p0 is not None:
                    y_hist = _bar_gauge(selected.p_pert) - _bar_gauge(np.asarray(p0))[:, None]
                else:
                    y_hist = _bar_gauge(selected.p_pert) - _bar_gauge(selected.p_base)
                y_label = "dP [bar g]"
            else:
                p_hist = selected.p_pert if cfg.plot_absolute_use_perturbed else selected.p_base
                if cfg.plot_absolute_gauge:
                    y_hist = _bar_gauge(p_hist)
                    y_label = "P [bar g]"
                else:
                    y_hist = np.asarray(p_hist, dtype=float) / 1e5
                    y_label = "P [bar abs]"

            for sim_idx in sim_indices:
                ax.plot(selected.time_s / 3600.0, y_hist[sim_idx, :], linewidth=0.8)

            if cfg.time_zoom_s is not None:
                ax.set_xlim(cfg.time_zoom_s[0] / 3600.0, cfg.time_zoom_s[1] / 3600.0)
            ax.set_title(backend_name)
            if col_idx == 0:
                ax.set_ylabel(y_label)
            if row_idx == 1:
                ax.set_xlabel("Time [h]")
            ax.grid(True, alpha=0.6)

        if value_kind == "dp":
            panel_title = f"{header_label}: Pressure Difference w/GasNetSim Steady-State"
        else:
            panel_title = f"{header_label}: Convergence to GasNetSim's Steady-State Pressure"
        fig.suptitle(panel_title, fontsize=12)
        fig.tight_layout()
        plt.show()

    _plot_single_case_panel("dp")
    _plot_single_case_panel("abs")


def _build_config(
    case_name: str,
    data_dir: Path,
    demands_base: Dict[int, np.ndarray],
    demands_pert: Dict[int, np.ndarray],
    demand_dt_s: float,
    n_profile_steps: int,
    perturbed_node_ids: List[int],
    day_index: int = 1,
    force_t_end_s: float = None,
) -> DynamicBenchmarkConfig:
    if force_t_end_s is None:
        force_t_end_s = float((n_profile_steps - 1) * demand_dt_s)
    return DynamicBenchmarkConfig(
        case_name=f"Archive-{case_name} (Day {int(day_index)})",
        data_dir=data_dir,
        events=[],  # no explicit perturbation events; use profile-based demands
        demands_base=demands_base,
        demands_pert=demands_pert,
        demand_dt_s=float(demand_dt_s),
        force_t_end_s=float(force_t_end_s),
        segment_pipes=True,
        segment_length_m=20_000.0,
        segments_per_pipe=None,
        length_overrides=None,
        dt_out_s=60.0,
        min_t_end_s=21600.0,  # ignored when force_t_end_s is provided
        post_event_settle_s=10800.0,  # ignored when force_t_end_s is provided
        z_default=0.88,
        m_default=0.0168,
        sundials_rtol=1e-6,
        sundials_atol=1e-8,
        ref_node_ids=None,  # all physical nodes
        plot_event_nodes=False,
        plot_node_ids=perturbed_node_ids,
        plot_backend_name="SUNDIALS",
        plot_absolute_nodes=False,
        plot_absolute_node_ids=None,  # all physical nodes
        plot_absolute_backend_name="SUNDIALS",
        plot_absolute_use_perturbed=True,
        plot_absolute_gauge=False,
        time_zoom_s=None,
        run_fixed_exp=True,
        run_fixed_semi=True,
        run_sundials=True,
        run_adaptive_exp=True,
        adaptive_dt_min_s=1e-3,
        adaptive_dt_max_s=60.0,
        adaptive_max_steps=1_000_000,
        adaptive_rtol=3e-4,
        adaptive_atol=3e3,
        adaptive_safety=0.75,
        adaptive_min_factor=0.1,
        adaptive_max_factor=1.3,
    )


def main():
    parser = argparse.ArgumentParser(description="Run dynamic benchmark on archive cases.")
    parser.add_argument(
        "--case",
        choices=["extracted_subnetwork", "northwest", "nrw", "south"],
        default="extracted_subnetwork",
        help="Which case to run.",
    )
    parser.add_argument(
        "--dt-out-s",
        type=float,
        default=60.0,
        help="Output step in seconds (used by fixed backends and output grid).",
    )
    parser.add_argument(
        "--min-t-end-s",
        type=float,
        default=21600.0,
        help="Minimum simulation horizon in seconds.",
    )
    parser.add_argument(
        "--post-event-settle-s",
        type=float,
        default=10800.0,
        help="Post-event window added after last event in seconds.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable event-node plot.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Run first N profile days (24 steps/day). Default: 1.",
    )
    parser.add_argument(
        "--day-index",
        type=int,
        default=1,
        help="1-based start day index in profile data. Example: day-index=2 with days=1 runs only day 2.",
    )
    parser.add_argument(
        "--adaptive-dt-min-s",
        type=float,
        default=1e-3,
        help="Minimum adaptive step size in seconds for Adaptive-Exp.",
    )
    parser.add_argument(
        "--adaptive-dt-max-s",
        type=float,
        default=60.0,
        help="Maximum adaptive step size in seconds for Adaptive-Exp.",
    )
    parser.add_argument(
        "--adaptive-max-steps",
        type=int,
        default=1_000_000,
        help="Maximum number of adaptive internal steps for Adaptive-Exp.",
    )
    parser.add_argument(
        "--adaptive-rtol",
        type=float,
        default=3e-4,
        help="Adaptive local relative tolerance for Adaptive-Exp.",
    )
    parser.add_argument(
        "--adaptive-atol",
        type=float,
        default=3e3,
        help="Adaptive local absolute tolerance [Pa] for Adaptive-Exp.",
    )
    parser.add_argument(
        "--adaptive-safety",
        type=float,
        default=0.75,
        help="Adaptive step-size safety factor for Adaptive-Exp.",
    )
    parser.add_argument(
        "--adaptive-min-factor",
        type=float,
        default=0.1,
        help="Minimum adaptive dt scaling factor.",
    )
    parser.add_argument(
        "--adaptive-max-factor",
        type=float,
        default=1.3,
        help="Maximum adaptive dt scaling factor.",
    )
    parser.add_argument(
        "--cv-kwh-per-sm3",
        type=float,
        default=11.5,
        help="Calorific value for automatic MW->Sm3/s conversion [kWh/sm3].",
    )
    parser.add_argument(
        "--std-temp-k",
        type=float,
        default=288.15,
        help="Standard temperature for rho_std computation [K].",
    )
    parser.add_argument(
        "--std-pressure-pa",
        type=float,
        default=101325.0,
        help="Standard pressure for rho_std computation [Pa].",
    )
    parser.add_argument(
        "--std-z",
        type=float,
        default=1.0,
        help="Standard compressibility factor for rho_std computation [-].",
    )
    parser.add_argument(
        "--m-fallback",
        type=float,
        default=0.0168,
        help="Fallback molar mass [kg/mol] if steady-state M estimation is unavailable.",
    )
    parser.add_argument(
        "--p0-source",
        choices=["daystart", "nodes"],
        default="daystart",
        help="Steady-state initialization source: day-start profile demands or original nodes.csv values.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    case_dirs = _case_dirs(base_dir)

    case_name = str(args.case)
    data_dir = case_dirs[case_name]
    if not data_dir.exists():
        print(f"Skipping {case_name}: missing directory {data_dir}")
        return

    print("\n" + "=" * 80)
    try:
        if args.day_index < 1:
            raise ValueError("day-index must be >= 1.")

        max_steps = int(args.days) * 24 if args.days and args.days > 0 else None
        start_step = (int(args.day_index) - 1) * 24 if args.days and args.days > 0 else 0

        daystart_profile_mw = None
        if args.p0_source == "daystart":
            profile_raw, _profile_dt_raw, _changed_nodes_raw, _n_rows_raw = load_profile_demands_from_csv(
                case_dir=data_dir,
                max_steps=max_steps,
                start_step=start_step,
                value_scale=1.0,
            )
            daystart_profile_mw = {int(nid): float(arr[0]) for nid, arr in profile_raw.items()}

        # Archive profile values are treated as MW and converted to kg/s
        value_scale, rho_std, m_used = _compute_mw_profile_scale(
            data_dir=data_dir,
            cv_kwh_per_sm3=float(args.cv_kwh_per_sm3),
            std_temp_k=float(args.std_temp_k),
            std_pressure_pa=float(args.std_pressure_pa),
            std_z=float(args.std_z),
            m_fallback=float(args.m_fallback),
        )
        scale_detail = (
            f"mw_to_kgps_scale={value_scale:.8g} "
            f"(cv={float(args.cv_kwh_per_sm3):g} kWh/sm3, "
            f"rho_std={rho_std:.6f} kg/sm3, M={m_used:.6f} kg/mol)"
        )

        demands_base, demands_pert, demand_dt_s, perturbed_node_ids, n_profile_steps = (
            prepare_profile_demands(
                case_dir=data_dir,
                max_steps=max_steps,
                start_step=start_step,
                value_scale=float(value_scale),
            )
        )
    except Exception as exc:
        print(f"Skipping {case_name}: failed to prepare profile demands ({exc})")
        return

    cfg = _build_config(
        case_name=case_name,
        data_dir=data_dir,
        demands_base=demands_base,
        demands_pert=demands_pert,
        demand_dt_s=demand_dt_s,
        n_profile_steps=n_profile_steps,
        perturbed_node_ids=perturbed_node_ids,
        day_index=int(args.day_index),
        force_t_end_s=float(args.days * 24 * 3600) if args.days and args.days > 0 else None,
    )
    cfg.dt_out_s = float(args.dt_out_s)
    cfg.min_t_end_s = float(args.min_t_end_s)
    cfg.post_event_settle_s = float(args.post_event_settle_s)
    cfg.adaptive_dt_min_s = (
        float(args.adaptive_dt_min_s) if args.adaptive_dt_min_s is not None else None
    )
    cfg.adaptive_dt_max_s = (
        float(args.adaptive_dt_max_s) if args.adaptive_dt_max_s is not None else None
    )
    cfg.adaptive_max_steps = int(args.adaptive_max_steps)
    cfg.adaptive_rtol = float(args.adaptive_rtol)
    cfg.adaptive_atol = float(args.adaptive_atol)
    cfg.adaptive_safety = float(args.adaptive_safety)
    cfg.adaptive_min_factor = float(args.adaptive_min_factor)
    cfg.adaptive_max_factor = float(args.adaptive_max_factor)
    cfg.plot_event_nodes = False
    cfg.plot_absolute_nodes = False
    print(
        f"Profile demands loaded: nodes={len(demands_pert)}, "
        f"steps={n_profile_steps}, demand_dt={demand_dt_s:.1f}s, "
        f"perturbed_nodes={len(perturbed_node_ids)}, day_index={int(args.day_index)}, "
        f"{scale_detail}"
    )

    if args.p0_source == "daystart":
        with tempfile.TemporaryDirectory(prefix=f"static_init_{case_name}_") as td:
            static_init_case_dir = Path(td)
            _build_static_init_case_with_daystart_profile(
                source_case_dir=data_dir,
                out_case_dir=static_init_case_dir,
                daystart_profile_mw=daystart_profile_mw,
                cv_kwh_per_sm3=float(args.cv_kwh_per_sm3),
            )
            cfg.static_init_data_dir = static_init_case_dir
            print(
                f"Static-init source: day-start profile demands from {data_dir.name} "
                f"(temporary case: {static_init_case_dir})"
            )
            # daystart
            p0_dbg, _, _, _ = get_static_initial_pressures_from_folder(
            path_to_folder=static_init_case_dir,
            segment_pipes=False,
            return_node_pressures=True,
            )
            print("P0 daystart [Pa]:", p0_dbg)
            print("P0 daystart [bar(g)]:", np.asarray(p0_dbg) / 1e5 - 1.01325)
            results, meta = run_dynamic_benchmark(cfg)
    else:
        cfg.static_init_data_dir = None
        print("Static-init source: original nodes.csv demands")
        # nodes
        p0_dbg, _, _, _ = get_static_initial_pressures_from_folder(
        path_to_folder=data_dir,
        segment_pipes=False,
        return_node_pressures=True,
        )
        print("P0 nodes [Pa]:", p0_dbg)
        print("P0 nodes [bar(g)]:", np.asarray(p0_dbg) / 1e5 - 1.01325)
        results, meta = run_dynamic_benchmark(cfg)

    if not args.no_plot:
        network = build_dynamic_network_from_folder(
            cfg.data_dir,
            segment_pipes=cfg.segment_pipes,
            segment_length_m=cfg.segment_length_m,
            segments_per_pipe=cfg.segments_per_pipe,
            length_overrides=cfg.length_overrides,
            temperature_interpolator=None,
        )
        _plot_single_case(
            {
                "cfg": cfg,
                "results": results,
                "meta": meta,
                "network": network,
                "perturbed_node_ids": list(perturbed_node_ids),
            }
        )


if __name__ == "__main__":
    main()
