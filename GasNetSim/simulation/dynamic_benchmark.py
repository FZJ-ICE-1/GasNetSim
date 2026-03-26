"""
Reusable transient benchmark runner for dynamic backends.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..components.utils.create_dynamic_network import (
    build_constant_demands_from_csv_volumetric,
    build_dynamic_network_from_folder,
    get_static_initial_pressures_from_folder,
)
from .dynamic import simulate_transient, simulate_transient_adaptive

try:
    from .dynamic_sundials import simulate_transient_sundials
except ImportError:
    simulate_transient_sundials = None


@dataclass
class BackendResult:
    name: str
    ok: bool
    detail: str
    runtime_base_s: float
    runtime_pert_s: float
    time_s: Optional[np.ndarray]
    p_base: Optional[np.ndarray]
    p_pert: Optional[np.ndarray]
    accepted_steps: Optional[int] = None
    rejected_steps: Optional[int] = None


@dataclass
class DynamicBenchmarkConfig:
    case_name: str
    data_dir: Path
    static_init_data_dir: Optional[Path] = None
    events: Optional[Sequence[Dict]] = None
    demands_base: Optional[Dict[int, np.ndarray]] = None
    demands_pert: Optional[Dict[int, np.ndarray]] = None
    demand_dt_s: Optional[float] = None
    force_t_end_s: Optional[float] = None
    segment_pipes: bool = True
    segment_length_m: Optional[float] = 20_000.0
    segments_per_pipe: Optional[int] = None
    length_overrides: Optional[Dict[int, float]] = None
    # Keep steady-state P0 on original (unsegmented) topology by default
    static_init_segment_pipes: bool = False
    dt_out_s: float = 60.0
    min_t_end_s: float = 21600.0
    post_event_settle_s: float = 21600.0
    z_default: float = 0.88
    m_default: float = 0.0168
    sundials_rtol: float = 1e-6
    sundials_atol: float = 1e-8
    ref_node_ids: Optional[Sequence[int]] = None
    plot_event_nodes: bool = True
    plot_node_ids: Optional[Sequence[int]] = None
    plot_backend_name: str = "SUNDIALS"
    plot_absolute_nodes: bool = False
    plot_absolute_node_ids: Optional[Sequence[int]] = None
    plot_absolute_backend_name: str = "Fixed-Semi"
    plot_absolute_use_perturbed: bool = True
    plot_absolute_gauge: bool = False
    time_zoom_s: Optional[Tuple[float, float]] = None
    run_fixed_exp: bool = True
    run_fixed_semi: bool = True
    run_sundials: bool = True
    run_adaptive_exp: bool = True
    adaptive_dt_min_s: Optional[float] = None
    adaptive_dt_max_s: Optional[float] = None
    adaptive_max_steps: int = 200000
    adaptive_rtol: float = 1e-4
    adaptive_atol: float = 1e3
    adaptive_safety: float = 0.9
    adaptive_min_factor: float = 0.2
    adaptive_max_factor: float = 2.0


def _single_line_error(exc: Exception) -> str:
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if first_line:
        return f"{type(exc).__name__}: {first_line}"
    return type(exc).__name__


def derive_horizon(
    events: Sequence[Dict],
    dt_out_s: float,
    min_t_end_s: float,
    post_event_settle_s: float,
) -> Tuple[float, int]:
    # Ensure horizon covers all events and a configurable post-event window.
    last_event = max((float(e["time"]) for e in events if "time" in e), default=0.0)
    t_end = max(float(min_t_end_s), last_event + float(post_event_settle_s))
    n_steps = int(np.ceil(t_end / dt_out_s))
    t_end = n_steps * dt_out_s
    return float(t_end), int(n_steps)


def load_profile_demands_from_csv(
    case_dir: Path,
    max_steps: Optional[int] = None,
    start_step: int = 0,
    value_scale: float = 1.0,
    profile_filename: str = "profiles.csv",
    delimiter: str = ";",
) -> Tuple[Dict[int, np.ndarray], float, List[int], int]:
    profile_path = Path(case_dir) / profile_filename
    if not profile_path.exists():
        raise FileNotFoundError(f"{profile_filename} not found in {case_dir}")

    with profile_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Empty profile header in {profile_path}")

        time_col = "time" if "time" in reader.fieldnames else reader.fieldnames[0]
        node_cols: List[Tuple[int, str]] = []
        for col in reader.fieldnames:
            if col == time_col:
                continue
            try:
                node_cols.append((int(col), col))
            except ValueError:
                continue

        if not node_cols:
            raise ValueError(f"No integer node-id columns found in {profile_path}")

        series: Dict[int, List[float]] = {nid: [] for nid, _ in node_cols}
        times: List[datetime] = []

        for row in reader:
            t_raw = (row.get(time_col) or "").strip()
            if t_raw:
                try:
                    times.append(datetime.fromisoformat(t_raw))
                except ValueError:
                    pass

            for nid, col in node_cols:
                raw = row.get(col)
                value = (float(raw) if raw not in (None, "") else 0.0) * float(value_scale)
                series[nid].append(value)

    n_rows_total = len(next(iter(series.values())))
    if n_rows_total < 2:
        raise ValueError(f"{profile_filename} in {case_dir} must contain at least 2 rows.")

    start_step = int(start_step)
    if start_step < 0:
        raise ValueError("start_step must be >= 0.")
    if start_step >= n_rows_total:
        raise ValueError(
            f"start_step={start_step} is outside available profile rows ({n_rows_total})."
        )

    if max_steps is None:
        end_step = n_rows_total
    else:
        max_steps = int(max_steps)
        if max_steps < 2:
            raise ValueError("max_steps must be >= 2 when provided.")
        end_step = min(n_rows_total, start_step + max_steps)

    if end_step - start_step < 2:
        raise ValueError(
            f"Selected profile window [{start_step}:{end_step}] has fewer than 2 rows."
        )

    for nid in list(series.keys()):
        series[nid] = series[nid][start_step:end_step]
    n_rows = end_step - start_step

    # Infer profile step from positive intra-day jumps, fallback to 1h.
    demand_dt_s = 3600.0
    times_window = times[start_step:end_step] if times else []
    if len(times_window) >= 2:
        deltas = []
        for i in range(len(times_window) - 1):
            dt = (times_window[i + 1] - times_window[i]).total_seconds()
            if 0 < dt <= 12 * 3600:
                deltas.append(dt)
        if deltas:
            demand_dt_s = float(min(deltas))

    demands = {nid: np.asarray(vals, dtype=float) for nid, vals in series.items()}
    changed_nodes = [
        nid
        for nid, arr in demands.items()
        if arr.size > 1 and np.any(np.abs(arr - arr[0]) > 1e-12)
    ]
    if not changed_nodes:
        changed_nodes = sorted(demands.keys())

    return demands, demand_dt_s, sorted(changed_nodes), n_rows


def demand_node_ids_from_nodes_csv(
    case_dir: Path,
    nodes_filename: str = "nodes.csv",
    delimiter: str = ";",
) -> List[int]:
    node_path = Path(case_dir) / nodes_filename
    if not node_path.exists():
        return []

    demand_node_ids: List[int] = []
    with node_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            node_type = str(row.get("node_type") or "").strip().lower()
            if node_type in {"reference", "supply", "slack"}:
                continue
            try:
                demand_node_ids.append(int(row["node_index"]))
            except Exception:
                continue
    return sorted(set(demand_node_ids))


def prepare_profile_demands(
    case_dir: Path,
    max_steps: Optional[int] = None,
    start_step: int = 0,
    value_scale: float = 1.0,
    profile_filename: str = "profiles.csv",
    nodes_filename: str = "nodes.csv",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], float, List[int], int]:
    demands_pert, demand_dt_s, perturbed_node_ids, n_profile_steps = load_profile_demands_from_csv(
        case_dir=case_dir,
        max_steps=max_steps,
        start_step=start_step,
        value_scale=value_scale,
        profile_filename=profile_filename,
    )

    valid_demand_nodes = set(
        demand_node_ids_from_nodes_csv(case_dir=case_dir, nodes_filename=nodes_filename)
    )
    if valid_demand_nodes:
        demands_pert = {
            nid: arr for nid, arr in demands_pert.items() if nid in valid_demand_nodes
        }
        perturbed_node_ids = [
            nid for nid in perturbed_node_ids if nid in valid_demand_nodes
        ]

    if not demands_pert:
        raise ValueError(f"no demand-node columns found in {profile_filename}")

    demands_base = {
        nid: np.full(n_profile_steps, float(arr[0]), dtype=float)
        for nid, arr in demands_pert.items()
    }
    return demands_base, demands_pert, demand_dt_s, perturbed_node_ids, n_profile_steps


def _build_network(
    data_dir: Path,
    segment_pipes: bool,
    segment_length_m: Optional[float],
    segments_per_pipe: Optional[int],
    length_overrides: Optional[Dict[int, float]],
):
    return build_dynamic_network_from_folder(
        data_dir,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=None,
    )


def _static_init(
    data_dir: Path,
    n_phys_nodes: int,
    target_network,
    segment_pipes: bool,
    segment_length_m: Optional[float],
    segments_per_pipe: Optional[int],
    length_overrides: Optional[Dict[int, float]],
    z_default: float,
    m_default: float,
) -> Tuple[Optional[np.ndarray], float, float]:
    p0, z_eff, m_eff, p0_by_node = get_static_initial_pressures_from_folder(
        data_dir,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=None,
        default_Z=z_default,
        default_M=m_default,
        return_node_pressures=True,
    )
    if p0 is not None and len(p0) != n_phys_nodes:
        # Static init on original topology, dynamic run on segmented topology
        # Map steady pressures by node ID, and use current dynamic-node pressures only for
        # inserted segmentation nodes that have no steady pressure
        if target_network is not None and p0_by_node:
            p0_mapped = np.zeros(n_phys_nodes, dtype=float)
            n_from_static = 0
            n_from_fallback = 0
            for sim_idx in range(n_phys_nodes):
                node_id = target_network.simulation_node_index_to_node_id(sim_idx)
                p_val = p0_by_node.get(int(node_id))
                if p_val is not None and np.isfinite(p_val):
                    p0_mapped[sim_idx] = float(p_val)
                    n_from_static += 1
                    continue

                node = target_network.nodes.get(node_id)
                p_seed = getattr(node, "pressure", np.nan)
                p0_mapped[sim_idx] = float(p_seed) if np.isfinite(p_seed) else np.nan
                n_from_fallback += 1

            print(
                "Static init length mismatch; mapped steady P0 by node ID "
                f"(steady={n_from_static}, inserted-fallback={n_from_fallback})."
            )
            p0 = p0_mapped
        else:
            print("Static init length mismatch; falling back to supply-based P0.")
            p0 = None

    z_val = float(z_eff) if z_eff is not None else float(z_default)
    m_val = float(m_eff) if m_eff is not None else float(m_default)
    return p0, z_val, m_val


def _build_demands(network, n_steps: int, m_val: float) -> Dict[int, np.ndarray]:
    demands, rho_std, total_q_std, total_m = build_constant_demands_from_csv_volumetric(
        network,
        n_steps=n_steps,
        M=m_val,
        T_std_K=288.15,
        P_std_Pa=101325.0,
        Z_std=1.0,
    )
    print(
        f"Demand basis: rho_std={rho_std:.6f} kg/sm3, "
        f"total={total_q_std:.3f} sm3/s -> {total_m:.3f} kg/s"
    )
    return demands


def _resample_demands_to_steps(
    demands: Dict[int, np.ndarray],
    n_steps: int,
    dt_out_s: float,
    demand_dt_s: Optional[float],
) -> Dict[int, np.ndarray]:
    # Fixed-step solver requires per-step arrays of length n_steps
    resampled: Dict[int, np.ndarray] = {}
    for node_id, profile in demands.items():
        arr = np.asarray(profile, dtype=float).reshape(-1)
        if arr.size == 0:
            resampled[int(node_id)] = np.zeros(n_steps, dtype=float)
            continue
        if arr.size == n_steps:
            resampled[int(node_id)] = arr.copy()
            continue
        if demand_dt_s is None or demand_dt_s <= 0:
            raise ValueError(
                "demand_dt_s must be positive when resampling profile demands for fixed-step runs."
            )
        t_grid = np.arange(n_steps, dtype=float) * float(dt_out_s)
        idx = np.floor(t_grid / float(demand_dt_s)).astype(int)
        idx = np.clip(idx, 0, arr.size - 1)
        resampled[int(node_id)] = arr[idx]
    return resampled


def _run_fixed(
    name: str,
    friction_treatment: str,
    data_dir: Path,
    segment_pipes: bool,
    segment_length_m: Optional[float],
    segments_per_pipe: Optional[int],
    length_overrides: Optional[Dict[int, float]],
    dt_s: float,
    t_end_s: float,
    z_val: float,
    m_val: float,
    p0: Optional[np.ndarray],
    demands_base: Dict[int, np.ndarray],
    demands_pert: Dict[int, np.ndarray],
    events_base: Sequence[Dict],
    events_pert: Sequence[Dict],
) -> BackendResult:
    net_base = _build_network(
        data_dir, segment_pipes, segment_length_m, segments_per_pipe, length_overrides
    )
    net_pert = _build_network(
        data_dir, segment_pipes, segment_length_m, segments_per_pipe, length_overrides
    )

    t0 = perf_counter()
    _, p_base = simulate_transient(
        net_base,
        dt=dt_s,
        t_end=t_end_s,
        Z=z_val,
        M=m_val,
        P0=p0,
        demands=demands_base,
        events=list(events_base),
        friction_treatment=friction_treatment,
    )
    runtime_base = perf_counter() - t0

    t1 = perf_counter()
    t_out, p_pert = simulate_transient(
        net_pert,
        dt=dt_s,
        t_end=t_end_s,
        Z=z_val,
        M=m_val,
        P0=p0,
        demands=demands_pert,
        events=list(events_pert),
        friction_treatment=friction_treatment,
    )
    runtime_pert = perf_counter() - t1

    finite_ok = bool(np.all(np.isfinite(p_base)) and np.all(np.isfinite(p_pert)))
    return BackendResult(
        name=name,
        ok=finite_ok,
        detail=f"friction={friction_treatment}",
        runtime_base_s=runtime_base,
        runtime_pert_s=runtime_pert,
        time_s=t_out,
        p_base=p_base,
        p_pert=p_pert,
    )


def _run_sundials(
    data_dir: Path,
    segment_pipes: bool,
    segment_length_m: Optional[float],
    segments_per_pipe: Optional[int],
    length_overrides: Optional[Dict[int, float]],
    t_end_s: float,
    output_times: np.ndarray,
    z_val: float,
    m_val: float,
    p0: Optional[np.ndarray],
    demands_base: Dict[int, np.ndarray],
    demands_pert: Dict[int, np.ndarray],
    demand_dt_s: float,
    events_base: Sequence[Dict],
    events_pert: Sequence[Dict],
    rtol: float,
    atol: float,
) -> BackendResult:
    if simulate_transient_sundials is None:
        return BackendResult(
            name="SUNDIALS",
            ok=False,
            detail="scikit-sundae not installed",
            runtime_base_s=0.0,
            runtime_pert_s=0.0,
            time_s=None,
            p_base=None,
            p_pert=None,
        )

    net_base = _build_network(
        data_dir, segment_pipes, segment_length_m, segments_per_pipe, length_overrides
    )
    net_pert = _build_network(
        data_dir, segment_pipes, segment_length_m, segments_per_pipe, length_overrides
    )

    t0 = perf_counter()
    _, p_base = simulate_transient_sundials(
        net_base,
        t_end=t_end_s,
        Z=z_val,
        M=m_val,
        P0=p0,
        demands=demands_base,
        demand_dt=demand_dt_s,
        events=list(events_base),
        output_times=output_times,
        rtol=rtol,
        atol=atol,
    )
    runtime_base = perf_counter() - t0

    t1 = perf_counter()
    t_out, p_pert = simulate_transient_sundials(
        net_pert,
        t_end=t_end_s,
        Z=z_val,
        M=m_val,
        P0=p0,
        demands=demands_pert,
        demand_dt=demand_dt_s,
        events=list(events_pert),
        output_times=output_times,
        rtol=rtol,
        atol=atol,
    )
    runtime_pert = perf_counter() - t1

    finite_ok = bool(np.all(np.isfinite(p_base)) and np.all(np.isfinite(p_pert)))
    return BackendResult(
        name="SUNDIALS",
        ok=finite_ok,
        detail=f"rtol={rtol:.1e}, atol={atol:.1e}",
        runtime_base_s=runtime_base,
        runtime_pert_s=runtime_pert,
        time_s=t_out,
        p_base=p_base,
        p_pert=p_pert,
    )


def _run_adaptive_explicit(
    data_dir: Path,
    segment_pipes: bool,
    segment_length_m: Optional[float],
    segments_per_pipe: Optional[int],
    length_overrides: Optional[Dict[int, float]],
    dt_init_s: float,
    t_end_s: float,
    output_times: np.ndarray,
    z_val: float,
    m_val: float,
    p0: Optional[np.ndarray],
    demands_base: Dict[int, np.ndarray],
    demands_pert: Dict[int, np.ndarray],
    demand_dt_s: float,
    events_base: Sequence[Dict],
    events_pert: Sequence[Dict],
    dt_min_s: Optional[float],
    dt_max_s: Optional[float],
    max_steps: int,
    rtol: float,
    atol: float,
    safety: float,
    min_factor: float,
    max_factor: float,
) -> BackendResult:
    net_base = _build_network(
        data_dir, segment_pipes, segment_length_m, segments_per_pipe, length_overrides
    )
    net_pert = _build_network(
        data_dir, segment_pipes, segment_length_m, segments_per_pipe, length_overrides
    )

    t0 = perf_counter()
    _, p_base, d_base = simulate_transient_adaptive(
        net_base,
        dt_init=dt_init_s,
        t_end=t_end_s,
        Z=z_val,
        M=m_val,
        P0=p0,
        demands=demands_base,
        demand_dt=demand_dt_s,
        events=list(events_base),
        friction_treatment="explicit",
        output_times=output_times,
        dt_min=dt_min_s,
        dt_max=dt_max_s,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        safety=safety,
        min_factor=min_factor,
        max_factor=max_factor,
    )
    runtime_base = perf_counter() - t0

    t1 = perf_counter()
    t_out, p_pert, d_pert = simulate_transient_adaptive(
        net_pert,
        dt_init=dt_init_s,
        t_end=t_end_s,
        Z=z_val,
        M=m_val,
        P0=p0,
        demands=demands_pert,
        demand_dt=demand_dt_s,
        events=list(events_pert),
        friction_treatment="explicit",
        output_times=output_times,
        dt_min=dt_min_s,
        dt_max=dt_max_s,
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        safety=safety,
        min_factor=min_factor,
        max_factor=max_factor,
    )
    runtime_pert = perf_counter() - t1

    finite_ok = bool(np.all(np.isfinite(p_base)) and np.all(np.isfinite(p_pert)))
    detail = (
        f"acc(base/pert)={d_base['accepted_steps']}/{d_pert['accepted_steps']}, "
        f"rej(base/pert)={d_base['rejected_steps']}/{d_pert['rejected_steps']}"
    )
    return BackendResult(
        name="Adaptive-Exp",
        ok=finite_ok,
        detail=detail,
        runtime_base_s=runtime_base,
        runtime_pert_s=runtime_pert,
        time_s=t_out,
        p_base=p_base,
        p_pert=p_pert,
        accepted_steps=int(d_base["accepted_steps"] + d_pert["accepted_steps"]),
        rejected_steps=int(d_base["rejected_steps"] + d_pert["rejected_steps"]),
    )


def _bar_gauge(p_pa: np.ndarray) -> np.ndarray:
    return p_pa / 1e5 - 1.01325


def _metrics_vs_static(
    p_static_bar: np.ndarray,
    p_base_final_bar: np.ndarray,
) -> Tuple[float, float]:
    diff = p_base_final_bar - p_static_bar
    return float(np.max(np.abs(diff))), float(np.mean(np.abs(diff)))


def _metrics_delta_response(
    delta_bar_hist: np.ndarray,
) -> Tuple[float, float, float]:
    peak_abs = float(np.max(np.abs(delta_bar_hist)))
    final_abs_mean = float(np.mean(np.abs(delta_bar_hist[:, -1])))
    final_abs_max = float(np.max(np.abs(delta_bar_hist[:, -1])))
    return peak_abs, final_abs_mean, final_abs_max


def _print_summary_table(
    rows: Sequence[Tuple],
    reference_label: str,
):
    print(f"\nBackend Comparison (Reference = {reference_label})")
    print("-" * 158)
    print(
        f"{'Backend':<16} {'Status':<8} {'Tbase[s]':>9} {'Tpert[s]':>9} "
        f"{'Base-vs-Static MaxAbs[bar]':>26} {'Base-vs-Static MeanAbs[bar]':>27} "
        f"{'Peak|Ppert-Pbase|[bar]':>22} "
        f"{'FinalMean|Ppert-Pbase|[bar]':>27} "
        f"{'FinalMax|Ppert-Pbase|[bar]':>26}"
    )
    print("-" * 158)
    for row in rows:
        (
            name,
            status,
            tb,
            tp,
            emax,
            emean,
            dpeak,
            dmean,
            dmax,
        ) = row
        print(
            f"{name:<16} {status:<8} {tb:9.3f} {tp:9.3f} "
            f"{emax:26.4f} {emean:27.4f} {dpeak:14.4f} {dmean:19.4f} {dmax:18.4f}"
        )
    print("-" * 158)


def _plot_delta_event_nodes(
    results: Sequence[BackendResult],
    network,
    events: Sequence[Dict],
    p_static_pa: Optional[np.ndarray],
    backend_name: str,
    case_label: Optional[str] = None,
    time_zoom_s: Optional[Tuple[float, float]] = None,
    use_baseline_reference: bool = False,
    node_ids: Optional[Sequence[int]] = None,
):
    if node_ids is not None and len(node_ids) > 0:
        event_node_ids = sorted({int(nid) for nid in node_ids})
    else:
        event_node_ids = sorted({int(e["node"]) for e in events if "node" in e})

    if not event_node_ids:
        return

    selected = None
    for r in results:
        if (
            r.name == backend_name
            and r.ok
            and r.time_s is not None
            and r.p_base is not None
            and r.p_pert is not None
        ):
            selected = r
            break

    if selected is None:
        for r in results:
            if r.ok and r.time_s is not None and r.p_base is not None and r.p_pert is not None:
                selected = r
                break

    if selected is None:
        print("No successful backend available for event-node plotting.")
        return

    event_times = [float(e["time"]) for e in events if "time" in e]
    if use_baseline_reference:
        delta_bar = _bar_gauge(selected.p_pert) - _bar_gauge(selected.p_base)
        ylabel = "dP = Perturbed - Baseline [bar gauge]"
        title = f"Perturbed Nodes vs Baseline ({selected.name})"
    else:
        if p_static_pa is None:
            print("No steady-state reference available for event-node plotting.")
            return
        p_static_bar_full = _bar_gauge(np.asarray(p_static_pa))
        delta_bar = _bar_gauge(selected.p_pert) - p_static_bar_full[:, None]
        ylabel = "dP = Perturbed - Steady-State [bar gauge]"
        title = f"Perturbed Nodes vs Steady-State ({selected.name})"
    if case_label:
        title = f"{title} - {case_label}"

    plt.figure(figsize=(11, 5))
    for node_id in event_node_ids:
        try:
            sim_idx = network.node_id_to_simulation_node_index(node_id)
        except Exception:
            continue
        plt.plot(selected.time_s / 3600.0, delta_bar[sim_idx, :], label=f"Node {node_id}")

    for te in event_times:
        plt.axvline(te / 3600.0, color="k", linestyle="--", alpha=0.25)

    if time_zoom_s is not None:
        plt.xlim(time_zoom_s[0] / 3600.0, time_zoom_s[1] / 3600.0)

    plt.xlabel("Time [h]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=3, fontsize="small")
    plt.tight_layout()
    plt.show()


def _plot_absolute_node_pressures(
    results: Sequence[BackendResult],
    network,
    backend_name: str,
    case_label: Optional[str] = None,
    time_zoom_s: Optional[Tuple[float, float]] = None,
    node_ids: Optional[Sequence[int]] = None,
    use_perturbed: bool = True,
    use_gauge: bool = False,
):
    if node_ids is not None and len(node_ids) > 0:
        selected_node_ids = sorted({int(nid) for nid in node_ids})
    else:
        selected_node_ids = sorted(
            int(network.simulation_node_index_to_node_id(i))
            for i in range(network.n_physical_nodes)
        )

    selected = None
    for r in results:
        if (
            r.name == backend_name
            and r.ok
            and r.time_s is not None
            and r.p_base is not None
            and r.p_pert is not None
        ):
            selected = r
            break

    if selected is None:
        for r in results:
            if r.ok and r.time_s is not None and r.p_base is not None and r.p_pert is not None:
                selected = r
                break

    if selected is None:
        print("No successful backend available for absolute-pressure plotting.")
        return

    p_hist = selected.p_pert if use_perturbed else selected.p_base
    if use_gauge:
        p_plot = _bar_gauge(p_hist)
        y_label = "Pressure [bar gauge]"
    else:
        p_plot = np.asarray(p_hist, dtype=float) / 1e5
        y_label = "Pressure [bar abs]"

    run_label = "Perturbed" if use_perturbed else "Baseline"
    plt.figure(figsize=(12, 6))
    for node_id in selected_node_ids:
        try:
            sim_idx = network.node_id_to_simulation_node_index(node_id)
        except Exception:
            continue
        plt.plot(selected.time_s / 3600.0, p_plot[sim_idx, :], linewidth=1.0, label=f"Node {node_id}")

    if time_zoom_s is not None:
        plt.xlim(time_zoom_s[0] / 3600.0, time_zoom_s[1] / 3600.0)

    plt.xlabel("Time [h]")
    plt.ylabel(y_label)
    title = f"Absolute Nodal Pressure ({run_label}, {selected.name})"
    if case_label:
        title = f"{title} - {case_label}"
    plt.title(title)
    plt.grid(True)
    if len(selected_node_ids) <= 20:
        plt.legend(ncol=3, fontsize="small")
    plt.tight_layout()
    plt.show()


def run_dynamic_benchmark(
    cfg: DynamicBenchmarkConfig,
) -> Tuple[List[BackendResult], Dict[str, object]]:
    data_dir = Path(cfg.data_dir).resolve()
    static_init_data_dir = (
        Path(cfg.static_init_data_dir).resolve()
        if cfg.static_init_data_dir is not None
        else data_dir
    )
    seed_network = _build_network(
        data_dir=data_dir,
        segment_pipes=cfg.segment_pipes,
        segment_length_m=cfg.segment_length_m,
        segments_per_pipe=cfg.segments_per_pipe,
        length_overrides=cfg.length_overrides,
    )

    events = list(cfg.events) if cfg.events is not None else []
    if cfg.force_t_end_s is not None:
        t_end_s = float(cfg.force_t_end_s)
        n_steps = int(np.ceil(t_end_s / cfg.dt_out_s))
        t_end_s = n_steps * cfg.dt_out_s
    else:
        t_end_s, n_steps = derive_horizon(
            events=events,
            dt_out_s=cfg.dt_out_s,
            min_t_end_s=cfg.min_t_end_s,
            post_event_settle_s=cfg.post_event_settle_s,
        )
    output_times = np.linspace(0.0, t_end_s, n_steps + 1)

    print(f"\nCase: {cfg.case_name}")
    print(f"Data directory: {data_dir}")
    if static_init_data_dir != data_dir:
        print(f"Static-init data directory: {static_init_data_dir}")
    print(f"Simulation horizon: t_end={t_end_s:.1f}s ({t_end_s/3600.0:.2f} h)")
    if events:
        print(f"Events: {len(events)} configured")
    else:
        print("Events: none (baseline and perturbed will match)")

    p0, z_val, m_val = _static_init(
        data_dir=static_init_data_dir,
        n_phys_nodes=seed_network.n_physical_nodes,
        target_network=seed_network,
        segment_pipes=cfg.static_init_segment_pipes,
        segment_length_m=cfg.segment_length_m if cfg.static_init_segment_pipes else None,
        segments_per_pipe=cfg.segments_per_pipe if cfg.static_init_segment_pipes else None,
        length_overrides=cfg.length_overrides,
        z_default=cfg.z_default,
        m_default=cfg.m_default,
    )
    if p0 is not None:
        p0 = np.asarray(p0, dtype=float)
        n_bad = int(np.size(p0) - np.isfinite(p0).sum())
        if n_bad > 0:
            print(
                f"Static init produced non-finite pressures ({n_bad}/{len(p0)}); "
                "falling back to supply-based P0."
            )
            p0 = None
    print(f"Using transient constants: Z={z_val:.4f}, M={m_val:.6f} kg/mol")

    using_external_demands = cfg.demands_base is not None or cfg.demands_pert is not None
    if using_external_demands:
        if cfg.demands_base is None or cfg.demands_pert is None:
            raise ValueError(
                "Provide both demands_base and demands_pert, or neither."
            )
        demand_dt_s = float(cfg.demand_dt_s) if cfg.demand_dt_s is not None else cfg.dt_out_s
        demands_base_raw = {
            int(k): np.asarray(v, dtype=float).reshape(-1)
            for k, v in cfg.demands_base.items()
        }
        demands_pert_raw = {
            int(k): np.asarray(v, dtype=float).reshape(-1)
            for k, v in cfg.demands_pert.items()
        }
        demands_base_fixed = _resample_demands_to_steps(
            demands=demands_base_raw,
            n_steps=n_steps,
            dt_out_s=cfg.dt_out_s,
            demand_dt_s=demand_dt_s,
        )
        demands_pert_fixed = _resample_demands_to_steps(
            demands=demands_pert_raw,
            n_steps=n_steps,
            dt_out_s=cfg.dt_out_s,
            demand_dt_s=demand_dt_s,
        )
        print(
            f"Demand input: external profiles (nodes={len(demands_pert_raw)}, "
            f"demand_dt={demand_dt_s:.1f}s)"
        )
    else:
        demand_dt_s = cfg.dt_out_s
        demands_constant = _build_demands(seed_network, n_steps=n_steps, m_val=m_val)
        demands_base_raw = demands_constant
        demands_pert_raw = demands_constant
        demands_base_fixed = demands_constant
        demands_pert_fixed = demands_constant

    events_base: Sequence[Dict] = []
    events_pert: Sequence[Dict] = events

    if cfg.ref_node_ids is None:
        ref_sim_idx = list(range(seed_network.n_physical_nodes))
    else:
        ref_sim_idx = [
            seed_network.node_id_to_simulation_node_index(int(nid))
            for nid in cfg.ref_node_ids
        ]

    p_static_bar = None
    if p0 is not None:
        p_static_bar = _bar_gauge(np.asarray(p0))[ref_sim_idx]
    else:
        print("No static P0 available; switching response metrics to baseline reference.")

    use_steady_reference = p_static_bar is not None
    reference_label = (
        "steady-state P0" if use_steady_reference else "baseline trajectory (events=[])"
    )

    results: List[BackendResult] = []

    if cfg.run_fixed_exp:
        try:
            results.append(
                _run_fixed(
                    name="Fixed-Exp",
                    friction_treatment="explicit",
                    data_dir=data_dir,
                    segment_pipes=cfg.segment_pipes,
                    segment_length_m=cfg.segment_length_m,
                    segments_per_pipe=cfg.segments_per_pipe,
                    length_overrides=cfg.length_overrides,
                    dt_s=cfg.dt_out_s,
                    t_end_s=t_end_s,
                    z_val=z_val,
                    m_val=m_val,
                    p0=p0,
                    demands_base=demands_base_fixed,
                    demands_pert=demands_pert_fixed,
                    events_base=events_base,
                    events_pert=events_pert,
                )
            )
        except Exception as exc:
            results.append(
                BackendResult(
                    name="Fixed-Exp",
                    ok=False,
                    detail=f"failed: {_single_line_error(exc)}",
                    runtime_base_s=0.0,
                    runtime_pert_s=0.0,
                    time_s=None,
                    p_base=None,
                    p_pert=None,
                )
            )

    if cfg.run_fixed_semi:
        try:
            results.append(
                _run_fixed(
                    name="Fixed-Semi",
                    friction_treatment="semi_implicit",
                    data_dir=data_dir,
                    segment_pipes=cfg.segment_pipes,
                    segment_length_m=cfg.segment_length_m,
                    segments_per_pipe=cfg.segments_per_pipe,
                    length_overrides=cfg.length_overrides,
                    dt_s=cfg.dt_out_s,
                    t_end_s=t_end_s,
                    z_val=z_val,
                    m_val=m_val,
                    p0=p0,
                    demands_base=demands_base_fixed,
                    demands_pert=demands_pert_fixed,
                    events_base=events_base,
                    events_pert=events_pert,
                )
            )
        except Exception as exc:
            results.append(
                BackendResult(
                    name="Fixed-Semi",
                    ok=False,
                    detail=f"failed: {_single_line_error(exc)}",
                    runtime_base_s=0.0,
                    runtime_pert_s=0.0,
                    time_s=None,
                    p_base=None,
                    p_pert=None,
                )
            )

    if cfg.run_sundials:
        try:
            results.append(
                _run_sundials(
                    data_dir=data_dir,
                    segment_pipes=cfg.segment_pipes,
                    segment_length_m=cfg.segment_length_m,
                    segments_per_pipe=cfg.segments_per_pipe,
                    length_overrides=cfg.length_overrides,
                    t_end_s=t_end_s,
                    output_times=output_times,
                    z_val=z_val,
                    m_val=m_val,
                    p0=p0,
                    demands_base=demands_base_raw,
                    demands_pert=demands_pert_raw,
                    demand_dt_s=demand_dt_s,
                    events_base=events_base,
                    events_pert=events_pert,
                    rtol=cfg.sundials_rtol,
                    atol=cfg.sundials_atol,
                )
            )
        except Exception as exc:
            results.append(
                BackendResult(
                    name="SUNDIALS",
                    ok=False,
                    detail=f"failed: {_single_line_error(exc)}",
                    runtime_base_s=0.0,
                    runtime_pert_s=0.0,
                    time_s=None,
                    p_base=None,
                    p_pert=None,
                )
            )

    if cfg.run_adaptive_exp:
        try:
            results.append(
                _run_adaptive_explicit(
                    data_dir=data_dir,
                    segment_pipes=cfg.segment_pipes,
                    segment_length_m=cfg.segment_length_m,
                    segments_per_pipe=cfg.segments_per_pipe,
                    length_overrides=cfg.length_overrides,
                    dt_init_s=cfg.dt_out_s,
                    t_end_s=t_end_s,
                    output_times=output_times,
                    z_val=z_val,
                    m_val=m_val,
                    p0=p0,
                    demands_base=demands_base_raw,
                    demands_pert=demands_pert_raw,
                    demand_dt_s=demand_dt_s,
                    events_base=events_base,
                    events_pert=events_pert,
                    dt_min_s=cfg.adaptive_dt_min_s,
                    dt_max_s=cfg.adaptive_dt_max_s,
                    max_steps=int(cfg.adaptive_max_steps),
                    rtol=float(cfg.adaptive_rtol),
                    atol=float(cfg.adaptive_atol),
                    safety=float(cfg.adaptive_safety),
                    min_factor=float(cfg.adaptive_min_factor),
                    max_factor=float(cfg.adaptive_max_factor),
                )
            )
        except Exception as exc:
            results.append(
                BackendResult(
                    name="Adaptive-Exp",
                    ok=False,
                    detail=f"failed: {_single_line_error(exc)}",
                    runtime_base_s=0.0,
                    runtime_pert_s=0.0,
                    time_s=None,
                    p_base=None,
                    p_pert=None,
                )
            )

    print("\nRun status")
    for r in results:
        status = "OK" if r.ok else "FAILED"
        print(f"{r.name:<16}: {status} - {r.detail}")

    table_rows = []
    for r in results:
        if not r.ok or r.p_base is None or r.p_pert is None:
            table_rows.append(
                (
                    r.name,
                    "FAILED",
                    r.runtime_base_s,
                    r.runtime_pert_s,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )
            continue

        p_base_bar = _bar_gauge(r.p_base)[ref_sim_idx, :]
        p_pert_bar = _bar_gauge(r.p_pert)[ref_sim_idx, :]
        p_base_final_bar = p_base_bar[:, -1]

        if use_steady_reference:
            emax, emean = _metrics_vs_static(p_static_bar, p_base_final_bar)
        else:
            emax, emean = np.nan, np.nan

        # Dynamic-response metric should be backend-internal perturbation response.
        delta_hist_bar = p_pert_bar - p_base_bar
        dpeak, dmean, dmax = _metrics_delta_response(delta_hist_bar)

        table_rows.append(
            (
                r.name,
                "OK",
                r.runtime_base_s,
                r.runtime_pert_s,
                emax,
                emean,
                dpeak,
                dmean,
                dmax,
            )
        )

    _print_summary_table(
        table_rows,
        reference_label=reference_label,
    )

    meta = {
        "events": events,
        "t_end_s": t_end_s,
        "n_steps": n_steps,
        "dt_out_s": cfg.dt_out_s,
        "p0_ref_pa": None if p0 is None else np.asarray(p0, dtype=float).copy(),
    }
    return results, meta


__all__ = [
    "BackendResult",
    "DynamicBenchmarkConfig",
    "demand_node_ids_from_nodes_csv",
    "derive_horizon",
    "load_profile_demands_from_csv",
    "prepare_profile_demands",
    "run_dynamic_benchmark",
]
