"""
Irish13 dynamic benchmark wrapper.
"""

from pathlib import Path

from GasNetSim.components.utils.create_dynamic_network import build_dynamic_network_from_folder
from GasNetSim.simulation.dynamic_benchmark import (
    DynamicBenchmarkConfig,
    _plot_absolute_node_pressures,
    _plot_delta_event_nodes,
    run_dynamic_benchmark,
)


def _events():
    return [
        {"node": 10, "time": 3600.0, "mult": 1.2},
        {"node": 6, "time": 7200.0, "add": 2.0},
        {"node": 9, "time": 10800.0, "mult": 0.85},
        {"node": 12, "time": 14400.0, "add": -1.5},
        {"node": 8, "time": 21600.0, "mult": 1.1},
    ]


def main():
    cfg = DynamicBenchmarkConfig(
        case_name="Irish13",
        data_dir=Path(__file__).resolve().parent,
        events=_events(),
        segment_pipes=True,
        segment_length_m=20_000.0,
        segments_per_pipe=None,
        length_overrides=None,
        dt_out_s=60.0,
        min_t_end_s=21600.0,
        post_event_settle_s=86400.0,
        z_default=0.88,
        m_default=0.0168,
        sundials_rtol=1e-6,
        sundials_atol=1e-8,
        ref_node_ids=list(range(1, 14)),
        plot_event_nodes=True,
        plot_backend_name="Adaptive-Exp",
        time_zoom_s=None,  # None -> full horizon
        run_fixed_exp=True,
        run_fixed_semi=True,
        run_sundials=True,
        run_adaptive_exp=True,
    )
    results, meta = run_dynamic_benchmark(cfg)

    if cfg.plot_event_nodes or cfg.plot_absolute_nodes:
        network = build_dynamic_network_from_folder(
            cfg.data_dir,
            segment_pipes=cfg.segment_pipes,
            segment_length_m=cfg.segment_length_m,
            segments_per_pipe=cfg.segments_per_pipe,
            length_overrides=cfg.length_overrides,
            temperature_interpolator=None,
        )
        t_end_s = float(meta["t_end_s"])
        zoom = cfg.time_zoom_s if cfg.time_zoom_s is not None else (0.0, t_end_s)
        p0 = meta.get("p0_ref_pa")
        use_baseline_reference = p0 is None
        events = list(cfg.events) if cfg.events is not None else []

        if cfg.plot_event_nodes:
            _plot_delta_event_nodes(
                results=results,
                network=network,
                events=events,
                p_static_pa=p0,
                backend_name=cfg.plot_backend_name,
                case_label=cfg.case_name,
                time_zoom_s=zoom,
                use_baseline_reference=use_baseline_reference,
                node_ids=cfg.plot_node_ids,
            )

        if cfg.plot_absolute_nodes:
            _plot_absolute_node_pressures(
                results=results,
                network=network,
                backend_name=cfg.plot_absolute_backend_name,
                case_label=cfg.case_name,
                time_zoom_s=zoom,
                node_ids=cfg.plot_absolute_node_ids,
                use_perturbed=cfg.plot_absolute_use_perturbed,
                use_gauge=cfg.plot_absolute_gauge,
            )


if __name__ == "__main__":
    main()
