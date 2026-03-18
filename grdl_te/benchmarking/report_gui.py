# -*- coding: utf-8 -*-
"""
NiceGUI Benchmark Dashboard — interactive web UI for benchmark results.

Renders benchmark records as a dark-themed developer dashboard with an
executive summary, sortable AG Grid step tables, per-workflow expansion
panels (time decomposition, branch analysis, memory profile), and an
optional "Save to Storage" button backed by any ``BenchmarkStore``.

Dependencies
------------
nicegui

Author
------
Claude Code (Anthropic)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-18

Modified
--------
2026-03-18
"""

# Standard library
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Third-party
from nicegui import ui

# Internal
from grdl_te.benchmarking.base import BenchmarkStore
from grdl_te.benchmarking.models import (
    BenchmarkRecord,
    StepBenchmarkResult,
    WorkflowTopology,
)
from grdl_te.benchmarking.report import _build_branch_chains
from grdl_te.benchmarking.report_md import (
    _fmt_bytes,
    _fmt_time,
    _short_name,
    _step_was_skipped,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _step_row_data(
    records: List[BenchmarkRecord],
) -> List[Dict]:
    """Flatten all step results across records into AG Grid row dicts."""
    rows: List[Dict] = []
    for record in records:
        critical_ids = set()
        if record.topology:
            critical_ids = set(record.topology.critical_path_step_ids)

        for step in record.step_results:
            if _step_was_skipped(step):
                continue
            step_key = step.step_id or f"__idx_{step.step_index}"
            rows.append({
                "workflow": record.workflow_name,
                "step_index": step.step_index,
                "step_name": _short_name(step.processor_name),
                "wall_mean": round(step.wall_time_s.mean, 6),
                "wall_median": round(step.wall_time_s.median, 6),
                "wall_stddev": round(step.wall_time_s.stddev, 6),
                "wall_p95": round(step.wall_time_s.p95, 6),
                "cpu_mean": round(step.cpu_time_s.mean, 6),
                "peak_rss": _fmt_bytes(step.peak_rss_bytes.mean),
                "peak_rss_raw": step.peak_rss_bytes.mean,
                "latency_pct": round(step.latency_pct, 1),
                "memory_pct": round(step.memory_pct, 1),
                "on_critical_path": step_key in critical_ids,
                "gpu_used": step.gpu_used,
            })
    return rows


def _record_step_rows(record: BenchmarkRecord) -> List[Dict]:
    """Build step rows for a single record's AG Grid."""
    return _step_row_data([record])


# ---------------------------------------------------------------------------
# AG Grid column definitions
# ---------------------------------------------------------------------------
_TIME_FORMATTER = "value != null ? value.toFixed(4) + 's' : '--'"
_PCT_FORMATTER = "value != null ? value.toFixed(1) + '%' : '--'"

_COMBINED_COLUMNS = [
    {
        "headerName": "Workflow", "field": "workflow",
        "sortable": True, "filter": True, "width": 160,
    },
    {
        "headerName": "#", "field": "step_index",
        "sortable": True, "width": 60,
    },
    {
        "headerName": "Step", "field": "step_name",
        "sortable": True, "filter": True, "width": 180,
    },
    {
        "headerName": "Mean Wall", "field": "wall_mean",
        "sortable": True, "width": 115,
        "valueFormatter": _TIME_FORMATTER,
    },
    {
        "headerName": "Median", "field": "wall_median",
        "sortable": True, "width": 100,
        "valueFormatter": _TIME_FORMATTER,
    },
    {
        "headerName": "StdDev", "field": "wall_stddev",
        "sortable": True, "width": 100,
        "valueFormatter": _TIME_FORMATTER,
    },
    {
        "headerName": "P95", "field": "wall_p95",
        "sortable": True, "width": 100,
        "valueFormatter": _TIME_FORMATTER,
    },
    {
        "headerName": "Mean CPU", "field": "cpu_mean",
        "sortable": True, "width": 115,
        "valueFormatter": _TIME_FORMATTER,
    },
    {
        "headerName": "Peak RSS", "field": "peak_rss",
        "sortable": True, "width": 120,
        "comparator": """function(a, b, nodeA, nodeB, isDescending) {
            return nodeA.data.peak_rss_raw - nodeB.data.peak_rss_raw;
        }""",
    },
    {
        "headerName": "Latency%", "field": "latency_pct",
        "sortable": True, "width": 100,
        "valueFormatter": _PCT_FORMATTER,
    },
    {
        "headerName": "Memory%", "field": "memory_pct",
        "sortable": True, "width": 100,
        "valueFormatter": _PCT_FORMATTER,
    },
    {
        "headerName": "Critical", "field": "on_critical_path",
        "sortable": True, "width": 90,
        "cellRenderer": """function(params) {
            return params.value ? '\\u2714' : '';
        }""",
    },
    {
        "headerName": "GPU", "field": "gpu_used",
        "sortable": True, "width": 70,
        "cellRenderer": """function(params) {
            return params.value ? '\\u2714' : '';
        }""",
    },
]

_DETAIL_COLUMNS = [c for c in _COMBINED_COLUMNS if c["field"] != "workflow"]


def _columns_for(
    records: List[BenchmarkRecord],
    base_columns: List[Dict],
) -> List[Dict]:
    """Filter column definitions based on iteration count.

    P95 is only meaningful when iterations > 10, so the column is
    removed for lower iteration counts.
    """
    if all(r.iterations > 10 for r in records):
        return base_columns
    return [c for c in base_columns if c["field"] != "wall_p95"]


# ---------------------------------------------------------------------------
# UI section builders
# ---------------------------------------------------------------------------
def _render_metric(label: str, value: str, accent: bool = False) -> None:
    """Render a single metric block (label + value)."""
    color_cls = "text-cyan-400" if accent else "text-slate-100"
    with ui.column().classes("gap-0"):
        ui.label(label).classes(
            "text-xs text-slate-400 uppercase tracking-wide"
        )
        ui.label(value).classes(f"text-2xl font-mono {color_cls}")


def _exec_summary(records: List[BenchmarkRecord]) -> None:
    """Render the executive summary card."""
    with ui.card().classes("bg-zinc-800 w-full"):
        ui.label("Executive Summary").classes(
            "text-lg font-semibold text-slate-200 mb-2"
        )
        with ui.row().classes("gap-8 flex-wrap items-end"):
            # Topology counts
            topo_counts: Dict[str, int] = {}
            for rec in records:
                name = (
                    rec.topology.topology.value if rec.topology else "unknown"
                )
                topo_counts[name] = topo_counts.get(name, 0) + 1
            topo_str = ", ".join(
                f"{c} {n}" for n, c in sorted(topo_counts.items())
            )
            _render_metric("Topology", topo_str)

            # Hardware one-liner
            hw = records[0].hardware
            mem_str = _fmt_bytes(hw.total_memory_bytes)
            gpu_part = (
                f", {len(hw.gpu_devices)} GPU(s)"
                if hw.gpu_available
                else ""
            )
            _render_metric("Hardware", f"{hw.cpu_count} CPUs, {mem_str}{gpu_part}")

            # Aggregate metrics
            if len(records) == 1:
                r = records[0]
                _render_metric("Iterations", str(r.iterations))
            else:
                import numpy as np

                walls = [r.total_wall_time.mean for r in records]
                cpus = [r.total_cpu_time.mean for r in records]
                mems = [r.total_peak_rss.mean for r in records]
                _render_metric(
                    "Avg Wall Time",
                    _fmt_time(float(np.mean(walls))),
                    accent=True,
                )
                _render_metric(
                    "Avg CPU Time", _fmt_time(float(np.mean(cpus)))
                )
                _render_metric(
                    "Avg Peak Memory", _fmt_bytes(float(np.mean(mems)))
                )


def _combined_grid(records: List[BenchmarkRecord]) -> None:
    """Render the combined step performance AG Grid."""
    rows = _step_row_data(records)
    ui.label("Step Performance").classes(
        "text-lg font-semibold text-slate-200 mt-4 mb-1"
    )
    cols = _columns_for(records, _COMBINED_COLUMNS)
    ui.aggrid({
        "columnDefs": cols,
        "rowData": rows,
        "defaultColDef": {"resizable": True},
        "domLayout": "autoHeight" if len(rows) <= 20 else "normal",
    }).classes("w-full").style(
        "height: 400px" if len(rows) > 20 else ""
    ).props(':theme-params=\'{"alpine-dark": true}\'')


def _time_decomposition(record: BenchmarkRecord) -> None:
    """Render time decomposition for parallel/mixed workflows."""
    if not record.topology:
        return
    topo = record.topology
    if topo.topology in (WorkflowTopology.COMPONENT, WorkflowTopology.SEQUENTIAL):
        return

    ui.label("Time Decomposition").classes(
        "text-sm font-semibold text-slate-300 mt-3"
    )
    rows = [
        {
            "metric": "Wall Clock (actual elapsed)",
            "value": _fmt_time(record.total_wall_time.mean),
        },
        {
            "metric": "Critical Path (longest chain)",
            "value": _fmt_time(topo.critical_path_wall_time_s),
        },
        {
            "metric": "Sum of Steps (if sequential)",
            "value": _fmt_time(topo.sum_of_steps_wall_time_s),
        },
        {
            "metric": "Parallelism Ratio",
            "value": f"{topo.parallelism_ratio:.2f}x",
        },
    ]
    columns = [
        {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
        {"name": "value", "label": "Value", "field": "value", "align": "right"},
    ]
    ui.table(columns=columns, rows=rows).props("dark dense flat bordered").classes(
        "w-full max-w-lg"
    )


def _branch_analysis(record: BenchmarkRecord) -> None:
    """Render branch analysis for multi-branch workflows."""
    if not record.topology:
        return
    topo = record.topology
    if topo.topology in (WorkflowTopology.COMPONENT, WorkflowTopology.SEQUENTIAL):
        return

    chains = _build_branch_chains(record.step_results)
    if len(chains) < 2:
        return

    chain_times = [sum(s.wall_time_s.mean for s in c) for c in chains]
    critical_time = max(chain_times) if chain_times else 0.0

    ui.label("Branch Analysis").classes(
        "text-sm font-semibold text-slate-300 mt-3"
    )
    ui.label(
        f"{len(chains)} branches, critical path: {_fmt_time(critical_time)}"
    ).classes("text-xs text-slate-400 italic mb-1")

    rows = []
    for i, (chain, ct) in enumerate(zip(chains, chain_times)):
        step_names = " \u2192 ".join(
            _short_name(s.processor_name) for s in chain
        )
        if ct >= critical_time - 1e-9:
            status = "critical path"
        else:
            idle = critical_time - ct
            status = f"idle {_fmt_time(idle)}"
        rows.append({
            "branch": i + 1,
            "steps": step_names,
            "chain_time": _fmt_time(ct),
            "status": status,
        })

    columns = [
        {"name": "branch", "label": "Branch", "field": "branch", "align": "left"},
        {"name": "steps", "label": "Steps", "field": "steps", "align": "left"},
        {"name": "chain_time", "label": "Chain Time", "field": "chain_time", "align": "right"},
        {"name": "status", "label": "Status", "field": "status", "align": "left"},
    ]
    ui.table(columns=columns, rows=rows).props("dark dense flat bordered").classes(
        "w-full"
    )


def _memory_profile(record: BenchmarkRecord) -> None:
    """Render memory profile table for steps with overhead data."""
    active = [s for s in record.step_results if not _step_was_skipped(s)]
    has_mem = any(
        s.peak_overhead_bytes is not None
        or s.end_of_step_footprint_bytes is not None
        for s in active
    )
    if not has_mem:
        return

    ui.label("Memory Profile").classes(
        "text-sm font-semibold text-slate-300 mt-3"
    )

    rows = []
    for step in active:
        overhead = (
            _fmt_bytes(step.peak_overhead_bytes.mean)
            if step.peak_overhead_bytes is not None
            else "N/A"
        )
        footprint = (
            _fmt_bytes(step.end_of_step_footprint_bytes.mean)
            if step.end_of_step_footprint_bytes is not None
            else "N/A"
        )
        mem_str = f"{step.memory_pct:.1f}%"
        if step.concurrent:
            mem_str += " (concurrent)"
        rows.append({
            "step": _short_name(step.processor_name),
            "overhead": overhead,
            "footprint": footprint,
            "memory_pct": mem_str,
        })

    rows.append({
        "step": "Overall Workflow Peak",
        "overhead": _fmt_bytes(record.total_peak_rss.mean),
        "footprint": "(high-water mark)",
        "memory_pct": "",
    })

    columns = [
        {"name": "step", "label": "Step", "field": "step", "align": "left"},
        {"name": "overhead", "label": "Peak Overhead", "field": "overhead", "align": "right"},
        {"name": "footprint", "label": "End-of-Step Footprint", "field": "footprint", "align": "right"},
        {"name": "memory_pct", "label": "Memory%", "field": "memory_pct", "align": "right"},
    ]
    ui.table(columns=columns, rows=rows).props("dark dense flat bordered").classes(
        "w-full"
    )


def _workflow_panel(record: BenchmarkRecord, index: int) -> None:
    """Render one expansion panel for a single benchmark record."""
    topo_label = ""
    if record.topology:
        topo_label = f" ({record.topology.topology.value})"

    header = f"{index}. {record.workflow_name}{topo_label}"

    with ui.expansion(header, icon="assessment").props(
        "dark dense header-class='bg-zinc-700/50'"
    ).classes("w-full"):
        # Summary line
        ui.label(
            f"Type: {record.benchmark_type}  |  "
            f"Version: {record.workflow_version}  |  "
            f"Iterations: {record.iterations}  |  "
            f"Wall: {_fmt_time(record.total_wall_time.mean)}  |  "
            f"CPU: {_fmt_time(record.total_cpu_time.mean)}  |  "
            f"Memory: {_fmt_bytes(record.total_peak_rss.mean)}"
        ).classes("text-sm text-slate-300 mb-2")

        if record.tags:
            tag_str = ", ".join(
                f"{k}={v}" for k, v in sorted(record.tags.items())
            )
            ui.label(f"Tags: {tag_str}").classes(
                "text-xs text-slate-400 mb-2"
            )

        # Per-record step grid
        step_rows = _record_step_rows(record)
        if step_rows:
            detail_cols = _columns_for([record], _DETAIL_COLUMNS)
            ui.aggrid({
                "columnDefs": detail_cols,
                "rowData": step_rows,
                "defaultColDef": {"resizable": True},
                "domLayout": "autoHeight" if len(step_rows) <= 15 else "normal",
            }).classes("w-full").style(
                "height: 300px" if len(step_rows) > 15 else ""
            ).props(':theme-params=\'{"alpine-dark": true}\'')

        # Optional sections
        _time_decomposition(record)
        _branch_analysis(record)
        _memory_profile(record)


# ---------------------------------------------------------------------------
# Save handler
# ---------------------------------------------------------------------------
async def _handle_save(
    records: List[BenchmarkRecord],
    store: BenchmarkStore,
    button: ui.button,
) -> None:
    """Persist all records via the given store."""
    button.disable()
    saved, errors = 0, []
    for record in records:
        try:
            store.save(record)
            saved += 1
        except Exception as exc:
            errors.append(f"{record.benchmark_id[:8]}: {exc}")
    if errors:
        ui.notify(
            f"Saved {saved}/{len(records)}. Errors: {'; '.join(errors)}",
            type="warning",
            position="top",
        )
    else:
        ui.notify(
            f"All {saved} record(s) saved successfully.",
            type="positive",
            position="top",
        )
    button.enable()


# ---------------------------------------------------------------------------
# Single-report view (used for both flat and tabbed modes)
# ---------------------------------------------------------------------------
def _render_report_view(
    records: List[BenchmarkRecord],
    store: Optional[BenchmarkStore],
) -> None:
    """Render a complete report view for a list of records."""
    # Executive summary
    _exec_summary(records)

    # Combined step grid
    _combined_grid(records)

    # Per-workflow expansion panels
    ui.label("Workflow Details").classes(
        "text-lg font-semibold text-slate-200 mt-4 mb-1"
    )
    sorted_records = sorted(
        records,
        key=lambda r: r.total_wall_time.mean,
        reverse=True,
    )
    for i, record in enumerate(sorted_records, start=1):
        _workflow_panel(record, i)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Type alias: a named group is (label, records_list)
ReportGroup = Tuple[str, List[BenchmarkRecord]]


def launch_ui(
    records: Union[List[BenchmarkRecord], List[ReportGroup]],
    store: Optional[BenchmarkStore] = None,
) -> None:
    """Launch an interactive NiceGUI benchmark dashboard.

    Renders the provided benchmark records as a dark-themed web
    dashboard with sortable step tables, per-workflow detail panels,
    and an optional persistence button.

    This is a **blocking** call — execution continues after the
    browser tab is closed or the server is stopped.

    Parameters
    ----------
    records : list
        Either a flat list of ``BenchmarkRecord`` objects (rendered as
        a single report), or a list of ``(label, records)`` tuples
        (rendered as a tabbed interface with one tab per group so that
        users can click through each report).
    store : BenchmarkStore, optional
        Persistence backend.  When provided a "Save to Storage"
        button is shown; otherwise it is hidden.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("Cannot launch dashboard with empty records list.")

    # Normalise input: detect grouped vs flat format.
    # Grouped format: list of (str, list) tuples.
    groups: List[ReportGroup]
    if records and isinstance(records[0], tuple):
        groups = records  # type: ignore[assignment]
    else:
        groups = [("All Records", records)]  # type: ignore[list-item]

    all_records = [r for _, recs in groups for r in recs]

    @ui.page("/")
    def dashboard() -> None:
        ui.dark_mode(True)

        with ui.column().classes(
            "w-full max-w-7xl mx-auto p-6 bg-slate-900 min-h-screen gap-4"
        ):
            # Header bar
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("GRDL Benchmark Dashboard").classes(
                    "text-2xl font-bold text-slate-100"
                )
                with ui.row().classes("gap-4 items-center"):
                    ui.badge(
                        f"{len(all_records)} record{'s' if len(all_records) != 1 else ''}",
                        color="cyan",
                    ).props("outline")

                    if store is not None:
                        save_btn = ui.button(
                            "Save to Storage",
                            icon="save",
                            on_click=lambda: _handle_save(
                                all_records, store, save_btn
                            ),
                        ).classes("bg-emerald-600 text-white")

            # Single group → render flat (no tabs)
            if len(groups) == 1:
                _render_report_view(groups[0][1], store)
            else:
                # Multiple groups → tabbed interface
                with ui.tabs().classes("w-full").props("dark dense") as tabs:
                    tab_objects = []
                    for label, _ in groups:
                        tab_objects.append(ui.tab(label))

                with ui.tab_panels(tabs).classes(
                    "w-full bg-slate-900"
                ).props("dark"):
                    for tab_obj, (label, group_records) in zip(
                        tab_objects, groups
                    ):
                        with ui.tab_panel(tab_obj):
                            _render_report_view(group_records, store)

    ui.run(reload=False, title="GRDL Benchmark Dashboard")
