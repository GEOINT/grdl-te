# -*- coding: utf-8 -*-
"""
NiceGUI Benchmark Dashboard — interactive web UI for benchmark results.

Renders benchmark records as a dark-themed developer dashboard with an
executive summary, sortable AG Grid step tables, per-workflow expansion
panels (time decomposition, branch analysis, memory profile), and an
optional "Save to Storage" button backed by any ``BenchmarkStore``.

This module is a thin presenter.  All data derivations are performed
by ``report_engine.build_report_data()``.

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
2026-03-19
"""

# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party
from nicegui import ui

# Internal
from grdl_te.benchmarking._formatting import (
    fmt_bytes as _fmt_bytes,
    fmt_throughput as _fmt_throughput,
    fmt_time as _fmt_time,
    short_name as _short_name,
)
from grdl_te.benchmarking.base import BenchmarkStore
from grdl_te.benchmarking.store import JSONBenchmarkStore
from grdl_te.benchmarking.models import (
    BenchmarkRecord,
    StepBenchmarkResult,
    WorkflowTopology,
)
from grdl_te.benchmarking.report_engine import (
    RecordReportData,
    ReportData,
    StepReportData,
    build_report_data,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _step_row_data_from_engine(data: ReportData) -> List[Dict]:
    """Flatten all step results across records into AG Grid row dicts."""
    rows: List[Dict] = []
    for rd in data.records:
        for sd in rd.steps:
            if sd.skipped:
                rows.append({
                    "workflow": rd.workflow_name,
                    "step_index": sd.step_index,
                    "step_name": sd.short_name,
                    "wall_mean": None,
                    "wall_median": None,
                    "wall_stddev": None,
                    "wall_p95": None,
                    "cpu_mean": None,
                    "latency_pct": None,
                    "on_critical_path": False,
                    "gpu_used": False,
                    "throughput": "--",
                    "throughput_raw": None,
                    "path": "skipped",
                })
                continue
            rows.append({
                "workflow": rd.workflow_name,
                "step_index": sd.step_index,
                "step_name": sd.short_name,
                "wall_mean": round(sd.wall_time.mean, 6),
                "wall_median": round(sd.wall_time.median, 6),
                "wall_stddev": round(sd.wall_time.stddev, 6),
                "wall_p95": round(sd.wall_time.p95, 6),
                "cpu_mean": round(sd.cpu_time.mean, 6),
                "latency_pct": round(sd.latency_pct, 1),
                "on_critical_path": sd.on_critical_path,
                "gpu_used": sd.gpu_used,
                "throughput": _fmt_throughput(sd.throughput_scalar),
                "throughput_raw": sd.throughput_scalar,
                "path": sd.path_classification,
            })
    return rows


def _record_step_rows(rd: RecordReportData) -> List[Dict]:
    """Build step rows for a single record's AG Grid."""
    rows: List[Dict] = []
    for sd in rd.steps:
        if sd.skipped:
            rows.append({
                "step_index": sd.step_index,
                "step_name": sd.short_name,
                "wall_mean": None,
                "wall_median": None,
                "wall_stddev": None,
                "wall_p95": None,
                "cpu_mean": None,
                "peak_rss": "--",
                "peak_rss_raw": 0,
                "latency_pct": None,
                "on_critical_path": False,
                "gpu_used": False,
                "throughput": "--",
                "throughput_raw": None,
                "path": "skipped",
            })
            continue
        rows.append({
            "step_index": sd.step_index,
            "step_name": sd.short_name,
            "wall_mean": round(sd.wall_time.mean, 6),
            "wall_median": round(sd.wall_time.median, 6),
            "wall_stddev": round(sd.wall_time.stddev, 6),
            "wall_p95": round(sd.wall_time.p95, 6),
            "cpu_mean": round(sd.cpu_time.mean, 6),
            "peak_rss": _fmt_bytes(sd.peak_rss.mean),
            "peak_rss_raw": sd.peak_rss.mean,
            "latency_pct": round(sd.latency_pct, 1),
            "on_critical_path": sd.on_critical_path,
            "gpu_used": sd.gpu_used,
            "throughput": _fmt_throughput(sd.throughput_scalar),
            "throughput_raw": sd.throughput_scalar,
            "path": sd.path_classification,
        })
    return rows


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
        "headerName": "Throughput", "field": "throughput",
        "sortable": True, "width": 120,
        "comparator": """function(a, b, nodeA, nodeB, isDescending) {
            var ra = nodeA.data.throughput_raw, rb = nodeB.data.throughput_raw;
            if (ra == null) return -1;
            if (rb == null) return 1;
            return ra - rb;
        }""",
    },
    {
        "headerName": "Latency%", "field": "latency_pct",
        "sortable": True, "width": 100,
        "valueFormatter": _PCT_FORMATTER,
    },
    {
        "headerName": "Path", "field": "path",
        "sortable": True, "filter": True, "width": 100,
        "cellRenderer": """function(params) {
            if (params.value === 'skipped') return '<em>skipped</em>';
            if (params.value === 'critical') return '<strong>critical</strong>';
            return params.value || '';
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


_STATS_FIELDS = {"wall_median", "wall_stddev", "wall_p95"}


def _columns_for(
    records: List[BenchmarkRecord],
    base_columns: List[Dict],
    has_gpu: bool = False,
) -> List[Dict]:
    """Filter column definitions based on iteration count and GPU presence."""
    all_single = all(r.iterations == 1 for r in records)

    if all_single:
        cols = []
        for c in base_columns:
            if c["field"] in _STATS_FIELDS:
                continue
            if c["field"] == "wall_mean":
                c = {**c, "headerName": "Wall Time"}
            elif c["field"] == "cpu_mean":
                c = {**c, "headerName": "CPU Time"}
            cols.append(c)
    elif all(r.iterations > 10 for r in records):
        cols = list(base_columns)
    else:
        cols = [c for c in base_columns if c["field"] != "wall_p95"]

    if not has_gpu:
        cols = [c for c in cols if c["field"] != "gpu_used"]
    return cols


def _columns_for_rd(
    rds: List[RecordReportData],
    base_columns: List[Dict],
    has_gpu: bool = False,
) -> List[Dict]:
    """Filter column definitions using RecordReportData."""
    all_single = all(rd.iterations == 1 for rd in rds)

    if all_single:
        cols = []
        for c in base_columns:
            if c["field"] in _STATS_FIELDS:
                continue
            if c["field"] == "wall_mean":
                c = {**c, "headerName": "Wall Time"}
            elif c["field"] == "cpu_mean":
                c = {**c, "headerName": "CPU Time"}
            cols.append(c)
    elif all(rd.iterations > 10 for rd in rds):
        cols = list(base_columns)
    else:
        cols = [c for c in base_columns if c["field"] != "wall_p95"]

    if not has_gpu:
        cols = [c for c in cols if c["field"] != "gpu_used"]
    return cols


# ---------------------------------------------------------------------------
# UI section builders
# ---------------------------------------------------------------------------
def _render_metric(label: str, value: str, accent: bool = False) -> None:
    """Render a single metric block (label + value)."""
    color_cls = "text-cyan-300" if accent else "text-slate-100"
    with ui.column().classes("gap-0 min-w-0"):
        ui.label(label).classes(
            "text-[11px] text-slate-500 uppercase tracking-widest font-medium"
        )
        ui.label(value).classes(
            f"text-xl font-mono {color_cls} truncate max-w-[260px]"
        ).props('title="' + value.replace('"', '&quot;') + '"')


def _exec_summary(data: ReportData) -> None:
    """Render the executive summary card."""
    with ui.card().classes(
        "w-full bg-slate-800/60 shadow-lg ring-1 ring-white/5 overflow-hidden"
    ).props("flat"):
        ui.label("Executive Summary").classes(
            "text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3"
        )
        with ui.row().classes("gap-8 flex-wrap items-end min-w-0"):
            hw = data.hardware

            _render_metric("Hostname", hw.hostname)

            if hw.captured_at:
                _render_metric("Captured", hw.captured_at[:19])

            mem_str = _fmt_bytes(hw.total_memory_bytes)
            gpu_part = (
                f", {len(hw.gpu_devices)} GPU(s)"
                if hw.gpu_available
                else ""
            )
            _render_metric("Hardware", f"{hw.cpu_count} CPUs, {mem_str}{gpu_part}")

            _render_metric("Records", str(data.record_count))

            # Per-record context
            if data.record_count == 1:
                rd = data.records[0]
                rec = rd.record
                _render_metric("Iterations", str(rd.iterations))

                array_size = rec.tags.get("array_size", "")
                if array_size:
                    _render_metric("Array Size", array_size)

                _render_metric(
                    "Wall Time",
                    _fmt_time(rec.total_wall_time.mean),
                    accent=True,
                )
                _render_metric("CPU Time", _fmt_time(rec.total_cpu_time.mean))
                _render_metric(
                    "Peak Memory", _fmt_bytes(rec.total_peak_rss.mean)
                )

        # Top bottleneck callout
        if data.bottlenecks:
            top = data.bottlenecks[0]
            with ui.row().classes(
                "w-full mt-4 px-4 py-2 rounded bg-slate-700/40"
                " items-center min-w-0 overflow-hidden"
            ):
                ui.html(
                    f'<span class="text-sm text-slate-300" style="word-break:break-word">'
                    f'<span class="text-slate-500 font-medium">Top Bottleneck</span>'
                    f'&ensp;'
                    f'<code class="text-cyan-300">{top.step_name}</code> '
                    f'&mdash; {top.latency_pct:.1f}% of latency '
                    f'({_fmt_time(top.wall_time_s)} mean wall) '
                    f'in {top.workflow}'
                    f'</span>'
                )

        # Bottleneck ranking table
        if len(data.bottlenecks) > 1:
            rows = [
                {
                    "rank": bn.rank,
                    "step": bn.step_name,
                    "latency_pct": f"{bn.latency_pct:.1f}%"
                    if bn.latency_pct > 0
                    else "--",
                    "workflow": bn.workflow,
                    "wall_time": _fmt_time(bn.wall_time_s),
                }
                for bn in data.bottlenecks
            ]
            columns = [
                {"name": "rank", "label": "Rank", "field": "rank", "align": "left"},
                {"name": "step", "label": "Step", "field": "step", "align": "left"},
                {"name": "latency_pct", "label": "Latency%", "field": "latency_pct", "align": "right"},
                {"name": "workflow", "label": "Workflow", "field": "workflow", "align": "left"},
                {"name": "wall_time", "label": "Mean Wall", "field": "wall_time", "align": "right"},
            ]
            ui.table(
                columns=columns, rows=rows,
            ).props(
                "dark dense flat bordered hide-bottom"
                " :rows-per-page-options=\"[0]\""
            ).classes("w-full mt-2").style("max-height: 280px; overflow-y: auto")


def _combined_grid(data: ReportData) -> None:
    """Render the combined step performance AG Grid."""
    rows = _step_row_data_from_engine(data)
    has_gpu = any(r.get("gpu_used") for r in rows)
    ui.label("Step Performance").classes(
        "text-sm font-semibold text-slate-400 uppercase tracking-widest mt-6 mb-2"
    )
    cols = _columns_for_rd(list(data.records), _COMBINED_COLUMNS, has_gpu=has_gpu)
    ui.aggrid({
        "columnDefs": cols,
        "rowData": rows,
        "defaultColDef": {"resizable": True},
        "domLayout": "autoHeight" if len(rows) <= 20 else "normal",
    }).classes("w-full").style(
        "height: 400px" if len(rows) > 20 else ""
    ).props(':theme-params=\'{"alpine-dark": true}\'')


def _time_decomposition(rd: RecordReportData) -> None:
    """Render time decomposition from pre-computed data."""
    td = rd.time_decomposition
    if td is None:
        return

    ui.label("Time Decomposition").classes(
        "text-xs font-semibold text-slate-500 uppercase tracking-widest mt-4"
    )
    rows = [
        {
            "metric": "Wall Clock (actual elapsed)",
            "value": _fmt_time(td.wall_clock_s),
        },
        {
            "metric": "Critical Path (longest chain)",
            "value": _fmt_time(td.critical_path_s),
        },
        {
            "metric": "Contended Step Sum \u2021",
            "value": _fmt_time(td.contended_step_sum_s),
        },
    ]
    columns = [
        {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
        {"name": "value", "label": "Value", "field": "value", "align": "right"},
    ]
    ui.table(columns=columns, rows=rows).props(
        "dark dense flat bordered hide-bottom"
        " :rows-per-page-options=\"[0]\""
    ).classes("w-full max-w-lg")


def _branch_analysis(rd: RecordReportData) -> None:
    """Render branch analysis from pre-computed data."""
    branches = rd.branches
    if len(branches) < 2:
        return

    critical_time = max(b.chain_time_s for b in branches)

    ui.label("Branch Analysis").classes(
        "text-xs font-semibold text-slate-500 uppercase tracking-widest mt-4"
    )
    ui.label(
        f"{len(branches)} branches, critical path: {_fmt_time(critical_time)}"
    ).classes("text-xs text-slate-400 italic mb-1")

    rows = []
    for b in branches:
        step_names = " \u2192 ".join(b.step_names)
        if b.is_critical:
            status = "critical path"
        else:
            status = f"idle {_fmt_time(b.idle_time_s)}"
        rows.append({
            "branch": b.branch_index,
            "steps": step_names,
            "chain_time": _fmt_time(b.chain_time_s),
            "status": status,
        })

    columns = [
        {"name": "branch", "label": "Branch", "field": "branch", "align": "left"},
        {"name": "steps", "label": "Steps", "field": "steps", "align": "left"},
        {"name": "chain_time", "label": "Chain Time", "field": "chain_time", "align": "right"},
        {"name": "status", "label": "Status", "field": "status", "align": "left"},
    ]
    ui.table(columns=columns, rows=rows).props(
        "dark dense flat bordered hide-bottom"
        " :rows-per-page-options=\"[0]\""
    ).classes("w-full")


def _workflow_panel(rd: RecordReportData, index: int) -> None:
    """Render one expansion panel for a single benchmark record."""
    topo_label = f" ({rd.topology_label})" if rd.topology_label else ""
    header = f"{index}. {rd.workflow_name}{topo_label}"
    rec = rd.record

    with ui.expansion(header).props(
        "dark dense header-class='bg-slate-800/60'"
    ).classes("w-full rounded ring-1 ring-white/5"):
        # Summary line
        ui.label(
            f"Type: {rd.benchmark_type}  |  "
            f"Version: {rd.workflow_version}  |  "
            f"Iterations: {rd.iterations}  |  "
            f"Wall: {_fmt_time(rec.total_wall_time.mean)}  |  "
            f"CPU: {_fmt_time(rec.total_cpu_time.mean)}  |  "
            f"Memory: {_fmt_bytes(rec.total_peak_rss.mean)}"
        ).classes("text-sm text-slate-300 mb-2").style("word-break: break-word")

        if rec.tags:
            tag_str = ", ".join(
                f"{k}={v}" for k, v in sorted(rec.tags.items())
            )
            ui.label(f"Tags: {tag_str}").classes(
                "text-xs text-slate-400 mb-2 break-all"
            )

        # Per-record step grid
        step_rows = _record_step_rows(rd)
        if step_rows:
            has_gpu = any(r.get("gpu_used") for r in step_rows)
            detail_cols = _columns_for([rec], _DETAIL_COLUMNS, has_gpu=has_gpu)
            ui.aggrid({
                "columnDefs": detail_cols,
                "rowData": step_rows,
                "defaultColDef": {"resizable": True},
                "domLayout": "autoHeight" if len(step_rows) <= 15 else "normal",
            }).classes("w-full").style(
                "height: 300px" if len(step_rows) > 15 else ""
            ).props(':theme-params=\'{"alpine-dark": true}\'')

        # Optional sections
        _time_decomposition(rd)
        _branch_analysis(rd)


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


def _validate_save_path(path_str: str) -> Tuple[bool, str]:
    """Validate a directory path for saving benchmark records."""
    if not path_str or not path_str.strip():
        return False, "Enter a directory path"
    try:
        p = Path(path_str.strip()).expanduser().resolve()
    except (ValueError, OSError):
        return False, "Invalid path"
    if p.is_dir():
        if os.access(p, os.W_OK):
            return True, f"Directory exists \u2014 {p}"
        return False, "Permission denied"
    # Check if parent (or nearest ancestor) is writable
    parent = p.parent
    while parent != parent.parent:
        if parent.is_dir():
            if os.access(parent, os.W_OK):
                return True, f"Directory will be created \u2014 {p}"
            return False, f"Permission denied on {parent}"
        parent = parent.parent
    return False, "Invalid path"


def _show_save_dialog(
    records: List[BenchmarkRecord],
    default_dir: Optional[str] = None,
) -> None:
    """Open a dialog for the user to choose a save directory."""
    default = default_dir or str(Path.cwd() / ".benchmarks")

    with ui.dialog().props("persistent") as dialog, \
            ui.card().classes(
                "bg-slate-800 ring-1 ring-white/5 min-w-[480px]"
            ).props("flat"):
        ui.label("Save Benchmark Records").classes(
            "text-sm font-semibold text-slate-400 uppercase tracking-widest"
        )

        path_input = ui.input(
            label="Directory",
            value=default,
        ).classes("w-full mt-2").props("dark dense outlined color=cyan")

        status_label = ui.label("").classes("text-xs mt-1")
        count_label = ui.label(
            f"{len(records)} record(s) will be saved."
        ).classes("text-xs text-slate-500 mt-2")

        save_btn = ui.button("Save Here", icon="save").props("flat").classes(
            "bg-cyan-600/20 text-cyan-300 hover:bg-cyan-600/30"
        )
        cancel_btn = ui.button("Cancel").props("flat").classes(
            "text-slate-400"
        )

        def _on_path_change() -> None:
            valid, msg = _validate_save_path(path_input.value)
            status_label.text = msg
            if valid:
                status_label.classes(replace="text-xs mt-1 text-emerald-400")
            else:
                status_label.classes(replace="text-xs mt-1 text-red-400")
            if valid:
                save_btn.enable()
            else:
                save_btn.disable()

        path_input.on("update:model-value", lambda: _on_path_change())

        async def _on_save() -> None:
            p = Path(path_input.value.strip()).expanduser().resolve()
            store = JSONBenchmarkStore(base_dir=p)
            save_btn.disable()
            cancel_btn.disable()
            saved, errors = 0, []
            for record in records:
                try:
                    store.save(record)
                    saved += 1
                except Exception as exc:
                    errors.append(f"{record.benchmark_id[:8]}: {exc}")
            if errors:
                ui.notify(
                    f"Saved {saved}/{len(records)}. "
                    f"Errors: {'; '.join(errors)}",
                    type="warning",
                    position="top",
                )
                save_btn.enable()
                cancel_btn.enable()
            else:
                ui.notify(
                    f"All {saved} record(s) saved to {p}",
                    type="positive",
                    position="top",
                )
                dialog.close()

        save_btn.on_click(_on_save)
        cancel_btn.on_click(dialog.close)

        # Run initial validation
        _on_path_change()

    dialog.open()


# ---------------------------------------------------------------------------
# Single-report view (used for both flat and tabbed modes)
# ---------------------------------------------------------------------------
def _render_report_view(
    data: ReportData,
    store: Optional[BenchmarkStore],
) -> None:
    """Render a complete report view from pre-computed data."""
    # Executive summary
    _exec_summary(data)

    # Combined step grid
    _combined_grid(data)

    # Per-workflow expansion panels
    ui.label("Workflow Details").classes(
        "text-sm font-semibold text-slate-400 uppercase tracking-widest mt-6 mb-2"
    )
    # Records already sorted slowest-first by engine
    for i, rd in enumerate(data.records, start=1):
        _workflow_panel(rd, i)


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
        Persistence backend.  When provided, its directory is used
        as the default path in the save dialog.

    Raises
    ------
    ValueError
        If *records* is empty.
    """
    if not records:
        raise ValueError("Cannot launch dashboard with empty records list.")

    # Normalise input: detect grouped vs flat format.
    groups: List[ReportGroup]
    if records and isinstance(records[0], tuple):
        groups = records  # type: ignore[assignment]
    else:
        groups = [("All Records", records)]  # type: ignore[list-item]

    all_records = [r for _, recs in groups for r in recs]

    # Pre-compute report data for each group
    group_data: List[Tuple[str, ReportData]] = []
    for label, group_records in groups:
        group_data.append((label, build_report_data(group_records)))

    @ui.page("/")
    def dashboard() -> None:
        ui.dark_mode(True)
        ui.add_head_html(
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
            "<style>"
            "body { background-color: #0f172a !important; font-family: 'Inter', sans-serif; }"
            ".q-table--dark .q-table__bottom, .q-table--dark td, .q-table--dark th,"
            ".q-table--dark thead, .q-table--dark tr { border-color: rgba(255,255,255,0.06) !important; }"
            ".q-tabs__content { overflow-x: auto !important; scroll-behavior: smooth; }"
            ".q-tabs__content::-webkit-scrollbar { height: 3px; }"
            ".q-tabs__content::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }"
            ".q-tabs__content::-webkit-scrollbar-track { background: transparent; }"
            "</style>"
        )

        with ui.column().classes(
            "w-full max-w-7xl mx-auto p-6 min-h-screen gap-4"
        ):
            # Header bar
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("GRDL Benchmark Dashboard").classes(
                    "text-xl font-semibold text-slate-200 tracking-wide"
                )
                with ui.row().classes("gap-4 items-center"):
                    ui.badge(
                        f"{len(all_records)} record{'s' if len(all_records) != 1 else ''}",
                    ).classes(
                        "bg-slate-700 text-slate-300"
                    )

                    default_dir = None
                    if store is not None and hasattr(store, '_base_dir'):
                        default_dir = str(store._base_dir)

                    ui.button(
                        "Save to Storage",
                        icon="save",
                        on_click=lambda: _show_save_dialog(
                            all_records, default_dir
                        ),
                    ).props("flat").classes(
                        "bg-cyan-600/20 text-cyan-300 hover:bg-cyan-600/30"
                    )

            # Single group → render flat (no tabs)
            if len(group_data) == 1:
                _render_report_view(group_data[0][1], store)
            else:
                # Multiple groups → tabbed interface
                with ui.tabs().classes("w-full").props(
                    "dark dense mobile-arrows outside-arrows"
                ) as tabs:
                    tab_objects = []
                    for label, _ in group_data:
                        tab_objects.append(ui.tab(label))

                with ui.tab_panels(tabs).classes(
                    "w-full bg-slate-900"
                ).props("dark"):
                    for tab_obj, (label, data) in zip(
                        tab_objects, group_data
                    ):
                        with ui.tab_panel(tab_obj):
                            _render_report_view(data, store)

    ui.run(reload=False, title="GRDL Benchmark Dashboard")
