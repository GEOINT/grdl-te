# -*- coding: utf-8 -*-
"""
Passive Benchmark Runner — forensic analysis of pre-recorded execution traces.

Accepts a list of :class:`ForensicExecutionTrace` objects produced by
:class:`ForensicTraceReader` and produces a :class:`BenchmarkRecord`
with ``benchmark_type="passive"``.  The record is structurally identical
to one produced by :class:`ActiveBenchmarkRunner` — it uses the same
aggregation code path, the same ``StepBenchmarkResult`` construction,
and the same topology classification.

Key difference from ``ActiveBenchmarkRunner``: this runner never executes
any workflow.  It reconstitutes ``StepMetrics`` objects from the stored
JSON traces and feeds them through the shared aggregation functions in
``_aggregation.py``.

**Hardware resolution policy** (no fallback to current machine):

1. Use ``hardware`` if explicitly provided to ``__init__``.
2. If all traces have embedded hardware *and* share identical hardware
   details (CPU count, RAM, GPU configuration), reconstruct from the
   first trace.
3. Otherwise set ``hardware=None`` on the record.  The report formatters
   will render "hardware information missing" rather than showing
   misleading data.

**Step dependency graph**: injected from
``trace.metrics_dict["step_depends_on"]`` when present.  This field is
populated by grdl-runtime's ``WorkflowExecutor`` (DAG path) for runs
produced after forensic benchmarking support was added.  Linear pipeline
runs (``Workflow`` builder) will have ``step_depends_on=None``, which
causes the topology classifier to fall back to ``SEQUENTIAL`` — correct
behaviour for linear workflows.

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-17

Modified
--------
2026-03-17
"""

# Standard library
import logging
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# Internal
from grdl_te.benchmarking._aggregation import (
    aggregate_step_metrics,
    aggregate_workflow_totals,
    apply_topology_and_contributions,
)
from grdl_te.benchmarking.base import BenchmarkRunner, BenchmarkStore
from grdl_te.benchmarking.forensic import ForensicExecutionTrace
from grdl_te.benchmarking.models import BenchmarkRecord, HardwareSnapshot

logger = logging.getLogger(__name__)


class PassiveBenchmarkRunner(BenchmarkRunner):
    """Produce a BenchmarkRecord from pre-recorded execution traces.

    Does not execute any workflow.  Treats each provided trace as one
    measurement iteration and aggregates them exactly as
    :class:`ActiveBenchmarkRunner` would aggregate live runs.

    Parameters
    ----------
    traces : List[ForensicExecutionTrace]
        Pre-loaded execution traces.  All must be from the same workflow
        (same ``workflow_name`` and ``workflow_version``).  At least
        one trace is required.
    tags : Dict[str, str], optional
        User-defined labels attached to the record.  Two tags are
        automatically added: ``forensic_source`` (comma-separated
        list of source types) and ``trace_count`` (number of traces).
    store : BenchmarkStore, optional
        If provided, the resulting ``BenchmarkRecord`` is persisted
        automatically after ``run()``.
    hardware : HardwareSnapshot, optional
        Explicit hardware snapshot override.  When ``None`` (default),
        the runner attempts to reconstruct hardware from the embedded
        ``hardware`` field in the traces.  If the traces have no
        embedded hardware, or originate from mixed hostnames,
        ``hardware`` is set to ``None`` on the record and the report
        formatters will display "hardware information missing".

    Raises
    ------
    ValueError
        If *traces* is empty, or if the traces span more than one
        ``workflow_name`` or ``workflow_version``.

    Examples
    --------
    >>> traces = ForensicTraceReader.from_json_file("run_output.json")
    >>> record = PassiveBenchmarkRunner(traces).run()
    >>> assert record.benchmark_type == "passive"
    """

    def __init__(
        self,
        traces: List[ForensicExecutionTrace],
        *,
        tags: Optional[Dict[str, str]] = None,
        store: Optional[BenchmarkStore] = None,
        hardware: Optional[HardwareSnapshot] = None,
    ) -> None:
        if not traces:
            raise ValueError(
                "traces must contain at least one ForensicExecutionTrace"
            )

        names = {t.workflow_name for t in traces}
        versions = {t.workflow_version for t in traces}
        if len(names) > 1:
            raise ValueError(
                f"All traces must be from the same workflow. "
                f"Found multiple names: {sorted(names)}"
            )
        if len(versions) > 1:
            raise ValueError(
                f"All traces must share the same workflow version. "
                f"Found: {sorted(versions)}"
            )

        self._traces = traces
        self._tags = tags or {}
        self._store = store
        self._hardware = hardware

    @property
    def benchmark_type(self) -> str:
        """Return ``"passive"``."""
        return "passive"

    def _resolve_hardware(self) -> Optional[HardwareSnapshot]:
        """Determine the hardware snapshot to attach to the record.

        Compares the raw hardware dicts embedded in each trace to detect
        whether the traces were produced on consistent hardware.  Hostname
        is not used because it is not captured by grdl-runtime; instead
        the hardware metrics themselves (CPU count, RAM, GPU availability)
        determine consistency.

        Returns
        -------
        HardwareSnapshot or None
            ``None`` when hardware cannot be reliably determined from the
            traces (missing field, or hardware details changed across traces).
        """
        # Explicit override always wins
        if self._hardware is not None:
            return self._hardware

        hw_dicts = [t.metrics_dict.get("hardware") for t in self._traces]

        if any(hw is None for hw in hw_dicts):
            if any(hw is not None for hw in hw_dicts):
                warnings.warn(
                    "Some traces have no embedded hardware information. "
                    "Hardware will be omitted from the benchmark record.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return None

        # Compare hardware details across all traces to detect machine changes.
        # Normalise to a frozenset of items so dict ordering doesn't matter.
        def _hw_fingerprint(hw: Dict[str, Any]) -> frozenset:
            # gpu_devices is a list of dicts — convert to a comparable form
            gpu = tuple(
                tuple(sorted(d.items())) for d in hw.get("gpu_devices", [])
            )
            return frozenset({
                "cpu_count": hw.get("cpu_count"),
                "total_memory_bytes": hw.get("total_memory_bytes"),
                "gpu_available": hw.get("gpu_available"),
                "gpu_memory_bytes": hw.get("gpu_memory_bytes"),
                "gpu_devices": gpu,
            }.items())

        fingerprints = {_hw_fingerprint(hw) for hw in hw_dicts}  # type: ignore[arg-type]
        if len(fingerprints) > 1:
            warnings.warn(
                "Hardware details changed across traces (different CPU count, "
                "RAM, or GPU configuration). Hardware will be omitted from "
                "the benchmark record to avoid misleading information.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None

        return self._traces[0].to_hardware_snapshot()

    def run(self) -> BenchmarkRecord:
        """Aggregate pre-recorded traces into a BenchmarkRecord.

        Returns
        -------
        BenchmarkRecord
            ``benchmark_type="passive"``, structurally identical to a
            record produced by :class:`ActiveBenchmarkRunner`.
        """
        hardware = self._resolve_hardware()

        # Wrap each trace's metrics_dict in a SimpleNamespace that
        # duck-types as a WorkflowMetrics object for the aggregation
        # functions (they only need .total_wall_time_s, .total_cpu_time_s,
        # .peak_rss_bytes, and .step_metrics).
        all_workflow_metrics: List[Any] = []
        for trace in self._traces:
            step_metrics = trace.to_step_metrics()
            d = trace.metrics_dict
            all_workflow_metrics.append(
                SimpleNamespace(
                    step_metrics=step_metrics,
                    total_wall_time_s=float(d["total_wall_time_s"]),
                    total_cpu_time_s=float(d["total_cpu_time_s"]),
                    peak_rss_bytes=int(d["peak_rss_bytes"]),
                )
            )

        total_wall, total_cpu, total_rss = aggregate_workflow_totals(
            all_workflow_metrics
        )
        step_results = aggregate_step_metrics(all_workflow_metrics)

        # Inject step_depends_on from first trace.  The dependency graph
        # is structural (same for all runs of a workflow), so the first
        # trace is authoritative.  If step_depends_on is absent (linear
        # pipelines, or old traces), topology falls back to SEQUENTIAL.
        dep_map: Dict[str, List[str]] = (
            self._traces[0].metrics_dict.get("step_depends_on") or {}
        )
        if dep_map:
            for sr in step_results:
                if sr.step_id and sr.step_id in dep_map:
                    sr.depends_on = dep_map[sr.step_id]

        # Build provenance tags
        source_types = {t.source_type for t in self._traces}
        merged_tags: Dict[str, str] = {
            "forensic_source": ",".join(sorted(source_types)),
            "trace_count": str(len(self._traces)),
        }
        merged_tags.update(self._tags)

        raw_metrics = [t.metrics_dict for t in self._traces]
        wf_name = self._traces[0].workflow_name
        wf_version = self._traces[0].workflow_version

        record = BenchmarkRecord.create(
            benchmark_type=self.benchmark_type,
            workflow_name=wf_name,
            workflow_version=wf_version,
            iterations=len(self._traces),
            hardware=hardware,
            total_wall_time=total_wall,
            total_cpu_time=total_cpu,
            total_peak_rss=total_rss,
            step_results=step_results,
            raw_metrics=raw_metrics,
            tags=merged_tags,
        )

        apply_topology_and_contributions(record)

        if self._store is not None:
            self._store.save(record)

        return record
