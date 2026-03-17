# -*- coding: utf-8 -*-
"""
Benchmark Data Models — structured result types for performance evaluation.

Provides dataclasses for capturing, aggregating, and persisting benchmark
results.  ``HardwareSnapshot`` freezes machine state at benchmark time,
``AggregatedMetrics`` computes statistics across repeated measurements,
``StepBenchmarkResult`` aggregates per-step metrics, and ``BenchmarkRecord``
is the atomic unit of persistence that wraps everything together.

All models support JSON round-tripping via ``to_dict()`` / ``from_dict()``.

Dependencies
------------
numpy

Author
------
Ava Courtney

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-13

Modified
--------
2026-02-18
"""

# Standard library
import json
import logging
import platform
import socket
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Third-party
import numpy as np

logger = logging.getLogger(__name__)

# Optional: grdl-runtime hardware context
try:
    from grdl_rt.execution.hardware import LocalHardwareContext
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False


class WorkflowTopology(Enum):
    """Classification of workflow execution topology.

    Attributes
    ----------
    COMPONENT : str
        Single callable benchmarked in isolation.
    SEQUENTIAL : str
        Linear step chain with no parallel branches.
    PARALLEL : str
        DAG with concurrent branches.
    MIXED : str
        Combination of sequential and parallel sections.
    """

    COMPONENT = "component"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MIXED = "mixed"


@dataclass(frozen=True)
class TopologyDescriptor:
    """Structural metadata describing a workflow's execution graph.

    Computed post-hoc from ``BenchmarkRecord`` step results after
    all iterations complete.  The critical path is the dependency
    chain with the longest accumulated wall time (not the most
    steps).

    Attributes
    ----------
    topology : WorkflowTopology
        Classification of the execution pattern.
    num_branches : int
        Number of parallel root branches (0 for sequential/component).
    critical_path_step_ids : tuple
        Ordered step IDs along the critical path.
    critical_path_wall_time_s : float
        Sum of mean wall times along the critical path.
    sum_of_steps_wall_time_s : float
        Sum of all step mean wall times (hypothetical sequential time).
    parallelism_ratio : float
        ``sum_of_steps / actual_wall_time``.  1.0 = purely sequential.
    """

    topology: WorkflowTopology
    num_branches: int = 0
    critical_path_step_ids: tuple = ()
    critical_path_wall_time_s: float = 0.0
    sum_of_steps_wall_time_s: float = 0.0
    parallelism_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "topology": self.topology.value,
            "num_branches": self.num_branches,
            "critical_path_step_ids": list(self.critical_path_step_ids),
            "critical_path_wall_time_s": self.critical_path_wall_time_s,
            "sum_of_steps_wall_time_s": self.sum_of_steps_wall_time_s,
            "parallelism_ratio": self.parallelism_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopologyDescriptor':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        TopologyDescriptor
        """
        return cls(
            topology=WorkflowTopology(data["topology"]),
            num_branches=data.get("num_branches", 0),
            critical_path_step_ids=tuple(
                data.get("critical_path_step_ids", ())
            ),
            critical_path_wall_time_s=data.get(
                "critical_path_wall_time_s", 0.0
            ),
            sum_of_steps_wall_time_s=data.get(
                "sum_of_steps_wall_time_s", 0.0
            ),
            parallelism_ratio=data.get("parallelism_ratio", 1.0),
        )


@dataclass(frozen=True)
class HardwareSnapshot:
    """Frozen hardware state captured at benchmark time.

    Wraps ``LocalHardwareContext`` output with supplementary platform
    metadata.  Survives JSON serialization and can be compared across
    machines.

    Attributes
    ----------
    cpu_count : int
        Number of logical CPUs.
    total_memory_bytes : int
        Total system RAM in bytes.
    gpu_available : bool
        Whether at least one GPU was detected.
    gpu_devices : List[Dict[str, Any]]
        Per-device GPU info (name, memory_bytes, device_index).
    gpu_memory_bytes : int
        Total GPU memory across all devices.
    platform_info : str
        Platform string (e.g. ``'Linux-5.14.0-x86_64'``).
    python_version : str
        Python version string.
    hostname : str
        Machine hostname.
    captured_at : str
        ISO 8601 UTC timestamp of capture.
    """

    cpu_count: int
    total_memory_bytes: int
    gpu_available: bool
    gpu_devices: tuple  # tuple of dicts for frozen hashability
    gpu_memory_bytes: int
    platform_info: str
    python_version: str
    hostname: str
    captured_at: str

    @classmethod
    def capture(cls, hw: Any = None) -> 'HardwareSnapshot':
        """Capture current hardware state.

        Parameters
        ----------
        hw : HardwareContext, optional
            Pre-existing hardware context.  If ``None``, creates a new
            ``LocalHardwareContext`` (requires grdl-runtime).  When
            hardware detection is unavailable, all hardware fields are
            zeroed out.

        Returns
        -------
        HardwareSnapshot
        """
        now = datetime.now(timezone.utc).isoformat()
        empty = {
            "cpu_count": 0,
            "total_memory_bytes": 0,
            "available_memory_bytes": 0,
            "gpu_available": False,
            "gpu_devices": [],
            "gpu_memory_bytes": 0,
        }

        if hw is not None:
            hw_dict = hw.to_dict()
        elif _HAS_RUNTIME:
            try:
                hw_dict = LocalHardwareContext().to_dict()
            except Exception:
                logger.warning(
                    "LocalHardwareContext() raised an exception; "
                    "hardware fields will be zeroed. Install or "
                    "update grdl-runtime, or pass a HardwareContext "
                    "to capture(hw=...) to provide hardware info.",
                    exc_info=True,
                )
                hw_dict = empty
        else:
            logger.warning(
                "grdl-runtime is not installed; hardware fields will "
                "be zeroed. Install grdl-runtime or pass a "
                "HardwareContext to capture(hw=...) to provide "
                "hardware info.",
            )
            hw_dict = empty

        return cls(
            cpu_count=hw_dict["cpu_count"],
            total_memory_bytes=hw_dict["total_memory_bytes"],
            gpu_available=hw_dict["gpu_available"],
            gpu_devices=tuple(hw_dict.get("gpu_devices", [])),
            gpu_memory_bytes=hw_dict.get("gpu_memory_bytes", 0),
            platform_info=platform.platform(),
            python_version=sys.version,
            hostname=socket.gethostname(),
            captured_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "cpu_count": self.cpu_count,
            "total_memory_bytes": self.total_memory_bytes,
            "gpu_available": self.gpu_available,
            "gpu_devices": list(self.gpu_devices),
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "platform_info": self.platform_info,
            "python_version": self.python_version,
            "hostname": self.hostname,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareSnapshot':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        HardwareSnapshot
        """
        return cls(
            cpu_count=data["cpu_count"],
            total_memory_bytes=data["total_memory_bytes"],
            gpu_available=data["gpu_available"],
            gpu_devices=tuple(data.get("gpu_devices", [])),
            gpu_memory_bytes=data.get("gpu_memory_bytes", 0),
            platform_info=data["platform_info"],
            python_version=data["python_version"],
            hostname=data["hostname"],
            captured_at=data["captured_at"],
        )


@dataclass(frozen=True)
class AggregatedMetrics:
    """Statistical aggregation of a metric across N measurements.

    Attributes
    ----------
    count : int
        Number of measurements.
    min : float
        Minimum value.
    max : float
        Maximum value.
    mean : float
        Arithmetic mean.
    median : float
        Median value.
    stddev : float
        Sample standard deviation (ddof=1 when N > 1, else 0).
    p95 : float
        95th percentile.
    values : tuple
        Raw measurement values (tuple for immutability).
    """

    count: int
    min: float
    max: float
    mean: float
    median: float
    stddev: float
    p95: float
    values: tuple

    @classmethod
    def from_values(cls, values: List[float]) -> 'AggregatedMetrics':
        """Compute aggregated statistics from raw values.

        Parameters
        ----------
        values : List[float]
            Raw measurement values.  Must contain at least one element.

        Returns
        -------
        AggregatedMetrics

        Raises
        ------
        ValueError
            If *values* is empty.
        """
        if not values:
            raise ValueError("Cannot aggregate empty values list.")

        arr = np.asarray(values, dtype=np.float64)
        ddof = 1 if len(arr) > 1 else 0

        return cls(
            count=len(arr),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            stddev=float(np.std(arr, ddof=ddof)),
            p95=float(np.percentile(arr, 95)),
            values=tuple(float(v) for v in arr),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "stddev": self.stddev,
            "p95": self.p95,
            "values": list(self.values),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregatedMetrics':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        AggregatedMetrics
        """
        return cls(
            count=data["count"],
            min=data["min"],
            max=data["max"],
            mean=data["mean"],
            median=data["median"],
            stddev=data["stddev"],
            p95=data["p95"],
            values=tuple(data["values"]),
        )


@dataclass
class StepBenchmarkResult:
    """Aggregated metrics for a single workflow step across N runs.

    Attributes
    ----------
    step_index : int
        Zero-based position within the workflow.
    processor_name : str
        Name of the processor executed.
    wall_time_s : AggregatedMetrics
        Wall-clock time statistics in seconds.
    cpu_time_s : AggregatedMetrics
        CPU time statistics in seconds.
    peak_rss_bytes : AggregatedMetrics
        Peak memory delta statistics in bytes.
    gpu_used : bool
        Whether GPU was used (consistent across runs).
    gpu_memory_bytes : Optional[AggregatedMetrics]
        GPU memory statistics, if applicable.
    sample_count : int
        Number of runs aggregated.
    step_id : Optional[str]
        Stable step identifier for DAG workflows.  ``None`` for
        linear workflows or pre-existing records.
    concurrent : bool
        Whether this step ran concurrently with other steps.  When
        ``True``, ``peak_rss_bytes`` is the shared level-wide peak
        (not isolated to this step).
    latency_pct : float
        Percentage contribution to total workflow wall-clock time.
        For sequential steps: ``step_wall / total_wall * 100``.
        For non-critical parallel steps: ``0.0`` (hidden by the
        critical path).  Populated by topology analysis.
    memory_pct : float
        Percentage contribution to workflow peak memory.  All steps
        contribute regardless of whether they are on the critical
        path — memory is process-wide.  Populated by topology
        analysis.
    """

    step_index: int
    processor_name: str
    wall_time_s: AggregatedMetrics
    cpu_time_s: AggregatedMetrics
    peak_rss_bytes: AggregatedMetrics
    gpu_used: bool
    gpu_memory_bytes: Optional[AggregatedMetrics] = None
    sample_count: int = 0
    step_id: Optional[str] = None
    concurrent: bool = False
    depends_on: Optional[List[str]] = None
    peak_overhead_bytes: Optional[AggregatedMetrics] = None
    end_of_step_footprint_bytes: Optional[AggregatedMetrics] = None
    latency_pct: float = 0.0
    memory_pct: float = 0.0

    @classmethod
    def from_step_metrics(cls, metrics: list) -> 'StepBenchmarkResult':
        """Build from a list of grdl-runtime ``StepMetrics`` objects.

        All metrics must belong to the same step (same ``step_index``
        or ``step_id``).

        Parameters
        ----------
        metrics : List[StepMetrics]
            Per-run step metrics for a single workflow step.

        Returns
        -------
        StepBenchmarkResult

        Raises
        ------
        ValueError
            If *metrics* is empty, or metrics belong to different
            processors (heterogeneous step_index collision).
        """
        if not metrics:
            raise ValueError("Cannot aggregate empty metrics list.")

        step_index = metrics[0].step_index
        processor_name = metrics[0].processor_name

        # Validate all metrics come from the same processor
        processor_names = {m.processor_name for m in metrics}
        if len(processor_names) > 1:
            raise ValueError(
                f"Heterogeneous metrics detected at step_index "
                f"{step_index}: expected one processor, got "
                f"{processor_names}. This usually indicates a "
                f"step_index collision in a parallel workflow."
            )

        # Extract step_id if consistently set
        step_ids = {getattr(m, 'step_id', None) for m in metrics}
        step_ids.discard(None)
        step_id = step_ids.pop() if len(step_ids) == 1 else None

        wall_times = [m.wall_time_s for m in metrics]
        cpu_times = [m.cpu_time_s for m in metrics]
        peak_rss = [float(m.peak_rss_bytes) for m in metrics]
        gpu_used = any(m.gpu_used for m in metrics)

        gpu_mem_values = [
            float(m.gpu_memory_bytes)
            for m in metrics
            if m.gpu_memory_bytes is not None
        ]
        gpu_mem = (
            AggregatedMetrics.from_values(gpu_mem_values)
            if gpu_mem_values
            else None
        )

        concurrent = any(getattr(m, 'concurrent', False) for m in metrics)

        overhead_values = [
            float(getattr(m, 'peak_overhead_bytes', 0) or 0)
            for m in metrics
        ]
        footprint_values = [
            float(getattr(m, 'end_of_step_footprint_bytes', 0) or 0)
            for m in metrics
        ]
        peak_overhead = (
            AggregatedMetrics.from_values(overhead_values)
            if any(v > 0 for v in overhead_values)
            else None
        )
        end_footprint = (
            AggregatedMetrics.from_values(footprint_values)
            if any(v > 0 for v in footprint_values)
            else None
        )

        return cls(
            step_index=step_index,
            processor_name=processor_name,
            wall_time_s=AggregatedMetrics.from_values(wall_times),
            cpu_time_s=AggregatedMetrics.from_values(cpu_times),
            peak_rss_bytes=AggregatedMetrics.from_values(peak_rss),
            gpu_used=gpu_used,
            gpu_memory_bytes=gpu_mem,
            sample_count=len(metrics),
            step_id=step_id,
            concurrent=concurrent,
            peak_overhead_bytes=peak_overhead,
            end_of_step_footprint_bytes=end_footprint,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: Dict[str, Any] = {
            "step_index": self.step_index,
            "processor_name": self.processor_name,
            "wall_time_s": self.wall_time_s.to_dict(),
            "cpu_time_s": self.cpu_time_s.to_dict(),
            "peak_rss_bytes": self.peak_rss_bytes.to_dict(),
            "gpu_used": self.gpu_used,
            "sample_count": self.sample_count,
            "concurrent": self.concurrent,
            "latency_pct": self.latency_pct,
            "memory_pct": self.memory_pct,
        }
        if self.step_id is not None:
            d["step_id"] = self.step_id
        if self.depends_on is not None:
            d["depends_on"] = self.depends_on
        if self.gpu_memory_bytes is not None:
            d["gpu_memory_bytes"] = self.gpu_memory_bytes.to_dict()
        if self.peak_overhead_bytes is not None:
            d["peak_overhead_bytes"] = self.peak_overhead_bytes.to_dict()
        if self.end_of_step_footprint_bytes is not None:
            d["end_of_step_footprint_bytes"] = self.end_of_step_footprint_bytes.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepBenchmarkResult':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        StepBenchmarkResult
        """
        gpu_mem = None
        if "gpu_memory_bytes" in data:
            gpu_mem = AggregatedMetrics.from_dict(data["gpu_memory_bytes"])

        peak_overhead = None
        if "peak_overhead_bytes" in data:
            peak_overhead = AggregatedMetrics.from_dict(data["peak_overhead_bytes"])

        end_footprint = None
        if "end_of_step_footprint_bytes" in data:
            end_footprint = AggregatedMetrics.from_dict(data["end_of_step_footprint_bytes"])

        return cls(
            step_index=data["step_index"],
            processor_name=data["processor_name"],
            wall_time_s=AggregatedMetrics.from_dict(data["wall_time_s"]),
            cpu_time_s=AggregatedMetrics.from_dict(data["cpu_time_s"]),
            peak_rss_bytes=AggregatedMetrics.from_dict(data["peak_rss_bytes"]),
            gpu_used=data["gpu_used"],
            gpu_memory_bytes=gpu_mem,
            sample_count=data["sample_count"],
            step_id=data.get("step_id"),
            concurrent=data.get("concurrent", False),
            depends_on=data.get("depends_on"),
            peak_overhead_bytes=peak_overhead,
            end_of_step_footprint_bytes=end_footprint,
            latency_pct=data.get("latency_pct", 0.0),
            memory_pct=data.get("memory_pct", 0.0),
        )


@dataclass
class BenchmarkRecord:
    """Complete record of a benchmark run.

    The atomic unit of persistence.  Every benchmark operation (active,
    passive, or component) produces one of these.

    Attributes
    ----------
    benchmark_id : str
        Unique identifier (UUID4).
    benchmark_type : str
        One of ``"active"``, ``"passive"``, ``"component"``.
    workflow_name : str
        Name of the workflow or component benchmarked.
    workflow_version : str
        Version string.
    iterations : int
        Number of measurement iterations.
    hardware : HardwareSnapshot, optional
        Hardware state at benchmark time.  ``None`` for passive (forensic)
        records when the original traces have no embedded hardware
        information (runs produced before forensic benchmarking support
        was added, or traces from mixed hostnames).
    total_wall_time : AggregatedMetrics
        Workflow-level wall-clock time statistics.
    total_cpu_time : AggregatedMetrics
        Workflow-level CPU time statistics.
    total_peak_rss : AggregatedMetrics
        Workflow-level peak memory statistics.
    step_results : List[StepBenchmarkResult]
        Per-step aggregated metrics.
    raw_metrics : List[Dict[str, Any]]
        Raw ``WorkflowMetrics.to_dict()`` per iteration.
    tags : Dict[str, str]
        User-defined labels.
    created_at : str
        ISO 8601 UTC timestamp.
    metadata : Dict[str, Any]
        Arbitrary extra information.
    topology : Optional[TopologyDescriptor]
        Workflow execution topology classification and critical-path
        data.  Populated automatically by benchmark runners after
        all iterations complete.  ``None`` for records created before
        topology analysis was available.
    step_latency_pct : Dict[str, float]
        Per-step latency contribution percentages, keyed by step_id
        (or ``"__idx_N"`` for linear workflows).
    step_memory_pct : Dict[str, float]
        Per-step memory contribution percentages, keyed by step_id
        (or ``"__idx_N"`` for linear workflows).
    """

    benchmark_id: str
    benchmark_type: str
    workflow_name: str
    workflow_version: str
    iterations: int
    hardware: Optional[HardwareSnapshot]
    total_wall_time: AggregatedMetrics
    total_cpu_time: AggregatedMetrics
    total_peak_rss: AggregatedMetrics
    step_results: List[StepBenchmarkResult] = field(default_factory=list)
    raw_metrics: List[Dict[str, Any]] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    topology: Optional[TopologyDescriptor] = None
    step_latency_pct: Dict[str, float] = field(default_factory=dict)
    step_memory_pct: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        benchmark_type: str,
        workflow_name: str,
        workflow_version: str,
        iterations: int,
        hardware: Optional[HardwareSnapshot],
        total_wall_time: AggregatedMetrics,
        total_cpu_time: AggregatedMetrics,
        total_peak_rss: AggregatedMetrics,
        step_results: Optional[List[StepBenchmarkResult]] = None,
        raw_metrics: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'BenchmarkRecord':
        """Create a new record with auto-generated ID and timestamp.

        Parameters
        ----------
        benchmark_type : str
            One of ``"active"``, ``"passive"``, ``"component"``.
        workflow_name : str
            Name of the workflow or component.
        workflow_version : str
            Version string.
        iterations : int
            Number of measurement iterations.
        hardware : HardwareSnapshot, optional
            Hardware snapshot.  Pass ``None`` for forensic records when
            the original hardware is unavailable.
        total_wall_time : AggregatedMetrics
            Overall wall-clock time stats.
        total_cpu_time : AggregatedMetrics
            Overall CPU time stats.
        total_peak_rss : AggregatedMetrics
            Overall peak memory stats.
        step_results : List[StepBenchmarkResult], optional
            Per-step results.
        raw_metrics : List[Dict], optional
            Raw per-iteration metrics.
        tags : Dict[str, str], optional
            User labels.
        metadata : Dict[str, Any], optional
            Extra info.

        Returns
        -------
        BenchmarkRecord
        """
        return cls(
            benchmark_id=str(uuid.uuid4()),
            benchmark_type=benchmark_type,
            workflow_name=workflow_name,
            workflow_version=workflow_version,
            iterations=iterations,
            hardware=hardware,
            total_wall_time=total_wall_time,
            total_cpu_time=total_cpu_time,
            total_peak_rss=total_peak_rss,
            step_results=step_results or [],
            raw_metrics=raw_metrics or [],
            tags=tags or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: Dict[str, Any] = {
            "benchmark_id": self.benchmark_id,
            "benchmark_type": self.benchmark_type,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "iterations": self.iterations,
            "hardware": self.hardware.to_dict() if self.hardware is not None else None,
            "total_wall_time": self.total_wall_time.to_dict(),
            "total_cpu_time": self.total_cpu_time.to_dict(),
            "total_peak_rss": self.total_peak_rss.to_dict(),
            "step_results": [s.to_dict() for s in self.step_results],
            "raw_metrics": self.raw_metrics,
            "tags": self.tags,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
        if self.topology is not None:
            d["topology"] = self.topology.to_dict()
        if self.step_latency_pct:
            d["step_latency_pct"] = self.step_latency_pct
        if self.step_memory_pct:
            d["step_memory_pct"] = self.step_memory_pct
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkRecord':
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        BenchmarkRecord
        """
        topo_data = data.get("topology")
        topology = (
            TopologyDescriptor.from_dict(topo_data)
            if topo_data is not None
            else None
        )

        return cls(
            benchmark_id=data["benchmark_id"],
            benchmark_type=data["benchmark_type"],
            workflow_name=data["workflow_name"],
            workflow_version=data["workflow_version"],
            iterations=data["iterations"],
            hardware=(
                HardwareSnapshot.from_dict(data["hardware"])
                if data.get("hardware") is not None
                else None
            ),
            total_wall_time=AggregatedMetrics.from_dict(
                data["total_wall_time"]
            ),
            total_cpu_time=AggregatedMetrics.from_dict(
                data["total_cpu_time"]
            ),
            total_peak_rss=AggregatedMetrics.from_dict(
                data["total_peak_rss"]
            ),
            step_results=[
                StepBenchmarkResult.from_dict(s)
                for s in data.get("step_results", [])
            ],
            raw_metrics=data.get("raw_metrics", []),
            tags=data.get("tags", {}),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
            topology=topology,
            step_latency_pct=data.get("step_latency_pct", {}),
            step_memory_pct=data.get("step_memory_pct", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level.

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkRecord':
        """Deserialize from JSON string.

        Parameters
        ----------
        json_str : str

        Returns
        -------
        BenchmarkRecord
        """
        return cls.from_dict(json.loads(json_str))
