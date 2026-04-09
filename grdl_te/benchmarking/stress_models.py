# -*- coding: utf-8 -*-
"""
Stress Test Data Models — structured result types for stress evaluation.

Provides dataclasses for specifying, executing, and persisting stress test
results.  ``StressTestConfig`` defines the ramp parameters, ``StressTestEvent``
captures the outcome of a single concurrent call, ``FailurePoint`` records
where and how a component broke, ``StressTestSummary`` synthesises headline
findings, and ``StressTestRecord`` is the atomic unit of persistence.

All models support JSON round-tripping via ``to_dict()`` / ``from_dict()``.
The ``schema_version`` field on ``StressTestRecord`` is an integer that is
incremented when the schema changes, enabling migration logic in
``from_dict()`` and reliable comparison between runs against different grdl
versions.

Dependencies
------------
numpy

Author
------
Ava Courtney <courtney-ava@zai.com>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-04-07

Modified
--------
2026-04-07
"""

# Standard library
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Internal — reuse existing hardware snapshot
from grdl_te.benchmarking.models import HardwareSnapshot

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
STRESS_SCHEMA_VERSION: int = 2
"""Increment this whenever the ``StressTestRecord`` on-disk schema changes.

Changelog
---------
2 : Added ``run_until_failure``, ``failure_escalation_mode``,
    ``failure_threshold_pct``, ``max_wall_time_s`` to ``StressTestConfig``;
    added ``wall_time_budget_exceeded`` and ``saturation_payload_shape`` to
    ``StressTestSummary``.
3 : Added ``ramp_mode`` to ``StressTestConfig`` as the canonical ramp-axis
    selector (supersedes ``failure_escalation_mode`` which is now legacy).
"""

# ---------------------------------------------------------------------------
# Default configuration constants
# ---------------------------------------------------------------------------
DEFAULT_START_CONCURRENCY: int = 1
DEFAULT_MAX_CONCURRENCY: int = 16
DEFAULT_RAMP_STEPS: int = 5
DEFAULT_DURATION_PER_STEP_S: float = 10.0
DEFAULT_PAYLOAD_SIZE: str = "medium"
DEFAULT_TIMEOUT_PER_CALL_S: float = 30.0
DEFAULT_RUN_UNTIL_FAILURE: bool = False
DEFAULT_FAILURE_ESCALATION_MODE: str = "concurrency"  # legacy name kept for old JSON compat
DEFAULT_RAMP_MODE: str = "concurrency"
"""Primary ramp axis: ``"concurrency"`` or ``"payload"``."""
DEFAULT_FAILURE_THRESHOLD_PCT: float = 10.0
DEFAULT_MAX_WALL_TIME_S: Optional[float] = None
DEFAULT_MAX_ESCALATION_CONCURRENCY: int = 512
"""Default hard ceiling for concurrency escalation (workers)."""
DEFAULT_MAX_ESCALATION_PAYLOAD_DIM: int = 16384
"""Default hard ceiling for payload escalation (linear dimension in pixels)."""

_SENTINEL = object()  # used to detect "not provided" for Optional[int] fields


# ---------------------------------------------------------------------------
# StressTestConfig
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StressTestConfig:
    """Complete specification for a stress test ramp.

    Attributes
    ----------
    start_concurrency : int
        First concurrency level in the ramp.  Must be >= 1.
    max_concurrency : int
        Upper bound for the ramp.  The ramp will not exceed this value.
    ramp_steps : int
        Number of discrete concurrency levels to evaluate.  The levels
        are computed via geometric doubling from *start_concurrency* up
        to *max_concurrency*.
    duration_per_step_s : float
        How long (in seconds) to sustain each concurrency level before
        advancing.  Total run time is bounded by
        ``ramp_steps * duration_per_step_s``.
    payload_size : str
        Array size preset key (``"small"``, ``"medium"``, ``"large"``)
        or a ``"ROWSxCOLS"`` string (e.g. ``"512x512"``).  Resolved by
        the runner against ``ARRAY_SIZES``.
    timeout_per_call_s : float
        Maximum seconds allowed for a single call.  Calls that exceed
        this are recorded as ``TimeoutError`` events.
    run_until_failure : bool
        When ``True``, the runner continues escalating the chosen
        dimension past *max_concurrency* (or the base payload size)
        after the standard ramp finishes, until
        *failure_threshold_pct* % of calls fail at a level or
        *max_wall_time_s* is exceeded.  Exactly one escalation axis is
        active at a time; see *failure_escalation_mode*.
    ramp_mode : str
        Primary ramp axis.  ``"concurrency"`` (default) runs the standard
        worker-count ramp at a fixed payload size.  ``"payload"`` fixes
        concurrency at *max_concurrency* and ramps payload size geometrically,
        skipping the worker ramp entirely.  When *run_until_failure* is
        ``True`` this becomes the escalation axis.  Supersedes the legacy
        ``failure_escalation_mode`` field.
    failure_threshold_pct : float
        Percentage of calls at a level that must fail before the runner
        declares saturation and stops.  Default 10 — one call in ten
        failing is a meaningful reliability boundary.
    max_wall_time_s : float, optional
        Hard budget for total run time across all steps (including the
        normal ramp and the escalation phase).  ``None`` means the run
        is bounded only by the configured ceilings.
    max_escalation_concurrency : int
        Hard ceiling for the number of concurrent workers when
        ``failure_escalation_mode='concurrency'``.  The runner will
        never exceed this regardless of how many doublings it takes.
        Default ``512``.  Set lower if your machine has fewer cores or
        you want faster results.
    max_escalation_payload_dim : int
        Hard ceiling on the linear dimension (rows and cols) of the
        payload array when ``failure_escalation_mode='payload'``.
        A ``16384×16384`` float32 array occupies ~1 GB; reduce this
        if you want to protect available RAM.  Default ``16384``.
    """

    start_concurrency: int = DEFAULT_START_CONCURRENCY
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    ramp_steps: int = DEFAULT_RAMP_STEPS
    duration_per_step_s: float = DEFAULT_DURATION_PER_STEP_S
    payload_size: str = DEFAULT_PAYLOAD_SIZE
    timeout_per_call_s: float = DEFAULT_TIMEOUT_PER_CALL_S
    run_until_failure: bool = DEFAULT_RUN_UNTIL_FAILURE
    failure_escalation_mode: str = DEFAULT_FAILURE_ESCALATION_MODE  # legacy; use ramp_mode
    ramp_mode: str = DEFAULT_RAMP_MODE
    failure_threshold_pct: float = DEFAULT_FAILURE_THRESHOLD_PCT
    max_wall_time_s: Optional[float] = DEFAULT_MAX_WALL_TIME_S
    max_escalation_concurrency: int = DEFAULT_MAX_ESCALATION_CONCURRENCY
    max_escalation_payload_dim: int = DEFAULT_MAX_ESCALATION_PAYLOAD_DIM

    def __post_init__(self) -> None:
        if self.start_concurrency < 1:
            raise ValueError(
                f"start_concurrency must be >= 1, got {self.start_concurrency}"
            )
        if self.max_concurrency < self.start_concurrency:
            raise ValueError(
                f"max_concurrency ({self.max_concurrency}) must be >= "
                f"start_concurrency ({self.start_concurrency})"
            )
        if self.ramp_steps < 1:
            raise ValueError(
                f"ramp_steps must be >= 1, got {self.ramp_steps}"
            )
        if self.duration_per_step_s <= 0:
            raise ValueError(
                f"duration_per_step_s must be > 0, got {self.duration_per_step_s}"
            )
        if self.timeout_per_call_s <= 0:
            raise ValueError(
                f"timeout_per_call_s must be > 0, got {self.timeout_per_call_s}"
            )
        if self.failure_escalation_mode not in ("concurrency", "payload"):
            raise ValueError(
                f"failure_escalation_mode must be 'concurrency' or 'payload', "
                f"got {self.failure_escalation_mode!r}"
            )
        if self.ramp_mode not in ("concurrency", "payload"):
            raise ValueError(
                f"ramp_mode must be 'concurrency' or 'payload', "
                f"got {self.ramp_mode!r}"
            )
        if not (0.0 < self.failure_threshold_pct <= 100.0):
            raise ValueError(
                f"failure_threshold_pct must be in (0, 100], "
                f"got {self.failure_threshold_pct}"
            )
        if self.max_wall_time_s is not None and self.max_wall_time_s <= 0:
            raise ValueError(
                f"max_wall_time_s must be > 0, got {self.max_wall_time_s}"
            )
        if self.max_escalation_concurrency < 1:
            raise ValueError(
                f"max_escalation_concurrency must be >= 1, "
                f"got {self.max_escalation_concurrency}"
            )
        if self.max_escalation_payload_dim < 1:
            raise ValueError(
                f"max_escalation_payload_dim must be >= 1, "
                f"got {self.max_escalation_payload_dim}"
            )

    def concurrency_levels(self) -> List[int]:
        """Compute the ordered concurrency levels for this config.

        Uses geometric spacing from *start_concurrency* to
        *max_concurrency*, capped at *ramp_steps* values.

        Returns
        -------
        List[int]
            Sorted, deduplicated list of concurrency levels.
        """
        levels = set()
        for i in range(self.ramp_steps):
            level = self.start_concurrency * (2 ** i)
            if level > self.max_concurrency:
                levels.add(self.max_concurrency)
                break
            levels.add(level)

        # Always include max_concurrency if we haven't hit it
        if max(levels) < self.max_concurrency and len(levels) < self.ramp_steps:
            levels.add(self.max_concurrency)

        return sorted(levels)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: Dict[str, Any] = {
            "start_concurrency": self.start_concurrency,
            "max_concurrency": self.max_concurrency,
            "ramp_steps": self.ramp_steps,
            "duration_per_step_s": self.duration_per_step_s,
            "payload_size": self.payload_size,
            "timeout_per_call_s": self.timeout_per_call_s,
            "ramp_mode": self.ramp_mode,
        }
        if self.run_until_failure:
            d["run_until_failure"] = True
            d["failure_escalation_mode"] = self.ramp_mode  # legacy compat
            d["failure_threshold_pct"] = self.failure_threshold_pct
            d["max_escalation_concurrency"] = self.max_escalation_concurrency
            d["max_escalation_payload_dim"] = self.max_escalation_payload_dim
        if self.max_wall_time_s is not None:
            d["max_wall_time_s"] = self.max_wall_time_s
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StressTestConfig":
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        StressTestConfig
        """
        return cls(
            start_concurrency=data.get(
                "start_concurrency", DEFAULT_START_CONCURRENCY
            ),
            max_concurrency=data.get(
                "max_concurrency", DEFAULT_MAX_CONCURRENCY
            ),
            ramp_steps=data.get("ramp_steps", DEFAULT_RAMP_STEPS),
            duration_per_step_s=data.get(
                "duration_per_step_s", DEFAULT_DURATION_PER_STEP_S
            ),
            payload_size=data.get("payload_size", DEFAULT_PAYLOAD_SIZE),
            timeout_per_call_s=data.get(
                "timeout_per_call_s", DEFAULT_TIMEOUT_PER_CALL_S
            ),
            run_until_failure=data.get(
                "run_until_failure", DEFAULT_RUN_UNTIL_FAILURE
            ),
            failure_escalation_mode=data.get(
                "failure_escalation_mode", DEFAULT_FAILURE_ESCALATION_MODE
            ),
            ramp_mode=data.get(
                "ramp_mode",
                data.get("failure_escalation_mode", DEFAULT_RAMP_MODE),
            ),
            failure_threshold_pct=data.get(
                "failure_threshold_pct", DEFAULT_FAILURE_THRESHOLD_PCT
            ),
            max_wall_time_s=data.get(
                "max_wall_time_s", DEFAULT_MAX_WALL_TIME_S
            ),
            max_escalation_concurrency=data.get(
                "max_escalation_concurrency", DEFAULT_MAX_ESCALATION_CONCURRENCY
            ),
            max_escalation_payload_dim=data.get(
                "max_escalation_payload_dim", DEFAULT_MAX_ESCALATION_PAYLOAD_DIM
            ),
        )


# ---------------------------------------------------------------------------
# StressTestEvent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StressTestEvent:
    """The outcome of a single call attempt during a stress ramp.

    Attributes
    ----------
    concurrency_level : int
        Number of concurrent workers active when this call was made.
    payload_shape : tuple
        Shape of the numpy array passed to the component.
    success : bool
        Whether the call completed without error or timeout.
    latency_s : float
        Wall-clock time for the call in seconds.  Set to
        ``timeout_per_call_s`` for timed-out calls.
    peak_rss_bytes : int
        Peak RSS memory delta during this call in bytes.  Zero when
        memory sampling is unavailable.
    error_type : str, optional
        The exception class name (e.g. ``"MemoryError"``,
        ``"TimeoutError"``).  ``None`` for successful calls.
    timestamp : str
        ISO 8601 UTC timestamp when this event was recorded.
    """

    concurrency_level: int
    payload_shape: Tuple[int, ...]
    success: bool
    latency_s: float
    peak_rss_bytes: int
    error_type: Optional[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: Dict[str, Any] = {
            "concurrency_level": self.concurrency_level,
            "payload_shape": list(self.payload_shape),
            "success": self.success,
            "latency_s": self.latency_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "timestamp": self.timestamp,
        }
        if self.error_type is not None:
            d["error_type"] = self.error_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StressTestEvent":
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        StressTestEvent
        """
        return cls(
            concurrency_level=data["concurrency_level"],
            payload_shape=tuple(data["payload_shape"]),
            success=data["success"],
            latency_s=data["latency_s"],
            peak_rss_bytes=data.get("peak_rss_bytes", 0),
            error_type=data.get("error_type"),
            timestamp=data["timestamp"],
        )


# ---------------------------------------------------------------------------
# FailurePoint
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FailurePoint:
    """A confirmed breaking point detected during a stress ramp.

    A ``FailurePoint`` is created when a concurrency level produces at
    least one failed call.  Multiple failure points may be recorded if
    different error types are encountered at different concurrency levels.

    Attributes
    ----------
    concurrency_level : int
        The concurrency level at which this failure was first observed.
    payload_shape : tuple
        Shape of the payload being processed when the failure occurred.
    error_type : str
        The exception class name (e.g. ``"MemoryError"``).
    error_message : str
        The string representation of the exception.
    memory_bytes_at_failure : int
        RSS memory in bytes at the time of failure.  Zero when sampling
        is unavailable.
    first_occurrence_at : str
        ISO 8601 UTC timestamp of the first failure event.
    """

    concurrency_level: int
    payload_shape: Tuple[int, ...]
    error_type: str
    error_message: str
    memory_bytes_at_failure: int
    first_occurrence_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "concurrency_level": self.concurrency_level,
            "payload_shape": list(self.payload_shape),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "memory_bytes_at_failure": self.memory_bytes_at_failure,
            "first_occurrence_at": self.first_occurrence_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailurePoint":
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        FailurePoint
        """
        return cls(
            concurrency_level=data["concurrency_level"],
            payload_shape=tuple(data["payload_shape"]),
            error_type=data["error_type"],
            error_message=data["error_message"],
            memory_bytes_at_failure=data.get("memory_bytes_at_failure", 0),
            first_occurrence_at=data["first_occurrence_at"],
        )


# ---------------------------------------------------------------------------
# StressTestSummary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StressTestSummary:
    """Headline findings from a completed stress test.

    Attributes
    ----------
    max_sustained_concurrency : int
        Highest concurrency level at which the component completed all
        calls with zero errors.
    saturation_concurrency : int, optional
        First concurrency level where errors appeared.  ``None`` if no
        failures were observed across the entire ramp.
    first_failure_mode : str, optional
        The ``error_type`` of the first failure observed.  ``None`` when
        no failures occurred.
    memory_high_water_mark_bytes : int
        Maximum RSS memory observed across all events.
    total_calls : int
        Total number of call attempts (successful and failed).
    failed_calls : int
        Number of calls that raised an exception or timed out.
    p99_latency_s : float
        99th-percentile latency across all *successful* calls in
        seconds.  Zero when no successful calls were made.
    """

    max_sustained_concurrency: int
    saturation_concurrency: Optional[int]
    first_failure_mode: Optional[str]
    memory_high_water_mark_bytes: int
    total_calls: int
    failed_calls: int
    p99_latency_s: float
    wall_time_budget_exceeded: bool = False
    saturation_payload_shape: Optional[Tuple[int, int]] = None
    ceiling_hit: bool = False
    """True when run-until-failure reached the configured ceiling without finding a failure."""
    largest_successful_payload_shape: Optional[Tuple[int, int]] = None
    """Largest payload that completed with < failure_threshold_pct errors (payload mode only)."""

    @classmethod
    def from_events(
        cls,
        events: List[StressTestEvent],
        failure_points: List[FailurePoint],
    ) -> "StressTestSummary":
        """Derive summary statistics from a completed event list.

        Parameters
        ----------
        events : List[StressTestEvent]
            All events recorded during the stress run.
        failure_points : List[FailurePoint]
            All detected failure points.

        Returns
        -------
        StressTestSummary
        """
        if not events:
            return cls(
                max_sustained_concurrency=0,
                saturation_concurrency=None,
                first_failure_mode=None,
                memory_high_water_mark_bytes=0,
                total_calls=0,
                failed_calls=0,
                p99_latency_s=0.0,
            )

        total = len(events)
        failed = sum(1 for e in events if not e.success)

        # Memory high-water mark
        hwm = max((e.peak_rss_bytes for e in events), default=0)

        # Successful latencies for p99
        successful_latencies = [e.latency_s for e in events if e.success]
        p99 = (
            float(np.percentile(successful_latencies, 99))
            if successful_latencies
            else 0.0
        )

        # Concurrency levels with zero failures
        levels = sorted({e.concurrency_level for e in events})
        failed_levels = {e.concurrency_level for e in events if not e.success}
        clean_levels = [l for l in levels if l not in failed_levels]
        max_sustained = max(clean_levels) if clean_levels else 0

        # Saturation and first failure mode from failure points
        saturation: Optional[int] = None
        first_mode: Optional[str] = None
        if failure_points:
            earliest = min(
                failure_points,
                key=lambda fp: fp.first_occurrence_at,
            )
            saturation = earliest.concurrency_level
            first_mode = earliest.error_type

        return cls(
            max_sustained_concurrency=max_sustained,
            saturation_concurrency=saturation,
            first_failure_mode=first_mode,
            memory_high_water_mark_bytes=hwm,
            total_calls=total,
            failed_calls=failed,
            p99_latency_s=p99,
            # wall_time_budget_exceeded and saturation_payload_shape are set
            # by the runner after from_events(); defaults are safe here.
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: Dict[str, Any] = {
            "max_sustained_concurrency": self.max_sustained_concurrency,
            "saturation_concurrency": self.saturation_concurrency,
            "first_failure_mode": self.first_failure_mode,
            "memory_high_water_mark_bytes": self.memory_high_water_mark_bytes,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "p99_latency_s": self.p99_latency_s,
        }
        if self.wall_time_budget_exceeded:
            d["wall_time_budget_exceeded"] = True
        if self.saturation_payload_shape is not None:
            d["saturation_payload_shape"] = list(self.saturation_payload_shape)
        if self.ceiling_hit:
            d["ceiling_hit"] = True
        if self.largest_successful_payload_shape is not None:
            d["largest_successful_payload_shape"] = list(self.largest_successful_payload_shape)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StressTestSummary":
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        StressTestSummary
        """
        sat_shape = data.get("saturation_payload_shape")
        lsp_shape = data.get("largest_successful_payload_shape")
        return cls(
            max_sustained_concurrency=data["max_sustained_concurrency"],
            saturation_concurrency=data.get("saturation_concurrency"),
            first_failure_mode=data.get("first_failure_mode"),
            memory_high_water_mark_bytes=data.get(
                "memory_high_water_mark_bytes", 0
            ),
            total_calls=data["total_calls"],
            failed_calls=data["failed_calls"],
            p99_latency_s=data.get("p99_latency_s", 0.0),
            wall_time_budget_exceeded=data.get("wall_time_budget_exceeded", False),
            saturation_payload_shape=(
                tuple(sat_shape) if sat_shape is not None else None
            ),
            ceiling_hit=data.get("ceiling_hit", False),
            largest_successful_payload_shape=(
                tuple(lsp_shape) if lsp_shape is not None else None
            ),
        )


# ---------------------------------------------------------------------------
# StressTestRecord
# ---------------------------------------------------------------------------
@dataclass
class StressTestRecord:
    """Complete record of a stress test run.

    The atomic unit of persistence for stress test results.  Every
    ``BaseStressTester.run()`` call produces one of these.

    Attributes
    ----------
    stress_test_id : str
        Unique identifier (UUID4).
    schema_version : int
        On-disk schema version.  Increment ``STRESS_SCHEMA_VERSION``
        when the schema changes.  ``from_dict()`` inspects this field
        to apply migration logic.
    component_name : str
        Name of the component or workflow under stress.
    component_version : str
        Version string for the component.
    grdl_version : str
        Version of the installed ``grdl`` package at run time.
    hardware : HardwareSnapshot, optional
        Hardware state at test time.  ``None`` when hardware detection
        is unavailable.
    config : StressTestConfig
        The configuration used for this run.
    events : List[StressTestEvent]
        All call events recorded during the ramp.
    failure_points : List[FailurePoint]
        Confirmed failure points detected during the ramp.
    summary : StressTestSummary
        Headline statistics derived from *events* and *failure_points*.
    related_benchmark_id : str, optional
        ``benchmark_id`` of a ``BenchmarkRecord`` run on this same
        component.  Provides a link to per-call timing statistics.
    created_at : str
        ISO 8601 UTC timestamp.
    tags : Dict[str, str]
        User-defined labels.
    """

    stress_test_id: str
    schema_version: int
    component_name: str
    component_version: str
    grdl_version: str
    hardware: Optional[HardwareSnapshot]
    config: StressTestConfig
    events: List[StressTestEvent]
    failure_points: List[FailurePoint]
    summary: StressTestSummary
    related_benchmark_id: Optional[str] = None
    created_at: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        component_name: str,
        component_version: str,
        hardware: Optional[HardwareSnapshot],
        config: StressTestConfig,
        events: List[StressTestEvent],
        failure_points: List[FailurePoint],
        related_benchmark_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "StressTestRecord":
        """Create a new record with auto-generated ID, timestamp, and summary.

        Parameters
        ----------
        component_name : str
            Name of the component under stress.
        component_version : str
            Version string.
        hardware : HardwareSnapshot, optional
            Hardware snapshot.
        config : StressTestConfig
            The run configuration.
        events : List[StressTestEvent]
            All recorded events.
        failure_points : List[FailurePoint]
            All detected failure points.
        related_benchmark_id : str, optional
            Link to an associated ``BenchmarkRecord``.
        tags : Dict[str, str], optional
            User labels.

        Returns
        -------
        StressTestRecord
        """
        try:
            import importlib.metadata as _meta
            grdl_version = _meta.version("grdl")
        except Exception:
            grdl_version = "unknown"

        summary = StressTestSummary.from_events(events, failure_points)

        return cls(
            stress_test_id=str(uuid.uuid4()),
            schema_version=STRESS_SCHEMA_VERSION,
            component_name=component_name,
            component_version=component_version,
            grdl_version=grdl_version,
            hardware=hardware,
            config=config,
            events=events,
            failure_points=failure_points,
            summary=summary,
            related_benchmark_id=related_benchmark_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: Dict[str, Any] = {
            "stress_test_id": self.stress_test_id,
            "schema_version": self.schema_version,
            "component_name": self.component_name,
            "component_version": self.component_version,
            "grdl_version": self.grdl_version,
            "hardware": (
                self.hardware.to_dict() if self.hardware is not None else None
            ),
            "config": self.config.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "failure_points": [fp.to_dict() for fp in self.failure_points],
            "summary": self.summary.to_dict(),
            "created_at": self.created_at,
            "tags": self.tags,
        }
        if self.related_benchmark_id is not None:
            d["related_benchmark_id"] = self.related_benchmark_id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StressTestRecord":
        """Deserialize from dictionary.

        Handles ``schema_version`` for forward-compatibility.

        Parameters
        ----------
        data : Dict[str, Any]

        Returns
        -------
        StressTestRecord
        """
        # Future schema migration hook
        version = data.get("schema_version", 1)
        if version > STRESS_SCHEMA_VERSION:
            import warnings
            warnings.warn(
                f"StressTestRecord schema_version {version} is newer than "
                f"the installed grdl-te supports ({STRESS_SCHEMA_VERSION}). "
                "Some fields may be missing.",
                UserWarning,
                stacklevel=2,
            )

        hardware_data = data.get("hardware")
        hardware = (
            HardwareSnapshot.from_dict(hardware_data)
            if hardware_data is not None
            else None
        )

        return cls(
            stress_test_id=data["stress_test_id"],
            schema_version=data.get("schema_version", 1),
            component_name=data["component_name"],
            component_version=data["component_version"],
            grdl_version=data.get("grdl_version", "unknown"),
            hardware=hardware,
            config=StressTestConfig.from_dict(data["config"]),
            events=[StressTestEvent.from_dict(e) for e in data.get("events", [])],
            failure_points=[
                FailurePoint.from_dict(fp)
                for fp in data.get("failure_points", [])
            ],
            summary=StressTestSummary.from_dict(data["summary"]),
            related_benchmark_id=data.get("related_benchmark_id"),
            created_at=data.get("created_at", ""),
            tags=data.get("tags", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string.

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
    def from_json(cls, text: str) -> "StressTestRecord":
        """Deserialize from a JSON string.

        Parameters
        ----------
        text : str
            JSON-encoded record.

        Returns
        -------
        StressTestRecord
        """
        return cls.from_dict(json.loads(text))
