# -*- coding: utf-8 -*-
"""
Stress Test ABCs — contracts for stress test executors.

Defines ``BaseStressTester``, which concrete implementations inherit from.
The abstract class owns the ramp execution loop, event collection, failure
detection, and record construction.  Subclasses only implement
``component_name`` and ``call_once()``.

This separation means the engine is fully decoupled from the CLI and from
pytest — a GUI can import ``BaseStressTester`` directly without pulling in
test infrastructure.

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
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import replace as _dataclass_replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# Internal
from grdl_te.benchmarking.models import HardwareSnapshot
from grdl_te.benchmarking.source import ARRAY_SIZES
from grdl_te.benchmarking.stress_models import (
    DEFAULT_PAYLOAD_SIZE,
    FailurePoint,
    StressTestConfig,
    StressTestEvent,
    StressTestRecord,
    StressTestSummary,
)

logger = logging.getLogger(__name__)

# Memory sampling — soft dependency on psutil
try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _sample_rss() -> int:
    """Return current process RSS memory in bytes.

    Returns 0 when psutil is not available.
    """
    if not _HAS_PSUTIL:
        return 0
    try:
        return _psutil.Process().memory_info().rss
    except Exception:
        return 0


def _resolve_payload_shape(payload_size: str) -> Tuple[int, int]:
    """Resolve a payload_size string to ``(rows, cols)``.

    Accepts size preset keys (``"small"``, ``"medium"``, ``"large"``) or
    an explicit ``"ROWSxCOLS"`` string (e.g. ``"512x512"``).

    Parameters
    ----------
    payload_size : str

    Returns
    -------
    Tuple[int, int]

    Raises
    ------
    ValueError
        If the string is not recognised.
    """
    if payload_size in ARRAY_SIZES:
        return ARRAY_SIZES[payload_size]
    if "x" in payload_size:
        parts = payload_size.lower().split("x")
        if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
            return (int(parts[0].strip()), int(parts[1].strip()))
    raise ValueError(
        f"Unrecognised payload_size: {payload_size!r}. "
        f"Use a preset ({list(ARRAY_SIZES)}) or 'ROWSxCOLS'."
    )


# ---------------------------------------------------------------------------
# Maximum consecutive failures before early abort within a single level
# ---------------------------------------------------------------------------
_MAX_CONSECUTIVE_FAILURES = 3


def _payload_escalation_sequence(
    base_shape: Tuple[int, int],
    max_dim: int,
) -> List[Tuple[int, int]]:
    """Generate shapes for payload escalation, doubling dimensions each step.

    Starts one doubling above *base_shape* and continues until *max_dim*
    is reached.

    Parameters
    ----------
    base_shape : Tuple[int, int]
        The *(rows, cols)* shape used during the normal ramp.  The first
        escalation step will be ``(rows*2, cols*2)``.
    max_dim : int
        Hard ceiling on the linear dimension.  Shapes will not exceed
        ``(max_dim, max_dim)``.

    Returns
    -------
    List[Tuple[int, int]]
        Ordered list of shapes to probe, smallest first.
    """
    rows, cols = base_shape
    shapes: List[Tuple[int, int]] = []
    while True:
        rows = min(rows * 2, max_dim)
        cols = min(cols * 2, max_dim)
        shapes.append((rows, cols))
        if rows >= max_dim and cols >= max_dim:
            break
    return shapes


def _fmt_shape(shape: Tuple[int, int]) -> str:
    return f"{shape[0]}\u00d7{shape[1]}"


class BaseStressTester(ABC):
    """Abstract base class for stress test executors.

    Subclasses must implement:

    - :attr:`component_name` — human-readable identifier
    - :meth:`call_once` — the single operation to stress

    The ramp loop, event collection, failure detection, and record
    construction are all handled by :meth:`run()` in this base class.
    Subclasses do not need to override ``run()``.

    Parameters
    ----------
    version : str
        Version string for the component under test.  Stored in the
        resulting ``StressTestRecord``.
    related_benchmark_id : str, optional
        ``benchmark_id`` of an associated ``BenchmarkRecord`` produced
        for the same component.  Links the two record types so a GUI can
        navigate from stress results to per-call timing statistics.
    store : object, optional
        If provided, the finished ``StressTestRecord`` is persisted
        automatically via ``store.save(record)`` after ``run()``
        completes.
    tags : Dict[str, str], optional
        User-defined labels attached to the produced record.
    """

    def __init__(
        self,
        version: str = "0.0.0",
        related_benchmark_id: Optional[str] = None,
        store: Optional[Any] = None,
        tags: Optional[dict] = None,
    ) -> None:
        self._version = version
        self._related_benchmark_id = related_benchmark_id
        self._store = store
        self._tags = tags or {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Human-readable name for the component under stress.

        Returns
        -------
        str
        """
        ...

    @abstractmethod
    def call_once(self, payload: np.ndarray) -> Any:
        """Execute a single call of the component with *payload* as input.

        This is the method the ramp loop calls concurrently.  It must be
        thread-safe (or tolerant of concurrent invocation) because the
        ramp executor submits multiple calls simultaneously.

        Parameters
        ----------
        payload : np.ndarray
            The input array for this call.

        Returns
        -------
        Any
            Return value is ignored by the ramp loop.

        Raises
        ------
        Any exception
            Raised exceptions are captured as failure events.  There is
            no need to catch them inside ``call_once``.
        """
        ...

    # ------------------------------------------------------------------
    # Ramp execution
    # ------------------------------------------------------------------

    def run(
        self, config: Optional[StressTestConfig] = None
    ) -> StressTestRecord:
        """Execute the stress ramp and return a completed ``StressTestRecord``.

        Parameters
        ----------
        config : StressTestConfig, optional
            Run configuration.  When ``None``, a ``StressTestConfig``
            with all defaults is used.

        Returns
        -------
        StressTestRecord
        """
        if config is None:
            config = StressTestConfig()

        hardware = HardwareSnapshot.capture()
        payload_shape = _resolve_payload_shape(config.payload_size)
        rng = np.random.default_rng(42)
        payload = rng.standard_normal(payload_shape).astype(np.float32)

        all_events: List[StressTestEvent] = []
        failure_points: List[FailurePoint] = []
        run_start = time.monotonic()

        # ── Choose ramp axis ──────────────────────────────────────────────
        # ramp_mode="concurrency" (default): run the concurrency ramp at a fixed
        # payload, then optionally escalate workers further if run_until_failure.
        # ramp_mode="payload": skip the concurrency ramp; ramp payload sizes
        # geometrically at a fixed concurrency level.  When run_until_failure is
        # True, stop at the failure threshold; otherwise run all sizes.
        payload_ramp_mode = config.ramp_mode == "payload"

        wall_time_budget_exceeded = False
        saturation_payload_shape = None
        ceiling_hit = False
        largest_successful_payload_shape = None

        if not payload_ramp_mode:
            # ── Concurrency ramp ──────────────────────────────────────────
            levels = config.concurrency_levels()
            logger.info(
                "Starting concurrency stress ramp for '%s': levels=%s, "
                "duration_per_step_s=%.1f, payload=%s",
                self.component_name,
                levels,
                config.duration_per_step_s,
                payload_shape,
            )

            for concurrency in levels:
                # Honour wall-time budget
                if config.max_wall_time_s is not None:
                    elapsed = time.monotonic() - run_start
                    if elapsed >= config.max_wall_time_s:
                        logger.warning(
                            "Wall-time budget (%.0fs) exceeded before completing "
                            "standard ramp at concurrency=%d.",
                            config.max_wall_time_s, concurrency,
                        )
                        break

                events, fp = self._run_level(
                    concurrency=concurrency,
                    payload=payload,
                    config=config,
                    wall_start=run_start,
                )
                all_events.extend(events)
                if fp is not None:
                    failure_points.append(fp)

                # Early abort on catastrophic failure (OOM, etc.)
                if fp is not None and fp.error_type in ("MemoryError", "OSError"):
                    logger.warning(
                        "Early abort at concurrency=%d: %s",
                        concurrency,
                        fp.error_type,
                    )
                    break

            # Optional concurrency escalation after the ramp
            if config.run_until_failure and not failure_points:
                (
                    extra_events,
                    extra_fps,
                    wall_time_budget_exceeded,
                    ceiling_hit,
                ) = self._escalate_concurrency(
                    config=config,
                    payload=payload,
                    levels_done=levels,
                    run_start=run_start,
                )
                all_events.extend(extra_events)
                failure_points.extend(extra_fps)

        else:
            # ── Payload ramp ──────────────────────────────────────────────
            # Skip the concurrency loop; probe payload sizes geometrically at
            # fixed concurrency (config.max_concurrency).
            # When run_until_failure=False, use threshold=100 so all sizes run.
            levels = []  # no concurrency levels for logging
            logger.info(
                "Starting payload stress ramp for '%s': "
                "starting shape=%s, concurrency=%d, "
                "run_until_failure=%s",
                self.component_name,
                payload_shape,
                config.max_concurrency,
                config.run_until_failure,
            )

            # Start one half-step below payload_size so the sequence begins AT
            # the configured payload_size (sequence doubles each step).
            half_shape = (max(1, payload_shape[0] // 2), max(1, payload_shape[1] // 2))

            if config.run_until_failure:
                _eff_config = config
            else:
                # 100% threshold means the run always completes all steps
                _eff_config = _dataclass_replace(config, failure_threshold_pct=100.0)

            (
                extra_events,
                extra_fps,
                saturation_payload_shape,
                largest_successful_payload_shape,
                wall_time_budget_exceeded,
                ceiling_hit,
            ) = self._escalate_payload(
                config=_eff_config,
                base_shape=half_shape,
                run_start=run_start,
            )
            all_events.extend(extra_events)
            failure_points.extend(extra_fps)

        record = StressTestRecord.create(
            component_name=self.component_name,
            component_version=self._version,
            hardware=hardware,
            config=config,
            events=all_events,
            failure_points=failure_points,
            related_benchmark_id=self._related_benchmark_id,
            tags=self._tags,
        )

        # Patch extra escalation-phase metadata into the frozen summary
        if wall_time_budget_exceeded or saturation_payload_shape is not None \
                or ceiling_hit or largest_successful_payload_shape is not None:
            record.summary = _dataclass_replace(
                record.summary,
                wall_time_budget_exceeded=wall_time_budget_exceeded,
                saturation_payload_shape=saturation_payload_shape,
                ceiling_hit=ceiling_hit,
                largest_successful_payload_shape=largest_successful_payload_shape,
            )

        if self._store is not None:
            try:
                self._store.save(record)
            except Exception as exc:
                logger.warning("Failed to persist stress record: %s", exc)

        return record

    # ------------------------------------------------------------------
    # Escalation helpers (run_until_failure phases)
    # ------------------------------------------------------------------

    def _escalate_concurrency(
        self,
        config: StressTestConfig,
        payload: np.ndarray,
        levels_done: List[int],
        run_start: float,
    ) -> Tuple[List[StressTestEvent], List[FailurePoint], bool, bool]:
        """Continue doubling concurrency past ``config.max_concurrency``.

        Returns
        -------
        extra_events, extra_fps, wall_time_budget_exceeded, ceiling_hit
        """
        extra_events: List[StressTestEvent] = []
        extra_fps: List[FailurePoint] = []
        wall_exceeded = False
        ceiling_hit = False

        # Build escalation levels: double from last level done
        concurrency = max(levels_done) if levels_done else config.max_concurrency
        ceiling = config.max_escalation_concurrency

        logger.info(
            "'%s' run_until_failure=concurrency: escalating from %d to ceiling=%d",
            self.component_name, concurrency, ceiling,
        )

        while True:
            concurrency = min(concurrency * 2, ceiling)

            if config.max_wall_time_s is not None:
                elapsed = time.monotonic() - run_start
                if elapsed >= config.max_wall_time_s:
                    logger.warning(
                        "'%s' wall-time budget (%.0fs) exceeded during "
                        "concurrency escalation at level=%d.",
                        self.component_name, config.max_wall_time_s, concurrency,
                    )
                    wall_exceeded = True
                    break

            logger.info("  escalating concurrency → %d workers", concurrency)
            events, fp = self._run_level(
                concurrency=concurrency,
                payload=payload,
                config=config,
                wall_start=run_start,
            )
            extra_events.extend(events)

            if fp is not None:
                extra_fps.append(fp)
                # Check if failure threshold reached
                total = len(events)
                failed = sum(1 for e in events if not e.success)
                pct = failed / total * 100.0 if total > 0 else 0.0
                if pct >= config.failure_threshold_pct:
                    logger.info(
                        "  saturation reached at concurrency=%d (%.1f%% failures)",
                        concurrency, pct,
                    )
                    break

            if concurrency >= ceiling:
                logger.info(
                    "  concurrency ceiling (%d workers) reached without saturation — "
                    "this is a configured hard limit, not a machine limit.",
                    ceiling,
                )
                ceiling_hit = True
                break

        return extra_events, extra_fps, wall_exceeded, ceiling_hit

    def _escalate_payload(
        self,
        config: StressTestConfig,
        base_shape: Tuple[int, int],
        run_start: float,
    ) -> Tuple[List[StressTestEvent], List[FailurePoint], Optional[Tuple[int, int]], Optional[Tuple[int, int]], bool, bool]:
        """Fix concurrency at ``max_concurrency`` and escalate payload size.

        Returns
        -------
        extra_events, extra_fps, saturation_payload_shape,
        largest_successful_payload_shape, wall_time_budget_exceeded, ceiling_hit
        """
        extra_events: List[StressTestEvent] = []
        extra_fps: List[FailurePoint] = []
        saturation_shape: Optional[Tuple[int, int]] = None
        largest_ok_shape: Optional[Tuple[int, int]] = None
        wall_exceeded = False
        ceiling_hit = False

        concurrency = config.max_concurrency
        max_dim = config.max_escalation_payload_dim
        shapes = _payload_escalation_sequence(base_shape, max_dim)

        logger.info(
            "'%s' run_until_failure=payload: will probe %d shapes up to %s "
            "at concurrency=%d",
            self.component_name,
            len(shapes),
            _fmt_shape(shapes[-1]) if shapes else "?",
            concurrency,
        )

        rng = np.random.default_rng(99)

        for shape in shapes:
            if config.max_wall_time_s is not None:
                elapsed = time.monotonic() - run_start
                if elapsed >= config.max_wall_time_s:
                    logger.warning(
                        "'%s' wall-time budget (%.0fs) exceeded during "
                        "payload escalation at shape=%s.",
                        self.component_name, config.max_wall_time_s, _fmt_shape(shape),
                    )
                    wall_exceeded = True
                    break

            megapixels = shape[0] * shape[1] / 1_000_000
            mb = shape[0] * shape[1] * 4 / 1_048_576  # float32 = 4 bytes
            logger.info(
                "  probing payload %s (%.1f MP, %.0f MB)",
                _fmt_shape(shape), megapixels, mb,
            )
            try:
                escalated_payload = rng.standard_normal(shape).astype(np.float32)
            except MemoryError:
                logger.warning(
                    "  OOM allocating payload %s — stopping escalation.", _fmt_shape(shape)
                )
                break

            events, fp = self._run_level(
                concurrency=concurrency,
                payload=escalated_payload,
                config=config,
                wall_start=run_start,
            )
            extra_events.extend(events)

            if fp is not None:
                extra_fps.append(fp)
                total = len(events)
                failed = sum(1 for e in events if not e.success)
                pct = failed / total * 100.0 if total > 0 else 0.0
                if pct >= config.failure_threshold_pct:
                    saturation_shape = shape
                    logger.info(
                        "  payload saturation at %s (%.1f%% failures — threshold %.1f%%)",
                        _fmt_shape(shape), pct, config.failure_threshold_pct,
                    )
                    break
            else:
                largest_ok_shape = shape
                logger.info("  %s: OK (0 failures)", _fmt_shape(shape))

            if shape[0] >= max_dim and shape[1] >= max_dim:
                logger.info(
                    "  payload ceiling (%s, %d px) reached without saturation — "
                    "this is a configured hard limit, not a machine limit.",
                    _fmt_shape(shape), max_dim,
                )
                ceiling_hit = True
                break

        return extra_events, extra_fps, saturation_shape, largest_ok_shape, wall_exceeded, ceiling_hit

    # ------------------------------------------------------------------
    # Level runner
    # ------------------------------------------------------------------

    def _run_level(
        self,
        concurrency: int,
        payload: np.ndarray,
        config: StressTestConfig,
        wall_start: Optional[float] = None,
    ) -> Tuple[List[StressTestEvent], Optional[FailurePoint]]:
        """Run one concurrency level for ``duration_per_step_s`` seconds.

        Submits batches of *concurrency* concurrent calls repeatedly until
        the configured duration is exhausted.

        Parameters
        ----------
        concurrency : int
            Number of simultaneous workers.
        payload : np.ndarray
            Read-only input array shared across workers.
        config : StressTestConfig
            Run configuration.
        wall_start : float, optional
            ``time.monotonic()`` value from the start of the full run.
            When provided and ``config.max_wall_time_s`` is set, the level
            deadline is clamped so the run never exceeds the wall budget.

        Returns
        -------
        events : List[StressTestEvent]
            All events recorded at this level.
        failure_point : FailurePoint or None
            The first detected failure at this level, or ``None`` when all
            calls succeeded.
        """
        events: List[StressTestEvent] = []
        failure_point: Optional[FailurePoint] = None
        consecutive_failures = 0

        step_deadline = time.monotonic() + config.duration_per_step_s
        if wall_start is not None and config.max_wall_time_s is not None:
            wall_deadline = wall_start + config.max_wall_time_s
            deadline = min(step_deadline, wall_deadline)
        else:
            deadline = step_deadline

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while time.monotonic() < deadline:
                # Don't submit more work if consecutive failures are piling up
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    break

                batch_start = time.monotonic()
                futures: List[Tuple[Future, float, int]] = []

                for _ in range(concurrency):
                    rss_before = _sample_rss()
                    t_submit = time.monotonic()
                    fut = executor.submit(self.call_once, payload)
                    futures.append((fut, t_submit, rss_before))

                for fut, t_submit, rss_before in futures:
                    ts = datetime.now(timezone.utc).isoformat()
                    try:
                        fut.result(timeout=config.timeout_per_call_s)
                        rss_after = _sample_rss()
                        latency = time.monotonic() - t_submit
                        events.append(
                            StressTestEvent(
                                concurrency_level=concurrency,
                                payload_shape=tuple(payload.shape),
                                success=True,
                                latency_s=latency,
                                peak_rss_bytes=max(0, rss_after - rss_before),
                                error_type=None,
                                timestamp=ts,
                            )
                        )
                        consecutive_failures = 0
                    except FuturesTimeout:
                        events.append(
                            StressTestEvent(
                                concurrency_level=concurrency,
                                payload_shape=tuple(payload.shape),
                                success=False,
                                latency_s=config.timeout_per_call_s,
                                peak_rss_bytes=0,
                                error_type="TimeoutError",
                                timestamp=ts,
                            )
                        )
                        consecutive_failures += 1
                        if failure_point is None:
                            failure_point = FailurePoint(
                                concurrency_level=concurrency,
                                payload_shape=tuple(payload.shape),
                                error_type="TimeoutError",
                                error_message="Call exceeded timeout",
                                memory_bytes_at_failure=_sample_rss(),
                                first_occurrence_at=ts,
                            )
                    except Exception as exc:
                        error_type = type(exc).__name__
                        rss_at = _sample_rss()
                        events.append(
                            StressTestEvent(
                                concurrency_level=concurrency,
                                payload_shape=tuple(payload.shape),
                                success=False,
                                latency_s=time.monotonic() - t_submit,
                                peak_rss_bytes=0,
                                error_type=error_type,
                                timestamp=ts,
                            )
                        )
                        consecutive_failures += 1
                        if failure_point is None:
                            failure_point = FailurePoint(
                                concurrency_level=concurrency,
                                payload_shape=tuple(payload.shape),
                                error_type=error_type,
                                error_message=str(exc),
                                memory_bytes_at_failure=rss_at,
                                first_occurrence_at=ts,
                            )

                # If this batch took less than the step duration, continue.
                # If we've exhausted the deadline, exit.
                if time.monotonic() >= deadline:
                    break

        return events, failure_point
