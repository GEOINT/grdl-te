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
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

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
# Maximum consecutive failures before early abort
# ---------------------------------------------------------------------------
_MAX_CONSECUTIVE_FAILURES = 3


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

        levels = config.concurrency_levels()
        logger.info(
            "Starting stress ramp for '%s': levels=%s, "
            "duration_per_step_s=%.1f, payload=%s",
            self.component_name,
            levels,
            config.duration_per_step_s,
            payload_shape,
        )

        for concurrency in levels:
            events, fp = self._run_level(
                concurrency=concurrency,
                payload=payload,
                config=config,
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

        if self._store is not None:
            try:
                self._store.save(record)
            except Exception as exc:
                logger.warning("Failed to persist stress record: %s", exc)

        return record

    def _run_level(
        self,
        concurrency: int,
        payload: np.ndarray,
        config: StressTestConfig,
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

        deadline = time.monotonic() + config.duration_per_step_s

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
