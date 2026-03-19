# -*- coding: utf-8 -*-
"""
Forensic Trace Reader — ingest pre-recorded grdl-runtime execution traces.

Provides :class:`ForensicExecutionTrace`, a thin dataclass that wraps one
``WorkflowMetrics.to_dict()`` payload with source provenance metadata, and
:class:`ForensicTraceReader`, which loads traces from four different source
types (JSON file, directory of JSON files, SQLite history DB, or in-memory
dicts).

The loaded traces are passed directly to :class:`PassiveBenchmarkRunner`
to produce a ``BenchmarkRecord`` without re-executing any workflow.

Dependencies
------------
grdl-runtime (optional — only required for ``ForensicTraceReader.from_history_db``)

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
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_metrics_dict(d: Dict[str, Any]) -> None:
    """Raise ValueError if *d* is missing required WorkflowMetrics fields.

    Parameters
    ----------
    d : Dict[str, Any]

    Raises
    ------
    ValueError
        If any required field is absent or ``step_metrics`` is not a list.
    """
    required = {"total_wall_time_s", "total_cpu_time_s", "peak_rss_bytes", "step_metrics"}
    missing = required - d.keys()
    if missing:
        raise ValueError(
            f"Not a valid WorkflowMetrics dict. Missing fields: {missing}"
        )
    if not isinstance(d["step_metrics"], list):
        raise ValueError("step_metrics must be a list")


def _step_metrics_from_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a step_metrics entry to StepMetrics constructor kwargs.

    Applies defaults for optional fields so that traces from older
    grdl-runtime versions still parse correctly.

    Parameters
    ----------
    d : Dict[str, Any]
        One entry from ``WorkflowMetrics.to_dict()["step_metrics"]``.

    Returns
    -------
    Dict[str, Any]
        Keyword arguments suitable for ``StepMetrics(**kwargs)``.
    """
    return {
        "step_index": int(d["step_index"]),
        "processor_name": d["processor_name"],
        "wall_time_s": float(d["wall_time_s"]),
        "cpu_time_s": float(d["cpu_time_s"]),
        "peak_rss_bytes": int(d["peak_rss_bytes"]),
        "gpu_used": bool(d.get("gpu_used", False)),
        "status": d.get("status", "success"),
        "error_message": d.get("error_message"),
        "step_id": d.get("step_id"),
        "gpu_memory_bytes": d.get("gpu_memory_bytes"),
        "global_pass_duration": d.get("global_pass_duration"),
        "global_pass_memory": d.get("global_pass_memory"),
        "concurrent": bool(d.get("concurrent", False)),
        "input_shape": tuple(d["input_shape"]) if "input_shape" in d else None,
        "input_dtype": d.get("input_dtype"),
    }


# ---------------------------------------------------------------------------
# ForensicExecutionTrace
# ---------------------------------------------------------------------------

@dataclass
class ForensicExecutionTrace:
    """A single execution trace for forensic benchmarking.

    Wraps one ``WorkflowMetrics.to_dict()`` payload with provenance
    metadata.  Produced by :class:`ForensicTraceReader` and consumed
    by :class:`PassiveBenchmarkRunner`.

    Attributes
    ----------
    workflow_name : str
        Name of the workflow that produced this trace.
    workflow_version : str
        Version of the workflow.
    run_id : str
        Unique run identifier (UUID4) from the original execution.
    metrics_dict : Dict[str, Any]
        ``WorkflowMetrics.to_dict()`` output from the original run.
    source_type : str
        Origin of this trace: ``"json_file"``, ``"json_dir"``,
        ``"history_db"``, or ``"memory"``.
    source_path : str, optional
        File path or DB path for file-backed sources.  ``None`` for
        in-memory traces.
    started_at : str
        ISO 8601 UTC timestamp when the original execution started.
    status : str
        Execution status: ``"success"``, ``"failed"``, or ``"cancelled"``.
    """

    workflow_name: str
    workflow_version: str
    run_id: str
    metrics_dict: Dict[str, Any]
    source_type: str
    source_path: Optional[str]
    started_at: str
    status: str

    @classmethod
    def from_metrics_dict(
        cls,
        d: Dict[str, Any],
        *,
        source_type: str = "memory",
        source_path: Optional[str] = None,
    ) -> "ForensicExecutionTrace":
        """Construct from a ``WorkflowMetrics.to_dict()`` dictionary.

        Parameters
        ----------
        d : Dict[str, Any]
            ``WorkflowMetrics.to_dict()`` output.
        source_type : str
            Origin label.  Defaults to ``"memory"``.
        source_path : str, optional
            File or DB path for provenance.

        Returns
        -------
        ForensicExecutionTrace
        """
        wf_id = d.get("workflow_id", "unknown:0.0.0")
        parts = wf_id.split(":", 1)
        name = d.get("workflow_name") or (parts[0] if parts else "unknown")
        version = d.get("workflow_version") or (
            parts[1] if len(parts) > 1 else "0.0.0"
        )
        return cls(
            workflow_name=name,
            workflow_version=version,
            run_id=d.get("run_id", ""),
            metrics_dict=d,
            source_type=source_type,
            source_path=source_path,
            started_at=d.get("started_at", ""),
            status=d.get("status", "success"),
        )

    def to_step_metrics(self) -> List[Any]:
        """Reconstitute StepMetrics objects from ``metrics_dict["step_metrics"]``.

        Returns
        -------
        List[StepMetrics]
            One object per step entry in the original trace.

        Raises
        ------
        ImportError
            If grdl-runtime is not installed.
        """
        try:
            from grdl_rt.execution.metrics import StepMetrics
        except ImportError:
            raise ImportError(
                "to_step_metrics() requires grdl-runtime. "
                "Install it with: pip install grdl-runtime"
            )
        return [
            StepMetrics(**_step_metrics_from_dict(sm))
            for sm in self.metrics_dict.get("step_metrics", [])
        ]

    def to_hardware_snapshot(self) -> Optional["HardwareSnapshot"]:
        """Reconstruct a HardwareSnapshot from embedded hardware data.

        Returns ``None`` if the trace predates the ``hardware`` field
        (grdl-runtime versions before forensic benchmarking support).

        Returns
        -------
        HardwareSnapshot or None
        """
        hw = self.metrics_dict.get("hardware")
        if hw is None:
            return None
        try:
            from grdl_te.benchmarking.models import HardwareSnapshot
            return HardwareSnapshot.from_dict(hw)
        except Exception as exc:
            logger.debug("Failed to reconstruct HardwareSnapshot from trace: %s", exc)
            return None


# ---------------------------------------------------------------------------
# ForensicTraceReader
# ---------------------------------------------------------------------------

class ForensicTraceReader:
    """Load execution traces from multiple source formats.

    All loader methods return ``List[ForensicExecutionTrace]``, normalised
    to the same structure regardless of source.  The resulting list is
    passed directly to :class:`PassiveBenchmarkRunner`.

    Examples
    --------
    Load from a single JSON file::

        traces = ForensicTraceReader.from_json_file("run_output.json")

    Load multiple runs from a directory::

        traces = ForensicTraceReader.from_json_directory(
            "/exports/sar_runs/", status_filter="success"
        )

    Load from the grdl-runtime execution history database::

        traces = ForensicTraceReader.from_history_db(
            ["abc-123", "def-456"]
        )

    Wrap in-memory dicts::

        traces = ForensicTraceReader.from_memory(active_record.raw_metrics)
    """

    @staticmethod
    def from_json_file(path: Union[str, Path]) -> List[ForensicExecutionTrace]:
        """Load a single ``WorkflowMetrics.to_json()`` file.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file produced by ``WorkflowMetrics.to_json()``.

        Returns
        -------
        List[ForensicExecutionTrace]
            Always length 1.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is not a valid WorkflowMetrics dict.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        _validate_metrics_dict(d)
        return [
            ForensicExecutionTrace.from_metrics_dict(
                d,
                source_type="json_file",
                source_path=str(p),
            )
        ]

    @staticmethod
    def from_json_directory(
        directory: Union[str, Path],
        *,
        glob_pattern: str = "*.json",
        status_filter: Optional[str] = "success",
    ) -> List[ForensicExecutionTrace]:
        """Load all matching JSON files from a directory.

        Files that are not valid WorkflowMetrics dicts are skipped with
        a warning rather than raising.

        Parameters
        ----------
        directory : str or Path
            Directory containing JSON trace files.
        glob_pattern : str
            File matching pattern.  Default ``"*.json"``.
        status_filter : str, optional
            Only include traces whose ``status`` matches this value.
            Pass ``None`` to include all.  Default ``"success"``.

        Returns
        -------
        List[ForensicExecutionTrace]
            Sorted by ``started_at`` ascending (oldest first).

        Raises
        ------
        FileNotFoundError
            If *directory* does not exist.
        """
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")

        traces = []
        for p in sorted(d.glob(glob_pattern)):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                _validate_metrics_dict(raw)
                trace = ForensicExecutionTrace.from_metrics_dict(
                    raw,
                    source_type="json_dir",
                    source_path=str(p),
                )
                if status_filter is None or trace.status == status_filter:
                    traces.append(trace)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning(
                    "Skipping malformed trace file %s: %s", p, exc
                )
                continue

        traces.sort(key=lambda t: t.started_at)
        return traces

    @staticmethod
    def from_history_db(
        run_ids: List[str],
        *,
        db_path: Optional[Path] = None,
    ) -> List[ForensicExecutionTrace]:
        """Load traces from the grdl-runtime ExecutionHistoryDB.

        Parameters
        ----------
        run_ids : List[str]
            UUIDs of runs to load.  Order is preserved in the result.
        db_path : Path, optional
            Override the default ``~/.grdl_rt/history.db`` location.

        Returns
        -------
        List[ForensicExecutionTrace]

        Raises
        ------
        ImportError
            If grdl-runtime is not installed.
        KeyError
            If any *run_id* is not found in the database.
        ValueError
            If a matching record has no ``metrics_json`` (run never
            completed).
        """
        try:
            from grdl_rt.execution.history import ExecutionHistoryDB
        except ImportError:
            raise ImportError(
                "from_history_db() requires grdl-runtime. "
                "Install it with: pip install grdl-runtime"
            )

        traces: List[ForensicExecutionTrace] = []

        with ExecutionHistoryDB(db_path=db_path) as db:
            db_str = str(db._db_path)
            for run_id in run_ids:
                rec = db.get_execution(run_id)
                if rec is None:
                    raise KeyError(
                        f"run_id not found in history DB: {run_id!r}"
                    )
                if rec.metrics_json is None:
                    raise ValueError(
                        f"run_id {run_id!r} has no metrics_json "
                        f"(status={rec.status!r}). "
                        "Did the run complete successfully?"
                    )
                raw = json.loads(rec.metrics_json)
                _validate_metrics_dict(raw)
                traces.append(
                    ForensicExecutionTrace.from_metrics_dict(
                        raw,
                        source_type="history_db",
                        source_path=db_str,
                    )
                )

        return traces

    @staticmethod
    def from_memory(
        metrics_dicts: List[Dict[str, Any]],
    ) -> List[ForensicExecutionTrace]:
        """Wrap in-memory ``WorkflowMetrics.to_dict()`` outputs.

        This is the primary entry point for the active-vs-passive parity
        test: ``active_record.raw_metrics`` is already a
        ``List[Dict[str, Any]]`` and can be passed directly here.

        Parameters
        ----------
        metrics_dicts : List[Dict[str, Any]]
            Each element must be the output of ``WorkflowMetrics.to_dict()``.

        Returns
        -------
        List[ForensicExecutionTrace]

        Raises
        ------
        ValueError
            If any dict is missing required WorkflowMetrics fields.
        """
        traces = []
        for d in metrics_dicts:
            _validate_metrics_dict(d)
            traces.append(
                ForensicExecutionTrace.from_metrics_dict(
                    d, source_type="memory"
                )
            )
        return traces
