# -*- coding: utf-8 -*-
"""
Unified GRDL Store — single file-based store for benchmark and stress records.

``GRDLStore`` manages both ``BenchmarkRecord`` and ``StressTestRecord``
objects under a single root directory, routing each record type to its
own subdirectory and selecting the correct report format automatically.

Storage layout::

    <base_dir>/
        index.json                      ← benchmark index
        records/
            <benchmark_id>.json         ← BenchmarkRecord files
        stress/
            index.json                  ← stress test index
            records/
                <stress_test_id>.json   ← StressTestRecord files

This layout is backward-compatible with ``JSONBenchmarkStore`` and
``JSONStressTestStore`` so existing on-disk data can be opened directly
with ``GRDLStore``.

Author
------
Ava Courtney <courtney-ava@zai.com>

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-05-13

Modified
--------
2026-05-13
"""

# Standard library
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Internal
from grdl_te.benchmarking.models import BenchmarkRecord
from grdl_te.benchmarking.stress_models import StressTestRecord


class GRDLStore:
    """Unified store for benchmark and stress test records.

    A single object that replaces the separate ``JSONBenchmarkStore`` and
    ``JSONStressTestStore`` classes.  Pass a ``BenchmarkRecord`` or a
    ``StressTestRecord`` to :meth:`save` — the store routes it to the
    correct subdirectory and updates the appropriate index.

    Parameters
    ----------
    base_dir : Path or str, optional
        Root directory for storage.  Defaults to ``<cwd>/.benchmarks/``.

    Examples
    --------
    >>> from grdl_te.benchmarking import GRDLStore
    >>> store = GRDLStore(base_dir=".benchmarks")

    Save a benchmark record:

    >>> store.save(benchmark_record)

    Save a stress test record:

    >>> store.save(stress_record)

    Load by type:

    >>> bench = store.load_benchmark(benchmark_id)
    >>> stress = store.load_stress(stress_id)
    """

    def __init__(self, base_dir: Optional[Union[Path, str]] = None) -> None:
        if base_dir is None:
            base_dir = Path.cwd() / ".benchmarks"
        self._base_dir = Path(base_dir)

        # Benchmark subtree
        self._bench_records_dir = self._base_dir / "records"
        self._bench_index_path = self._base_dir / "index.json"

        # Stress subtree
        self._stress_dir = self._base_dir / "stress"
        self._stress_records_dir = self._stress_dir / "records"
        self._stress_index_path = self._stress_dir / "index.json"

    # ------------------------------------------------------------------
    # Directory / index helpers
    # ------------------------------------------------------------------

    def _ensure_bench_dirs(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._bench_records_dir.mkdir(parents=True, exist_ok=True)
        if not self._bench_index_path.exists():
            self._write_index(self._bench_index_path, [])

    def _ensure_stress_dirs(self) -> None:
        self._stress_records_dir.mkdir(parents=True, exist_ok=True)
        if not self._stress_index_path.exists():
            self._write_index(self._stress_index_path, [])

    def _read_index(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def _write_index(self, path: Path, entries: List[Dict[str, Any]]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Unified save / load
    # ------------------------------------------------------------------

    def save(self, record: Union[BenchmarkRecord, StressTestRecord]) -> str:
        """Persist a record of either type.

        Dispatches on the record's type and returns the record's primary
        ID string.

        Parameters
        ----------
        record : BenchmarkRecord or StressTestRecord

        Returns
        -------
        str
            The ``benchmark_id`` or ``stress_test_id`` of the saved record.

        Raises
        ------
        TypeError
            If *record* is neither a ``BenchmarkRecord`` nor a
            ``StressTestRecord``.
        """
        if isinstance(record, StressTestRecord):
            return self._save_stress(record)
        if isinstance(record, BenchmarkRecord):
            return self._save_benchmark(record)
        raise TypeError(
            f"save() expects BenchmarkRecord or StressTestRecord, "
            f"got {type(record).__name__}"
        )

    # ------------------------------------------------------------------
    # Benchmark record methods
    # ------------------------------------------------------------------

    def _save_benchmark(self, record: BenchmarkRecord) -> str:
        self._ensure_bench_dirs()
        path = self._bench_records_dir / f"{record.benchmark_id}.json"
        path.write_text(record.to_json(), encoding="utf-8")

        index = self._read_index(self._bench_index_path)
        topo_str = None
        if record.topology is not None:
            topo_str = record.topology.topology.value

        index.append({
            "benchmark_id": record.benchmark_id,
            "benchmark_type": record.benchmark_type,
            "workflow_name": record.workflow_name,
            "workflow_version": record.workflow_version,
            "iterations": record.iterations,
            "created_at": record.created_at,
            "topology": topo_str,
        })
        self._write_index(self._bench_index_path, index)
        return record.benchmark_id

    def load_benchmark(self, benchmark_id: str) -> BenchmarkRecord:
        """Load a benchmark record by ID.

        Parameters
        ----------
        benchmark_id : str

        Returns
        -------
        BenchmarkRecord

        Raises
        ------
        KeyError
            If no record exists for the given ID.
        """
        path = self._bench_records_dir / f"{benchmark_id}.json"
        if not path.exists():
            raise KeyError(f"No benchmark record found with ID: {benchmark_id}")
        return BenchmarkRecord.from_json(path.read_text(encoding="utf-8"))

    def list_benchmarks(
        self,
        workflow_name: Optional[str] = None,
        benchmark_type: Optional[str] = None,
        topology: Optional[str] = None,
        limit: int = 50,
    ) -> List[BenchmarkRecord]:
        """List benchmark records, optionally filtered.

        Parameters
        ----------
        workflow_name : str, optional
        benchmark_type : str, optional
        topology : str, optional
        limit : int

        Returns
        -------
        List[BenchmarkRecord]
            Newest first.
        """
        index = self._read_index(self._bench_index_path)
        index.sort(key=lambda e: e.get("created_at", ""), reverse=True)

        filtered: List[Dict[str, Any]] = []
        for entry in index:
            if workflow_name and entry.get("workflow_name") != workflow_name:
                continue
            if benchmark_type and entry.get("benchmark_type") != benchmark_type:
                continue
            if topology and entry.get("topology") != topology:
                continue
            filtered.append(entry)
            if len(filtered) >= limit:
                break

        records: List[BenchmarkRecord] = []
        for entry in filtered:
            try:
                records.append(self.load_benchmark(entry["benchmark_id"]))
            except KeyError:
                continue
        return records

    def delete_benchmark(self, benchmark_id: str) -> None:
        """Delete a benchmark record and remove it from the index.

        Parameters
        ----------
        benchmark_id : str

        Raises
        ------
        KeyError
            If no record exists with the given ID.
        """
        path = self._bench_records_dir / f"{benchmark_id}.json"
        if not path.exists():
            raise KeyError(f"No benchmark record found with ID: {benchmark_id}")
        path.unlink()

        index = self._read_index(self._bench_index_path)
        index = [e for e in index if e.get("benchmark_id") != benchmark_id]
        self._write_index(self._bench_index_path, index)

    def rebuild_benchmark_index(self) -> int:
        """Rebuild benchmark index.json from files on disk.

        Returns
        -------
        int
            Number of records indexed.
        """
        self._ensure_bench_dirs()
        entries: List[Dict[str, Any]] = []
        for p in sorted(self._bench_records_dir.glob("*.json")):
            try:
                record = BenchmarkRecord.from_json(p.read_text(encoding="utf-8"))
                topo_str = None
                if record.topology is not None:
                    topo_str = record.topology.topology.value
                entries.append({
                    "benchmark_id": record.benchmark_id,
                    "benchmark_type": record.benchmark_type,
                    "workflow_name": record.workflow_name,
                    "workflow_version": record.workflow_version,
                    "iterations": record.iterations,
                    "created_at": record.created_at,
                    "topology": topo_str,
                })
            except (json.JSONDecodeError, KeyError):
                continue
        self._write_index(self._bench_index_path, entries)
        return len(entries)

    # ------------------------------------------------------------------
    # Stress test record methods
    # ------------------------------------------------------------------

    def _save_stress(self, record: StressTestRecord) -> str:
        self._ensure_stress_dirs()
        path = self._stress_records_dir / f"{record.stress_test_id}.json"
        path.write_text(record.to_json(), encoding="utf-8")

        index = self._read_index(self._stress_index_path)
        index.append({
            "stress_test_id": record.stress_test_id,
            "component_name": record.component_name,
            "component_version": record.component_version,
            "grdl_version": record.grdl_version,
            "created_at": record.created_at,
            "max_sustained_concurrency": (
                record.summary.max_sustained_concurrency
            ),
            "first_failure_mode": record.summary.first_failure_mode,
            "schema_version": record.schema_version,
        })
        self._write_index(self._stress_index_path, index)
        return record.stress_test_id

    def load_stress(self, stress_test_id: str) -> StressTestRecord:
        """Load a stress test record by ID.

        Parameters
        ----------
        stress_test_id : str

        Returns
        -------
        StressTestRecord

        Raises
        ------
        KeyError
            If no record exists for the given ID.
        """
        path = self._stress_records_dir / f"{stress_test_id}.json"
        if not path.exists():
            raise KeyError(
                f"No stress test record found with ID: {stress_test_id}"
            )
        return StressTestRecord.from_json(path.read_text(encoding="utf-8"))

    def list_stress_tests(
        self,
        component_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[StressTestRecord]:
        """List stress test records, optionally filtered.

        Parameters
        ----------
        component_name : str, optional
            Filter by exact component name.
        limit : int

        Returns
        -------
        List[StressTestRecord]
            Newest first.
        """
        index = self._read_index(self._stress_index_path)
        if component_name is not None:
            index = [e for e in index if e.get("component_name") == component_name]

        index = sorted(
            index, key=lambda e: e.get("created_at", ""), reverse=True
        )[:limit]

        records: List[StressTestRecord] = []
        for entry in index:
            sid = entry.get("stress_test_id", "")
            try:
                records.append(self.load_stress(sid))
            except (KeyError, Exception):
                continue
        return records

    def delete_stress(self, stress_test_id: str) -> None:
        """Delete a stress test record and remove it from the index.

        Parameters
        ----------
        stress_test_id : str

        Raises
        ------
        KeyError
            If no record exists with the given ID.
        """
        path = self._stress_records_dir / f"{stress_test_id}.json"
        if not path.exists():
            raise KeyError(
                f"No stress test record found with ID: {stress_test_id}"
            )
        path.unlink()

        index = self._read_index(self._stress_index_path)
        index = [e for e in index if e.get("stress_test_id") != stress_test_id]
        self._write_index(self._stress_index_path, index)

    def rebuild_stress_index(self) -> int:
        """Rebuild stress index.json from files on disk.

        Returns
        -------
        int
            Number of records indexed.
        """
        self._ensure_stress_dirs()
        entries: List[Dict[str, Any]] = []
        for p in sorted(self._stress_records_dir.glob("*.json")):
            try:
                record = StressTestRecord.from_json(p.read_text(encoding="utf-8"))
                entries.append({
                    "stress_test_id": record.stress_test_id,
                    "component_name": record.component_name,
                    "component_version": record.component_version,
                    "grdl_version": record.grdl_version,
                    "created_at": record.created_at,
                    "max_sustained_concurrency": (
                        record.summary.max_sustained_concurrency
                    ),
                    "first_failure_mode": record.summary.first_failure_mode,
                    "schema_version": record.schema_version,
                })
            except (json.JSONDecodeError, KeyError):
                continue
        self._write_index(self._stress_index_path, entries)
        return len(entries)

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    def save_report(
        self,
        record: Union[BenchmarkRecord, StressTestRecord],
        path: Union[str, Path],
        *,
        fmt: str = "text",
    ) -> None:
        """Save a report for either record type.

        Automatically selects the correct report formatter based on the
        record type.

        Parameters
        ----------
        record : BenchmarkRecord or StressTestRecord
        path : str or Path
            Output file path.
        fmt : str
            ``"text"`` (default) or ``"markdown"``.

        Raises
        ------
        TypeError
            If *record* is not a supported type.
        ValueError
            If *fmt* is not ``"text"`` or ``"markdown"``.
        """
        if fmt not in ("text", "markdown"):
            raise ValueError(f"fmt must be 'text' or 'markdown', got {fmt!r}")

        if isinstance(record, StressTestRecord):
            from grdl_te.benchmarking.stress_report import (
                save_stress_report,
                save_stress_report_md,
            )
            if fmt == "markdown":
                save_stress_report_md(record, path)
            else:
                save_stress_report(record, path)

        elif isinstance(record, BenchmarkRecord):
            from grdl_te.benchmarking.report import save_report
            from grdl_te.benchmarking.report_md import save_report_md
            if fmt == "markdown":
                save_report_md([record], path)
            else:
                save_report([record], path)

        else:
            raise TypeError(
                f"save_report() expects BenchmarkRecord or StressTestRecord, "
                f"got {type(record).__name__}"
            )
