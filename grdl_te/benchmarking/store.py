# -*- coding: utf-8 -*-
"""
JSON Benchmark Store — file-based persistence for benchmark records.

Stores each ``BenchmarkRecord`` as an individual JSON file under a
``records/`` directory, with a lightweight ``index.json`` for fast
filtering without loading every record.

Storage layout::

    <base_dir>/
        index.json
        records/
            <benchmark_id>.json

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
2026-02-13
"""

# Standard library
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Internal
from grdl_te.benchmarking.base import BenchmarkStore
from grdl_te.benchmarking.models import BenchmarkRecord


class JSONBenchmarkStore(BenchmarkStore):
    """File-system benchmark store using one JSON file per record.

    Parameters
    ----------
    base_dir : Path or str, optional
        Root directory for benchmark storage.  Defaults to
        ``<cwd>/.benchmarks/``.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            base_dir = Path.cwd() / ".benchmarks"
        self._base_dir = Path(base_dir)
        self._records_dir = self._base_dir / "records"
        self._index_path = self._base_dir / "index.json"

        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._records_dir.mkdir(parents=True, exist_ok=True)

        if not self._index_path.exists():
            self._write_index([])

    def save(self, record: BenchmarkRecord) -> str:
        """Persist a benchmark record as a JSON file.

        Parameters
        ----------
        record : BenchmarkRecord

        Returns
        -------
        str
            The ``benchmark_id`` of the saved record.
        """
        record_path = self._records_dir / f"{record.benchmark_id}.json"
        record_path.write_text(record.to_json(), encoding="utf-8")

        index = self._read_index()
        index.append({
            "benchmark_id": record.benchmark_id,
            "benchmark_type": record.benchmark_type,
            "workflow_name": record.workflow_name,
            "workflow_version": record.workflow_version,
            "iterations": record.iterations,
            "created_at": record.created_at,
        })
        self._write_index(index)

        return record.benchmark_id

    def load(self, benchmark_id: str) -> BenchmarkRecord:
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
            If no record file exists for the given ID.
        """
        record_path = self._records_dir / f"{benchmark_id}.json"
        if not record_path.exists():
            raise KeyError(
                f"No benchmark record found with ID: {benchmark_id}"
            )
        return BenchmarkRecord.from_json(
            record_path.read_text(encoding="utf-8")
        )

    def list_records(
        self,
        workflow_name: Optional[str] = None,
        benchmark_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[BenchmarkRecord]:
        """List benchmark records, optionally filtered.

        Reads the lightweight index for filtering, then loads full
        records only for matching entries.

        Parameters
        ----------
        workflow_name : str, optional
            Filter by workflow name.
        benchmark_type : str, optional
            Filter by benchmark type.
        limit : int
            Maximum records to return.  Default 50.

        Returns
        -------
        List[BenchmarkRecord]
            Sorted by ``created_at`` descending (newest first).
        """
        index = self._read_index()

        # Sort newest first
        index.sort(key=lambda e: e.get("created_at", ""), reverse=True)

        filtered: List[Dict[str, Any]] = []
        for entry in index:
            if workflow_name and entry.get("workflow_name") != workflow_name:
                continue
            if benchmark_type and entry.get("benchmark_type") != benchmark_type:
                continue
            filtered.append(entry)
            if len(filtered) >= limit:
                break

        records: List[BenchmarkRecord] = []
        for entry in filtered:
            try:
                records.append(self.load(entry["benchmark_id"]))
            except KeyError:
                continue  # stale index entry, skip

        return records

    def delete(self, benchmark_id: str) -> None:
        """Remove a benchmark record and its index entry.

        Parameters
        ----------
        benchmark_id : str

        Raises
        ------
        KeyError
            If no record exists with the given ID.
        """
        record_path = self._records_dir / f"{benchmark_id}.json"
        if not record_path.exists():
            raise KeyError(
                f"No benchmark record found with ID: {benchmark_id}"
            )
        record_path.unlink()

        index = self._read_index()
        index = [
            e for e in index if e.get("benchmark_id") != benchmark_id
        ]
        self._write_index(index)

    def rebuild_index(self) -> int:
        """Rebuild index.json from record files on disk.

        Useful for recovery if the index becomes corrupted or
        out of sync with the actual record files.

        Returns
        -------
        int
            Number of records indexed.
        """
        entries: List[Dict[str, Any]] = []
        for path in sorted(self._records_dir.glob("*.json")):
            try:
                record = BenchmarkRecord.from_json(
                    path.read_text(encoding="utf-8")
                )
                entries.append({
                    "benchmark_id": record.benchmark_id,
                    "benchmark_type": record.benchmark_type,
                    "workflow_name": record.workflow_name,
                    "workflow_version": record.workflow_version,
                    "iterations": record.iterations,
                    "created_at": record.created_at,
                })
            except (json.JSONDecodeError, KeyError):
                continue  # skip corrupted files

        self._write_index(entries)
        return len(entries)

    def _read_index(self) -> List[Dict[str, Any]]:
        """Read the index file.

        Returns
        -------
        List[Dict[str, Any]]
        """
        if not self._index_path.exists():
            return []
        try:
            data = json.loads(
                self._index_path.read_text(encoding="utf-8")
            )
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def _write_index(self, entries: List[Dict[str, Any]]) -> None:
        """Write the index file atomically.

        Writes to a temporary file first, then renames to avoid
        partial writes on crash.

        Parameters
        ----------
        entries : List[Dict[str, Any]]
        """
        tmp_path = self._index_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(entries, indent=2), encoding="utf-8"
        )
        tmp_path.replace(self._index_path)
