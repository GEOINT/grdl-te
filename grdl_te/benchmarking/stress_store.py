# -*- coding: utf-8 -*-
"""
JSON Stress Test Store — file-based persistence for stress test records.

Stores each ``StressTestRecord`` as an individual JSON file under a
``stress/records/`` directory, with a lightweight ``stress/index.json``
for fast filtering without loading every record.

Storage layout::

    <base_dir>/
        stress/
            index.json
            records/
                <stress_test_id>.json

The same *base_dir* used by ``JSONBenchmarkStore`` is accepted here so
that both record types live under one shared store directory.

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
from pathlib import Path
from typing import Any, Dict, List, Optional

# Internal
from grdl_te.benchmarking.stress_models import StressTestRecord


class JSONStressTestStore:
    """File-system stress test store using one JSON file per record.

    Parameters
    ----------
    base_dir : Path or str, optional
        Root directory for storage.  Defaults to ``<cwd>/.benchmarks/``.
        Stress records are kept under ``<base_dir>/stress/``.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            base_dir = Path.cwd() / ".benchmarks"
        self._base_dir = Path(base_dir)
        self._stress_dir = self._base_dir / "stress"
        self._records_dir = self._stress_dir / "records"
        self._index_path = self._stress_dir / "index.json"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create storage directories on first save."""
        self._records_dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict[str, Any]]:
        """Load the index file, returning an empty list if absent."""
        if not self._index_path.exists():
            return []
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_index(self, index: List[Dict[str, Any]]) -> None:
        """Persist the index file."""
        self._index_path.write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, record: StressTestRecord) -> str:
        """Persist a stress test record as a JSON file.

        Parameters
        ----------
        record : StressTestRecord

        Returns
        -------
        str
            The ``stress_test_id`` of the saved record.
        """
        self._ensure_dirs()
        record_path = self._records_dir / f"{record.stress_test_id}.json"
        record_path.write_text(record.to_json(), encoding="utf-8")

        index = self._read_index()
        index.append(
            {
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
            }
        )
        self._write_index(index)

        return record.stress_test_id

    def load(self, stress_test_id: str) -> StressTestRecord:
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
            If no record file exists for the given ID.
        """
        record_path = self._records_dir / f"{stress_test_id}.json"
        if not record_path.exists():
            raise KeyError(
                f"No stress test record found with ID: {stress_test_id}"
            )
        return StressTestRecord.from_json(
            record_path.read_text(encoding="utf-8")
        )

    def list_records(
        self,
        component_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[StressTestRecord]:
        """List stress test records, optionally filtered by component name.

        Reads the lightweight index for filtering, then loads full records
        only for matching entries.

        Parameters
        ----------
        component_name : str, optional
            Filter by component name (exact match).
        limit : int
            Maximum records to return.  Default 50.

        Returns
        -------
        List[StressTestRecord]
            Records ordered by ``created_at`` descending (newest first).
        """
        index = self._read_index()

        if component_name is not None:
            index = [e for e in index if e.get("component_name") == component_name]

        # Newest first
        index = sorted(
            index, key=lambda e: e.get("created_at", ""), reverse=True
        )[:limit]

        records = []
        for entry in index:
            sid = entry.get("stress_test_id", "")
            try:
                records.append(self.load(sid))
            except (KeyError, Exception):
                continue

        return records

    def delete(self, stress_test_id: str) -> None:
        """Delete a stress test record and remove it from the index.

        Parameters
        ----------
        stress_test_id : str

        Raises
        ------
        KeyError
            If no record exists for the given ID.
        """
        record_path = self._records_dir / f"{stress_test_id}.json"
        if not record_path.exists():
            raise KeyError(
                f"No stress test record found with ID: {stress_test_id}"
            )
        record_path.unlink()

        index = self._read_index()
        index = [e for e in index if e.get("stress_test_id") != stress_test_id]
        self._write_index(index)
