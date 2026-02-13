# -*- coding: utf-8 -*-
"""
Tests for JSONBenchmarkStore.

Validates save/load round-trips, filtering, deletion, and index
rebuild using temporary directories.

Author
------
Steven Siebert

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

# Third-party
import pytest

# Internal
from grdl_te.benchmarking.models import (
    AggregatedMetrics,
    BenchmarkRecord,
    HardwareSnapshot,
    StepBenchmarkResult,
)
from grdl_te.benchmarking.store import JSONBenchmarkStore


def _make_record(
    workflow_name: str = "TestWorkflow",
    benchmark_type: str = "active",
    version: str = "1.0.0",
) -> BenchmarkRecord:
    """Create a minimal valid BenchmarkRecord for testing."""
    return BenchmarkRecord.create(
        benchmark_type=benchmark_type,
        workflow_name=workflow_name,
        workflow_version=version,
        iterations=3,
        hardware=HardwareSnapshot.capture(),
        total_wall_time=AggregatedMetrics.from_values([1.0, 2.0, 3.0]),
        total_cpu_time=AggregatedMetrics.from_values([0.5, 1.0, 1.5]),
        total_peak_rss=AggregatedMetrics.from_values(
            [1000.0, 2000.0, 3000.0]
        ),
    )


class TestJSONBenchmarkStore:
    """Tests for JSONBenchmarkStore."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load produces identical record."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        record = _make_record()

        saved_id = store.save(record)
        loaded = store.load(saved_id)

        assert loaded.benchmark_id == record.benchmark_id
        assert loaded.workflow_name == record.workflow_name
        assert loaded.benchmark_type == record.benchmark_type
        assert loaded.iterations == record.iterations
        assert loaded.total_wall_time.mean == pytest.approx(
            record.total_wall_time.mean
        )

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading a nonexistent ID raises KeyError."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        with pytest.raises(KeyError, match="No benchmark record"):
            store.load("nonexistent-id")

    def test_list_records_all(self, tmp_path):
        """list_records with no filters returns all records."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        for _ in range(3):
            store.save(_make_record())

        records = store.list_records()
        assert len(records) == 3

    def test_list_records_filter_by_name(self, tmp_path):
        """list_records filters by workflow_name."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        store.save(_make_record(workflow_name="Alpha"))
        store.save(_make_record(workflow_name="Beta"))
        store.save(_make_record(workflow_name="Alpha"))

        records = store.list_records(workflow_name="Alpha")
        assert len(records) == 2
        assert all(r.workflow_name == "Alpha" for r in records)

    def test_list_records_filter_by_type(self, tmp_path):
        """list_records filters by benchmark_type."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        store.save(_make_record(benchmark_type="active"))
        store.save(_make_record(benchmark_type="component"))
        store.save(_make_record(benchmark_type="active"))

        records = store.list_records(benchmark_type="component")
        assert len(records) == 1
        assert records[0].benchmark_type == "component"

    def test_list_records_limit(self, tmp_path):
        """list_records respects the limit parameter."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        for _ in range(5):
            store.save(_make_record())

        records = store.list_records(limit=2)
        assert len(records) == 2

    def test_list_records_newest_first(self, tmp_path):
        """list_records returns newest records first."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        ids = []
        for _ in range(3):
            record = _make_record()
            store.save(record)
            ids.append(record.benchmark_id)

        records = store.list_records()
        # Last saved should be first returned (newest)
        returned_ids = [r.benchmark_id for r in records]
        assert returned_ids == list(reversed(ids))

    def test_delete(self, tmp_path):
        """Delete removes record and index entry."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        record = _make_record()
        store.save(record)

        store.delete(record.benchmark_id)

        with pytest.raises(KeyError):
            store.load(record.benchmark_id)

        records = store.list_records()
        assert len(records) == 0

    def test_delete_nonexistent_raises(self, tmp_path):
        """Deleting a nonexistent ID raises KeyError."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        with pytest.raises(KeyError, match="No benchmark record"):
            store.delete("nonexistent-id")

    def test_rebuild_index(self, tmp_path):
        """rebuild_index reconstructs index from record files."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        for _ in range(3):
            store.save(_make_record())

        # Corrupt the index
        (tmp_path / "index.json").write_text("[]", encoding="utf-8")

        # Rebuild
        count = store.rebuild_index()
        assert count == 3

        # Verify records are now listed
        records = store.list_records()
        assert len(records) == 3

    def test_creates_directories(self, tmp_path):
        """Store creates base_dir and records/ on init."""
        store_dir = tmp_path / "benchmarks" / "deep"
        store = JSONBenchmarkStore(base_dir=store_dir)

        assert store_dir.exists()
        assert (store_dir / "records").exists()
        assert (store_dir / "index.json").exists()

    def test_record_files_on_disk(self, tmp_path):
        """Each record is stored as a separate JSON file."""
        store = JSONBenchmarkStore(base_dir=tmp_path)
        record = _make_record()
        store.save(record)

        record_file = tmp_path / "records" / f"{record.benchmark_id}.json"
        assert record_file.exists()

        # File contains valid JSON
        import json
        data = json.loads(record_file.read_text(encoding="utf-8"))
        assert data["benchmark_id"] == record.benchmark_id
