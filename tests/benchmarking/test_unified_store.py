# -*- coding: utf-8 -*-
"""
Tests for GRDLStore — unified benchmark and stress test store.

Validates that GRDLStore correctly routes BenchmarkRecord and
StressTestRecord to their respective subdirectories, supports the full
round-trip (save → load), filtering, deletion, index rebuild, type
dispatch via save(), and save_report() format dispatch.

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

# Third-party
import numpy as np
import pytest

# Internal
from grdl_te.benchmarking.models import BenchmarkRecord
from grdl_te.benchmarking.stress_models import (
    StressTestConfig,
    StressTestRecord,
)
from grdl_te.benchmarking.unified_store import GRDLStore

pytestmark = pytest.mark.stress_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stress_record(component_name: str = "TestComp") -> StressTestRecord:
    """Build a minimal StressTestRecord with predictable content."""
    import datetime
    from grdl_te.benchmarking.models import HardwareSnapshot
    from grdl_te.benchmarking.stress_models import StressTestEvent

    config = StressTestConfig(
        start_concurrency=1,
        max_concurrency=2,
        ramp_steps=2,
        duration_per_step_s=0.1,
        payload_size="small",
    )
    events = [
        StressTestEvent(
            concurrency_level=1,
            payload_shape=(512, 512),
            success=True,
            latency_s=0.01,
            peak_rss_bytes=1024,
            error_type=None,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
    ]
    return StressTestRecord.create(
        component_name=component_name,
        component_version="0.0.1",
        hardware=HardwareSnapshot.capture(),
        config=config,
        events=events,
        failure_points=[],
    )


def _make_benchmark_record() -> BenchmarkRecord:
    """Build a minimal BenchmarkRecord using its factory."""
    from grdl_te.benchmarking.models import (
        AggregatedMetrics,
        HardwareSnapshot,
        StepBenchmarkResult,
    )

    hw = HardwareSnapshot.capture()
    metrics = AggregatedMetrics.from_values([0.1, 0.11, 0.09])
    step = StepBenchmarkResult(
        step_index=0,
        processor_name="test_step",
        wall_time_s=metrics,
        cpu_time_s=AggregatedMetrics.from_values([0.05, 0.06, 0.04]),
        peak_rss_bytes=AggregatedMetrics.from_values([1024.0, 1024.0, 1024.0]),
        gpu_used=False,
        sample_count=3,
    )
    return BenchmarkRecord.create(
        benchmark_type="component",
        workflow_name="test_workflow",
        workflow_version="0.0.1",
        iterations=3,
        hardware=hw,
        total_wall_time=metrics,
        total_cpu_time=AggregatedMetrics.from_values([0.05, 0.06, 0.04]),
        total_peak_rss=AggregatedMetrics.from_values([1024.0, 1024.0, 1024.0]),
        step_results=[step],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> GRDLStore:
    """GRDLStore backed by a temporary directory."""
    return GRDLStore(base_dir=tmp_path / "store")


@pytest.fixture()
def stress_record() -> StressTestRecord:
    return _make_stress_record()


@pytest.fixture()
def benchmark_record() -> BenchmarkRecord:
    return _make_benchmark_record()


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

class TestStoreLayout:
    def test_benchmark_dirs_created_on_first_save(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord
    ) -> None:
        store.save(benchmark_record)
        assert (store._bench_records_dir).is_dir()
        assert (store._bench_index_path).is_file()

    def test_stress_dirs_created_on_first_save(
        self, store: GRDLStore, stress_record: StressTestRecord
    ) -> None:
        store.save(stress_record)
        assert (store._stress_records_dir).is_dir()
        assert (store._stress_index_path).is_file()

    def test_benchmark_and_stress_use_separate_subdirs(
        self, store: GRDLStore,
        benchmark_record: BenchmarkRecord,
        stress_record: StressTestRecord,
    ) -> None:
        store.save(benchmark_record)
        store.save(stress_record)
        bench_files = list(store._bench_records_dir.glob("*.json"))
        stress_files = list(store._stress_records_dir.glob("*.json"))
        assert len(bench_files) == 1
        assert len(stress_files) == 1


# ---------------------------------------------------------------------------
# save() dispatch
# ---------------------------------------------------------------------------

class TestSaveDispatch:
    def test_save_benchmark_record_returns_benchmark_id(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord
    ) -> None:
        saved_id = store.save(benchmark_record)
        assert saved_id == benchmark_record.benchmark_id

    def test_save_stress_record_returns_stress_test_id(
        self, store: GRDLStore, stress_record: StressTestRecord
    ) -> None:
        saved_id = store.save(stress_record)
        assert saved_id == stress_record.stress_test_id

    def test_save_unknown_type_raises_type_error(
        self, store: GRDLStore
    ) -> None:
        with pytest.raises(TypeError, match="BenchmarkRecord or StressTestRecord"):
            store.save("not a record")  # type: ignore[arg-type]

    def test_save_benchmark_writes_json_file(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord
    ) -> None:
        store.save(benchmark_record)
        path = store._bench_records_dir / f"{benchmark_record.benchmark_id}.json"
        assert path.is_file()
        data = json.loads(path.read_text())
        assert data["benchmark_id"] == benchmark_record.benchmark_id

    def test_save_stress_writes_json_file(
        self, store: GRDLStore, stress_record: StressTestRecord
    ) -> None:
        store.save(stress_record)
        path = store._stress_records_dir / f"{stress_record.stress_test_id}.json"
        assert path.is_file()
        data = json.loads(path.read_text())
        assert data["stress_test_id"] == stress_record.stress_test_id


# ---------------------------------------------------------------------------
# Benchmark round-trip
# ---------------------------------------------------------------------------

class TestBenchmarkRoundTrip:
    def test_load_benchmark_returns_identical_record(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord
    ) -> None:
        store.save(benchmark_record)
        loaded = store.load_benchmark(benchmark_record.benchmark_id)
        assert loaded.benchmark_id == benchmark_record.benchmark_id
        assert loaded.workflow_name == benchmark_record.workflow_name

    def test_load_benchmark_unknown_id_raises_key_error(
        self, store: GRDLStore
    ) -> None:
        with pytest.raises(KeyError, match="No benchmark record found"):
            store.load_benchmark("does-not-exist")

    def test_list_benchmarks_empty_store(self, store: GRDLStore) -> None:
        assert store.list_benchmarks() == []

    def test_list_benchmarks_returns_saved_record(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord
    ) -> None:
        store.save(benchmark_record)
        records = store.list_benchmarks()
        assert len(records) == 1
        assert records[0].benchmark_id == benchmark_record.benchmark_id

    def test_list_benchmarks_filter_by_workflow_name(
        self, store: GRDLStore
    ) -> None:
        r1 = _make_benchmark_record()
        r2 = _make_benchmark_record()
        store.save(r1)
        store.save(r2)
        results = store.list_benchmarks(workflow_name="test_workflow")
        assert len(results) == 2
        results_miss = store.list_benchmarks(workflow_name="nonexistent")
        assert results_miss == []

    def test_delete_benchmark_removes_file_and_index(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord
    ) -> None:
        store.save(benchmark_record)
        store.delete_benchmark(benchmark_record.benchmark_id)
        path = store._bench_records_dir / f"{benchmark_record.benchmark_id}.json"
        assert not path.exists()
        index = json.loads(store._bench_index_path.read_text())
        ids = [e["benchmark_id"] for e in index]
        assert benchmark_record.benchmark_id not in ids

    def test_delete_benchmark_unknown_id_raises_key_error(
        self, store: GRDLStore
    ) -> None:
        with pytest.raises(KeyError, match="No benchmark record found"):
            store.delete_benchmark("not-there")

    def test_rebuild_benchmark_index(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord
    ) -> None:
        store.save(benchmark_record)
        # Corrupt the index
        store._bench_index_path.write_text("[]")
        count = store.rebuild_benchmark_index()
        assert count == 1
        loaded = store.load_benchmark(benchmark_record.benchmark_id)
        assert loaded.benchmark_id == benchmark_record.benchmark_id


# ---------------------------------------------------------------------------
# Stress record round-trip
# ---------------------------------------------------------------------------

class TestStressRoundTrip:
    def test_load_stress_returns_identical_record(
        self, store: GRDLStore, stress_record: StressTestRecord
    ) -> None:
        store.save(stress_record)
        loaded = store.load_stress(stress_record.stress_test_id)
        assert loaded.stress_test_id == stress_record.stress_test_id
        assert loaded.component_name == stress_record.component_name

    def test_load_stress_unknown_id_raises_key_error(
        self, store: GRDLStore
    ) -> None:
        with pytest.raises(KeyError, match="No stress test record found"):
            store.load_stress("does-not-exist")

    def test_list_stress_tests_empty_store(self, store: GRDLStore) -> None:
        assert store.list_stress_tests() == []

    def test_list_stress_tests_returns_saved_record(
        self, store: GRDLStore, stress_record: StressTestRecord
    ) -> None:
        store.save(stress_record)
        records = store.list_stress_tests()
        assert len(records) == 1
        assert records[0].stress_test_id == stress_record.stress_test_id

    def test_list_stress_tests_filter_by_component_name(
        self, store: GRDLStore
    ) -> None:
        r1 = _make_stress_record("CompA")
        r2 = _make_stress_record("CompB")
        store.save(r1)
        store.save(r2)
        results_a = store.list_stress_tests(component_name="CompA")
        assert len(results_a) == 1
        assert results_a[0].component_name == "CompA"
        results_miss = store.list_stress_tests(component_name="CompC")
        assert results_miss == []

    def test_delete_stress_removes_file_and_index(
        self, store: GRDLStore, stress_record: StressTestRecord
    ) -> None:
        store.save(stress_record)
        store.delete_stress(stress_record.stress_test_id)
        path = store._stress_records_dir / f"{stress_record.stress_test_id}.json"
        assert not path.exists()
        index = json.loads(store._stress_index_path.read_text())
        ids = [e["stress_test_id"] for e in index]
        assert stress_record.stress_test_id not in ids

    def test_delete_stress_unknown_id_raises_key_error(
        self, store: GRDLStore
    ) -> None:
        with pytest.raises(KeyError, match="No stress test record found"):
            store.delete_stress("not-there")

    def test_rebuild_stress_index(
        self, store: GRDLStore, stress_record: StressTestRecord
    ) -> None:
        store.save(stress_record)
        # Corrupt the index
        store._stress_index_path.write_text("[]")
        count = store.rebuild_stress_index()
        assert count == 1
        loaded = store.load_stress(stress_record.stress_test_id)
        assert loaded.stress_test_id == stress_record.stress_test_id


# ---------------------------------------------------------------------------
# save_report() dispatch
# ---------------------------------------------------------------------------

class TestSaveReportDispatch:
    def test_save_report_text_for_stress_record(
        self, store: GRDLStore, stress_record: StressTestRecord, tmp_path: Path
    ) -> None:
        path = tmp_path / "stress.txt"
        store.save_report(stress_record, path, fmt="text")
        assert path.is_file()
        content = path.read_text()
        assert stress_record.component_name in content

    def test_save_report_markdown_for_stress_record(
        self, store: GRDLStore, stress_record: StressTestRecord, tmp_path: Path
    ) -> None:
        path = tmp_path / "stress.md"
        store.save_report(stress_record, path, fmt="markdown")
        assert path.is_file()
        content = path.read_text()
        assert "##" in content

    def test_save_report_text_for_benchmark_record(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord, tmp_path: Path
    ) -> None:
        path = tmp_path / "bench.txt"
        store.save_report(benchmark_record, path, fmt="text")
        assert path.is_file()
        content = path.read_text()
        assert benchmark_record.workflow_name in content

    def test_save_report_markdown_for_benchmark_record(
        self, store: GRDLStore, benchmark_record: BenchmarkRecord, tmp_path: Path
    ) -> None:
        path = tmp_path / "bench.md"
        store.save_report(benchmark_record, path, fmt="markdown")
        assert path.is_file()

    def test_save_report_invalid_fmt_raises_value_error(
        self, store: GRDLStore, stress_record: StressTestRecord, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="fmt must be"):
            store.save_report(stress_record, tmp_path / "out.txt", fmt="html")

    def test_save_report_unknown_type_raises_type_error(
        self, store: GRDLStore, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="BenchmarkRecord or StressTestRecord"):
            store.save_report("bad", tmp_path / "out.txt")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default base_dir
# ---------------------------------------------------------------------------

class TestDefaultBaseDir:
    def test_default_base_dir_is_dotbenchmarks(self) -> None:
        store = GRDLStore()
        assert store._base_dir == Path.cwd() / ".benchmarks"

    def test_str_base_dir_accepted(self, tmp_path: Path) -> None:
        store = GRDLStore(base_dir=str(tmp_path / "s"))
        assert isinstance(store._base_dir, Path)
