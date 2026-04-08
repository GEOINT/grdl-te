# -*- coding: utf-8 -*-
"""
Tests for Stress Test Report Generators.

Validates ``format_stress_report`` (plain text) and
``format_stress_report_md`` (Markdown) output format, required sections,
failure analysis rendering, and file persistance via ``save_stress_report``
and ``save_stress_report_md``.

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
from datetime import datetime, timezone
from pathlib import Path

# Third-party
import pytest

# Internal
from grdl_te.benchmarking.stress_models import (
    FailurePoint,
    StressTestConfig,
    StressTestEvent,
    StressTestRecord,
)
from grdl_te.benchmarking.stress_report import (
    format_stress_report,
    format_stress_report_md,
    save_stress_report,
    save_stress_report_md,
)

pytestmark = pytest.mark.stress_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_no_failures() -> StressTestRecord:
    """Build a test record with no failures."""
    config = StressTestConfig(
        start_concurrency=1, max_concurrency=4, ramp_steps=2,
        duration_per_step_s=1.0, payload_size="small",
    )
    events = [
        StressTestEvent(1, (512, 512), True, 0.05, 1024, None, _ts()),
        StressTestEvent(2, (512, 512), True, 0.08, 2048, None, _ts()),
        StressTestEvent(4, (512, 512), True, 0.12, 3072, None, _ts()),
    ]
    return StressTestRecord.create(
        component_name="TestComp",
        component_version="1.0.0",
        hardware=None,
        config=config,
        events=events,
        failure_points=[],
        related_benchmark_id=None,
    )


def _record_with_failures() -> StressTestRecord:
    """Build a test record with a failure at concurrency=4."""
    config = StressTestConfig(
        start_concurrency=1, max_concurrency=4, ramp_steps=2,
        duration_per_step_s=1.0, payload_size="small",
    )
    events = [
        StressTestEvent(1, (512, 512), True, 0.05, 1024, None, _ts()),
        StressTestEvent(4, (512, 512), False, 0.50, 0, "MemoryError", _ts()),
    ]
    fp = FailurePoint(
        concurrency_level=4,
        payload_shape=(512, 512),
        error_type="MemoryError",
        error_message="simulated OOM",
        memory_bytes_at_failure=1024 * 1024 * 512,
        first_occurrence_at=_ts(),
    )
    return StressTestRecord.create(
        component_name="TestComp",
        component_version="1.0.0",
        hardware=None,
        config=config,
        events=events,
        failure_points=[fp],
        related_benchmark_id="bench-linked-id",
    )


# ---------------------------------------------------------------------------
# Plain-text report
# ---------------------------------------------------------------------------

class TestFormatStressReport:
    """Tests for format_stress_report() plain-text output."""

    def test_returns_nonempty_string(self):
        """Output is a non-empty string."""
        report = format_stress_report(_record_no_failures())
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_component_name(self):
        """Report includes the component name."""
        report = format_stress_report(_record_no_failures())
        assert "TestComp" in report

    def test_contains_run_id(self):
        """Report includes the stress_test_id."""
        record = _record_no_failures()
        report = format_stress_report(record)
        assert record.stress_test_id in report

    def test_contains_saturation_curve_header(self):
        """Report contains a SATURATION CURVE section."""
        report = format_stress_report(_record_no_failures())
        assert "SATURATION CURVE" in report

    def test_contains_failure_analysis_header(self):
        """Report has a FAILURE ANALYSIS section."""
        report = format_stress_report(_record_no_failures())
        assert "FAILURE ANALYSIS" in report

    def test_contains_summary_header(self):
        """Report has a SUMMARY section."""
        report = format_stress_report(_record_no_failures())
        assert "SUMMARY" in report

    def test_no_failures_note(self):
        """No-failure report says 'No failures detected'."""
        report = format_stress_report(_record_no_failures())
        assert "No failures detected" in report

    def test_failure_mode_in_report(self):
        """Report with failures shows the error type."""
        report = format_stress_report(_record_with_failures())
        assert "MemoryError" in report

    def test_cross_reference_present_when_linked(self):
        """Cross-reference section appears when related_benchmark_id is set."""
        report = format_stress_report(_record_with_failures())
        assert "bench-linked-id" in report
        assert "CROSS-REFERENCE" in report

    def test_no_cross_reference_when_no_link(self):
        """Cross-reference section absent when related_benchmark_id is None."""
        report = format_stress_report(_record_no_failures())
        assert "CROSS-REFERENCE" not in report

    def test_saturation_curve_lists_concurrency_levels(self):
        """Saturation curve contains at least one numeric worker row."""
        report = format_stress_report(_record_no_failures())
        # Check that each configured level appears in the saturation table
        for level in _record_no_failures().config.concurrency_levels():
            assert str(level) in report

    def test_hardware_absent_when_none(self):
        """HARDWARE section is absent when hardware is None."""
        report = format_stress_report(_record_no_failures())
        # No hardware was attached; the section should not appear
        assert "HARDWARE" not in report


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

class TestFormatStressReportMd:
    """Tests for format_stress_report_md() Markdown output."""

    def test_returns_nonempty_string(self):
        """Output is a non-empty string."""
        md = format_stress_report_md(_record_no_failures())
        assert isinstance(md, str)
        assert len(md) > 0

    def test_h1_header_present(self):
        """Report starts with an H1 markdown header."""
        md = format_stress_report_md(_record_no_failures())
        assert "# GRDL Stress Test Report" in md

    def test_h2_sections_present(self):
        """Report contains all major H2 sections."""
        md = format_stress_report_md(_record_no_failures())
        for section in ["## Stress Configuration", "## Saturation Curve",
                        "## Failure Analysis", "## Summary"]:
            assert section in md, f"Missing section: {section}"

    def test_component_name_in_table(self):
        """Component name appears in the metadata table."""
        md = format_stress_report_md(_record_no_failures())
        assert "TestComp" in md

    def test_failure_analysis_shows_error_type(self):
        """MemoryError appears in the Markdown failure analysis."""
        md = format_stress_report_md(_record_with_failures())
        assert "MemoryError" in md

    def test_no_failures_note_in_markdown(self):
        """Markdown failure section says no failures when none occurred."""
        md = format_stress_report_md(_record_no_failures())
        assert "No failures detected" in md

    def test_cross_reference_section_in_markdown(self):
        """Cross-reference section appears in Markdown when linked."""
        md = format_stress_report_md(_record_with_failures())
        assert "## Cross-Reference" in md
        assert "bench-linked-id" in md

    def test_table_pipes_present(self):
        """Markdown tables use pipe syntax."""
        md = format_stress_report_md(_record_no_failures())
        # At least a few table rows
        assert md.count("|") > 10

    def test_hardware_section_absent_when_none(self):
        """Hardware section is absent when no hardware snapshot."""
        md = format_stress_report_md(_record_no_failures())
        assert "## Hardware" not in md


# ---------------------------------------------------------------------------
# File persistance
# ---------------------------------------------------------------------------

class TestSaveStressReport:
    """Tests for save_stress_report() and save_stress_report_md()."""

    def test_save_to_directory_creates_txt(self, tmp_path):
        """Saving to a directory creates a .txt file."""
        record = _record_no_failures()
        result = save_stress_report(record, tmp_path)
        assert result.exists()
        assert result.suffix == ".txt"
        assert result.name.startswith("stress_")

    def test_save_to_file_path(self, tmp_path):
        """Saving to an explicit file path writes that exact file."""
        record = _record_no_failures()
        target = tmp_path / "my_report.txt"
        result = save_stress_report(record, target)
        assert result == target
        assert target.exists()
        assert len(target.read_text()) > 0

    def test_saved_txt_contains_component_name(self, tmp_path):
        """Saved text file contains the component name."""
        record = _record_no_failures()
        result = save_stress_report(record, tmp_path)
        assert "TestComp" in result.read_text()

    def test_save_md_to_directory_creates_md(self, tmp_path):
        """Saving Markdown to a directory creates a .md file."""
        record = _record_no_failures()
        result = save_stress_report_md(record, tmp_path)
        assert result.exists()
        assert result.suffix == ".md"
        assert result.name.startswith("stress_")

    def test_save_md_to_file_path(self, tmp_path):
        """Saving Markdown to an explicit path writes that file."""
        record = _record_no_failures()
        target = tmp_path / "report.md"
        result = save_stress_report_md(record, target)
        assert result == target
        assert target.exists()

    def test_saved_md_contains_h1(self, tmp_path):
        """Saved Markdown file begins with an H1 heading."""
        record = _record_no_failures()
        result = save_stress_report_md(record, tmp_path)
        content = result.read_text()
        assert "# GRDL Stress Test Report" in content

    def test_saves_create_parent_dirs(self, tmp_path):
        """Saving to a nested path creates intermediate directories."""
        record = _record_no_failures()
        deep_path = tmp_path / "deep" / "nested" / "report.txt"
        result = save_stress_report(record, deep_path)
        assert result.exists()
