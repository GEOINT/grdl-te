# -*- coding: utf-8 -*-
"""
Benchmark ABCs — contracts for runners and storage backends.

Defines ``BenchmarkRunner`` (abstract benchmark executor) and
``BenchmarkStore`` (abstract persistence backend).  Concrete
implementations inherit from these and provide specific execution
or storage strategies.

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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Internal
from grdl_te.benchmarking.models import BenchmarkRecord


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark execution modes.

    Subclasses implement specific execution strategies (active workflow
    benchmarking, component profiling, etc.) but all return a
    ``BenchmarkRecord`` from their ``run()`` method.
    """

    @property
    @abstractmethod
    def benchmark_type(self) -> str:
        """Return the benchmark type identifier.

        Returns
        -------
        str
            One of ``"active"``, ``"passive"``, ``"component"``.
        """
        ...

    @abstractmethod
    def run(self, **kwargs: Any) -> BenchmarkRecord:
        """Execute the benchmark and return a complete record.

        Parameters
        ----------
        **kwargs
            Runner-specific execution arguments.

        Returns
        -------
        BenchmarkRecord
        """
        ...


class BenchmarkStore(ABC):
    """Abstract base class for benchmark result persistence.

    Implementations may use JSON files, SQLite, or other backends.
    All records are identified by their ``benchmark_id``.
    """

    @abstractmethod
    def save(self, record: BenchmarkRecord) -> str:
        """Persist a benchmark record.

        Parameters
        ----------
        record : BenchmarkRecord
            The record to persist.

        Returns
        -------
        str
            The ``benchmark_id`` of the saved record.
        """
        ...

    @abstractmethod
    def load(self, benchmark_id: str) -> BenchmarkRecord:
        """Load a benchmark record by ID.

        Parameters
        ----------
        benchmark_id : str
            The unique benchmark identifier.

        Returns
        -------
        BenchmarkRecord

        Raises
        ------
        KeyError
            If no record exists with the given ID.
        """
        ...

    @abstractmethod
    def list_records(
        self,
        workflow_name: Optional[str] = None,
        benchmark_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[BenchmarkRecord]:
        """List benchmark records, optionally filtered.

        Parameters
        ----------
        workflow_name : str, optional
            Filter by workflow name.
        benchmark_type : str, optional
            Filter by benchmark type (``"active"``, ``"component"``, etc.).
        limit : int
            Maximum number of records to return.  Default 50.

        Returns
        -------
        List[BenchmarkRecord]
            Records sorted by ``created_at`` descending (newest first).
        """
        ...

    @abstractmethod
    def delete(self, benchmark_id: str) -> None:
        """Remove a benchmark record.

        Parameters
        ----------
        benchmark_id : str
            The unique benchmark identifier.

        Raises
        ------
        KeyError
            If no record exists with the given ID.
        """
        ...
