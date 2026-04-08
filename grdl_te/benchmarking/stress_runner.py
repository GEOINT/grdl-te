# -*- coding: utf-8 -*-
"""
Stress test runners — callable and grdl-runtime workflow stress testers.

``ComponentStressTester`` wraps any Python callable, mirroring the
``ComponentBenchmark`` pattern so that components already set up for
benchmarking can be stress-tested with minimal extra code.

``WorkflowStressTester`` wraps a grdl-runtime ``Workflow`` or
``WorkflowDefinition`` object so users do not have to implement
``call_once`` by hand for pipeline stress tests.

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
2026-05-13
"""

# Standard library
from typing import Any, Callable, Dict, Optional, Tuple

# Third-party
import numpy as np

# Internal
from grdl_te.benchmarking.stress_base import BaseStressTester
from grdl_te.benchmarking.stress_models import StressTestConfig, StressTestRecord


class ComponentStressTester(BaseStressTester):
    """Stress test a single callable with the standard concurrency ramp.

    Wraps any callable so it can be stress-tested without subclassing
    ``BaseStressTester`` directly.  The callable receives the payload
    array as its first positional argument unless a *setup* function is
    provided.

    Parameters
    ----------
    name : str
        Human-readable component name.  Stored in the ``StressTestRecord``.
    fn : callable
        The function to call under stress.  Must accept at least one
        positional argument (the payload array) when *setup* is ``None``.
    setup : callable, optional
        Called once before each call to produce ``(args, kwargs)``.
        Must accept one argument — the payload ``np.ndarray``.  When
        provided, *fn* is called as ``fn(*args, **kwargs)`` and the
        payload is not passed directly.  When ``None``, *fn* is called
        as ``fn(payload)``.
    version : str
        Version string for the component.  Default ``"0.0.0"``.
    related_benchmark_id : str, optional
        ``benchmark_id`` of an associated ``BenchmarkRecord``.
    store : JSONStressTestStore, optional
        If provided, the record is persisted after ``run()`` completes.
    tags : Dict[str, str], optional
        User-defined labels.

    Examples
    --------
    Basic usage — wrap a normalizer:

    >>> import numpy as np
    >>> from grdl.data_prep import Normalizer
    >>> from grdl_te.benchmarking import ComponentStressTester, StressTestConfig
    >>>
    >>> image = np.random.rand(512, 512).astype(np.float32)
    >>> norm = Normalizer(method='minmax')
    >>> tester = ComponentStressTester(
    ...     "Normalizer.minmax",
    ...     norm.normalize,
    ... )
    >>> config = StressTestConfig(max_concurrency=8, ramp_steps=3)
    >>> record = tester.run(config)
    >>> print(record.summary.max_sustained_concurrency)

    Using a setup function to control what arguments are passed:

    >>> def my_setup(payload):
    ...     return (payload,), {"clip": True}
    >>>
    >>> tester = ComponentStressTester("MyFunc", my_fn, setup=my_setup)
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        setup: Optional[Callable[[np.ndarray], Tuple[tuple, dict]]] = None,
        version: str = "0.0.0",
        related_benchmark_id: Optional[str] = None,
        store: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            version=version,
            related_benchmark_id=related_benchmark_id,
            store=store,
            tags=tags,
        )
        self._name = name
        self._fn = fn
        self._setup = setup

    @property
    def component_name(self) -> str:
        """Human-readable name for the component under stress.

        Returns
        -------
        str
        """
        return self._name

    def call_once(self, payload: np.ndarray) -> Any:
        """Execute a single call of the wrapped function.

        Parameters
        ----------
        payload : np.ndarray
            Input array for this call.

        Returns
        -------
        Any
        """
        if self._setup is not None:
            args, kwargs = self._setup(payload)
            return self._fn(*args, **kwargs)
        return self._fn(payload)


# ---------------------------------------------------------------------------
# grdl-runtime workflow integration
# ---------------------------------------------------------------------------

try:
    from grdl_rt.execution.builder import Workflow as _WorkflowBuilder  # type: ignore
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False


class WorkflowStressTester(BaseStressTester):
    """Stress test a grdl-runtime ``Workflow`` without writing ``call_once``.

    Accepts any grdl-runtime ``Workflow`` (the fluent builder object) and
    calls its ``execute`` method on each payload during the concurrency
    ramp.  The result's ``result`` attribute (the processed array) is
    returned from ``call_once`` without further processing.

    Parameters
    ----------
    workflow : Workflow
        A configured grdl-runtime ``Workflow`` builder.  The workflow
        must be executable in *array mode* — i.e., ``workflow.execute(array)``
        must return a ``WorkflowResult``.
    name : str, optional
        Human-readable name stored in the ``StressTestRecord``.  Defaults
        to ``workflow.name``.
    prefer_gpu : bool
        Forwarded to ``workflow.execute``.  Default ``False``.
    version : str
        Version string for the component.  Default ``"0.0.0"``.
    related_benchmark_id : str, optional
        ``benchmark_id`` of an associated ``BenchmarkRecord``.
    store : optional
        If provided, the record is persisted after ``run()`` completes.
    tags : Dict[str, str], optional
        User-defined labels.

    Raises
    ------
    ImportError
        If ``grdl_rt`` is not installed.  grdl-runtime is required for
        workflow stress testing.

    Examples
    --------
    >>> from grdl_rt import Workflow
    >>> from grdl.image_processing.intensity import ToDecibels
    >>> from grdl_te.benchmarking import WorkflowStressTester, StressTestConfig
    >>>
    >>> wf = Workflow("SAR pipeline").step(ToDecibels())
    >>> tester = WorkflowStressTester(wf, prefer_gpu=False)
    >>> config = StressTestConfig(max_concurrency=4, ramp_steps=3)
    >>> record = tester.run(config)
    >>> print(record.summary.max_sustained_concurrency)
    """

    def __init__(
        self,
        workflow: Any,
        name: Optional[str] = None,
        *,
        prefer_gpu: bool = False,
        version: str = "0.0.0",
        related_benchmark_id: Optional[str] = None,
        store: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        if not _HAS_RUNTIME:
            raise ImportError(
                "grdl_rt is required for WorkflowStressTester. "
                "Install grdl-runtime or use ComponentStressTester instead."
            )
        super().__init__(
            version=version,
            related_benchmark_id=related_benchmark_id,
            store=store,
            tags=tags,
        )
        self._workflow = workflow
        self._prefer_gpu = prefer_gpu
        self._name: str = name if name is not None else getattr(workflow, "name", "workflow")

    @property
    def component_name(self) -> str:
        """Workflow name used in the stress test record.

        Returns
        -------
        str
        """
        return self._name

    def call_once(self, payload: np.ndarray) -> Any:
        """Execute the workflow once against *payload*.

        Calls ``self._workflow.execute(payload, prefer_gpu=...)`` and
        returns ``result.result`` (the processed output array).

        Parameters
        ----------
        payload : np.ndarray
            Input array for this execution.

        Returns
        -------
        Any
            The ``WorkflowResult.result`` from the execution.
        """
        result = self._workflow.execute(payload, prefer_gpu=self._prefer_gpu)
        return result.result
