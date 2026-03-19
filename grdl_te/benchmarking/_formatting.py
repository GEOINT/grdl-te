# -*- coding: utf-8 -*-
"""
Shared formatting helpers for benchmark report presenters.

Pure display utilities — no data derivation.  All three report formats
(text, Markdown, GUI) and the comparison module import from here to
avoid duplicating byte/time/throughput formatting logic.

Dependencies
------------
(none — standard library only)

Author
------
Claude Code (Anthropic)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-19

Modified
--------
2026-03-19
"""

# Standard library
from typing import Optional


# ---------------------------------------------------------------------------
# Byte formatting
# ---------------------------------------------------------------------------
def fmt_bytes(value_bytes: float) -> str:
    """Format a byte count as a human-readable string.

    Parameters
    ----------
    value_bytes : float
        Value in bytes.

    Returns
    -------
    str
        Formatted string with unit suffix (B, KB, MB, or GB).
    """
    abs_val = abs(value_bytes)
    if abs_val < 1024:
        return f"{value_bytes:.0f} B"
    if abs_val < 1024 ** 2:
        return f"{value_bytes / 1024:.1f} KB"
    if abs_val < 1024 ** 3:
        return f"{value_bytes / 1024 ** 2:.1f} MB"
    return f"{value_bytes / 1024 ** 3:.2f} GB"


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    """Format a duration in seconds with appropriate precision.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string with ``s`` suffix.  Sub-second values get four
        decimal places; values >= 1 s get two.
    """
    if abs(seconds) < 1.0:
        return f"{seconds:.4f}s"
    return f"{seconds:.2f}s"


# ---------------------------------------------------------------------------
# Throughput formatting
# ---------------------------------------------------------------------------
def fmt_throughput(elements_per_sec: Optional[float]) -> str:
    """Format throughput as a human-readable string (e.g., ``26.2 Mpx/s``).

    Parameters
    ----------
    elements_per_sec : float or None
        Elements processed per second.

    Returns
    -------
    str
        ``"--"`` when *elements_per_sec* is ``None``.
    """
    if elements_per_sec is None:
        return "--"
    if elements_per_sec >= 1e9:
        return f"{elements_per_sec / 1e9:.1f} Gpx/s"
    if elements_per_sec >= 1e6:
        return f"{elements_per_sec / 1e6:.1f} Mpx/s"
    if elements_per_sec >= 1e3:
        return f"{elements_per_sec / 1e3:.1f} Kpx/s"
    return f"{elements_per_sec:.1f} px/s"


# ---------------------------------------------------------------------------
# Name shortening
# ---------------------------------------------------------------------------
def short_name(processor_name: str) -> str:
    """Extract the short processor name from a dotted path.

    Parameters
    ----------
    processor_name : str
        Fully-qualified processor name (e.g., ``grdl.filters.Gaussian``).

    Returns
    -------
    str
        Last segment after the final dot (e.g., ``Gaussian``).
    """
    return processor_name.rsplit(".", 1)[-1]
