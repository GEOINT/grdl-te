# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GRDL-TE test suite.

Provides fixtures for locating real-world data files, handling missing data
gracefully, and common test utilities for IO reader validation.

Author
------
Duane Smalley
duane.d.smalley@gmail.com

Steven Siebert

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
from pathlib import Path
from typing import Optional

# Third-party
import pytest


# Data directory relative to repository root
DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def data_dir():
    """Root data directory for all test files."""
    return DATA_DIR


@pytest.fixture(scope="session")
def landsat_data_dir(data_dir):
    """Landsat test data directory."""
    return data_dir / "landsat"


@pytest.fixture(scope="session")
def viirs_data_dir(data_dir):
    """VIIRS test data directory."""
    return data_dir / "viirs"


@pytest.fixture(scope="session")
def sentinel2_data_dir(data_dir):
    """Sentinel-2 test data directory."""
    return data_dir / "sentinel2"


@pytest.fixture(scope="session")
def umbra_data_dir(data_dir):
    """Umbra SICD test data directory."""
    return data_dir / "umbra"


def find_data_file(directory: Path, pattern: str) -> Optional[Path]:
    """
    Find first file matching pattern in directory.

    Parameters
    ----------
    directory : Path
        Directory to search
    pattern : str
        Glob pattern (e.g., '*.h5', 'MOD09*.hdf')

    Returns
    -------
    Optional[Path]
        Path to first matching file, or None if not found
    """
    if not directory.exists():
        return None

    matches = list(directory.glob(pattern))
    return matches[0] if matches else None


def require_data_file(directory: Path, pattern: str,
                      readme_name: str = "README.md") -> Path:
    """
    Require a data file exists, otherwise skip test with helpful message.

    Parameters
    ----------
    directory : Path
        Directory to search
    pattern : str
        Glob pattern (e.g., '*.h5', 'MOD09*.hdf')
    readme_name : str
        Name of README file with download instructions

    Returns
    -------
    Path
        Path to first matching file

    Raises
    ------
    pytest.skip
        If file not found, with message pointing to README
    """
    file_path = find_data_file(directory, pattern)

    if file_path is None:
        readme_path = directory / readme_name
        readme_exists = readme_path.exists()

        msg = (
            f"Data file '{pattern}' not found in {directory}. "
            f"Download instructions: {readme_path if readme_exists else 'see data/ folder'}"
        )
        pytest.skip(msg)

    return file_path


@pytest.fixture
def require_landsat_file(landsat_data_dir):
    """Landsat 8/9 Surface Reflectance COG file."""
    return require_data_file(landsat_data_dir, "LC0[89]*_SR_B*.TIF")


@pytest.fixture
def require_viirs_file(viirs_data_dir):
    """VIIRS VNP09GA HDF5 file."""
    return require_data_file(viirs_data_dir, "V?P09GA*.h5")


@pytest.fixture
def require_sentinel2_file(sentinel2_data_dir):
    """Sentinel-2 JP2 file (standalone or within SAFE structure)."""
    # Try standalone JP2 first
    file_path = find_data_file(sentinel2_data_dir, "T*_B*.jp2")
    if file_path:
        return file_path

    # Try within SAFE structure
    safe_dirs = list(sentinel2_data_dir.glob("S2*.SAFE"))
    for safe_dir in safe_dirs:
        jp2_files = list(safe_dir.glob("**/IMG_DATA/**/*_B04*.jp2"))
        if jp2_files:
            return jp2_files[0]

    # Not found - skip with helpful message
    readme_path = sentinel2_data_dir / "README.md"
    msg = (
        f"No Sentinel-2 JP2 files found in {sentinel2_data_dir}. "
        f"Download instructions: {readme_path}"
    )
    pytest.skip(msg)


@pytest.fixture
def require_umbra_file(umbra_data_dir):
    """Umbra SICD NITF file."""
    return require_data_file(umbra_data_dir, "*.nitf")
