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
DATA_DIR = Path(__file__).parent.parent.parent / "data"


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


@pytest.fixture(scope="session")
def cphd_data_dir(data_dir):
    """CPHD phase history data directory."""
    return data_dir / "cphd"


@pytest.fixture(scope="session")
def crsd_data_dir(data_dir):
    """CRSD data directory."""
    return data_dir / "crsd"


@pytest.fixture(scope="session")
def sidd_data_dir(data_dir):
    """SIDD detected imagery data directory."""
    return data_dir / "sidd"


@pytest.fixture(scope="session")
def sentinel1_data_dir(data_dir):
    """Sentinel-1 SLC data directory."""
    return data_dir / "sentinel1"


@pytest.fixture(scope="session")
def aster_data_dir(data_dir):
    """ASTER L1T data directory."""
    return data_dir / "aster"


@pytest.fixture(scope="session")
def biomass_data_dir(data_dir):
    """BIOMASS L1 data directory."""
    return data_dir / "biomass"


@pytest.fixture(scope="session")
def dted_data_dir(data_dir):
    """DTED elevation tile directory."""
    return data_dir / "dted"


@pytest.fixture(scope="session")
def dem_data_dir(data_dir):
    """GeoTIFF DEM data directory."""
    return data_dir / "dem"


@pytest.fixture(scope="session")
def geoid_data_dir(data_dir):
    """Geoid model data directory."""
    return data_dir / "geoid"


@pytest.fixture(scope="session")
def terrasar_data_dir(data_dir):
    """TerraSAR-X / TanDEM-X product data directory."""
    return data_dir / "terrasar"


@pytest.fixture(scope="session")
def nisar_data_dir(data_dir):
    """NISAR HDF5 SAR data directory."""
    return data_dir / "nisar"


@pytest.fixture(scope="session")
def eo_nitf_data_dir(data_dir):
    """EO NITF (RPC/RSM) data directory."""
    return data_dir / "eo_nitf"


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


def require_data_dir(directory: Path, pattern: str,
                     readme_name: str = "README.md") -> Path:
    """
    Require a subdirectory matching pattern exists, otherwise skip test.

    Parameters
    ----------
    directory : Path
        Parent directory to search
    pattern : str
        Glob pattern for subdirectory name (e.g., 'BIO_S*')
    readme_name : str
        Name of README file with download instructions

    Returns
    -------
    Path
        Path to first matching subdirectory

    Raises
    ------
    pytest.skip
        If no matching directory found, with message pointing to README
    """
    if directory.exists():
        matches = [p for p in directory.glob(pattern) if p.is_dir()]
        if matches:
            return matches[0]

    readme_path = directory / readme_name
    readme_exists = readme_path.exists()
    msg = (
        f"Data directory '{pattern}' not found in {directory}. "
        f"Download instructions: {readme_path if readme_exists else 'see data/ folder'}"
    )
    pytest.skip(msg)


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


@pytest.fixture
def require_cphd_file(cphd_data_dir):
    """CPHD phase history file."""
    return require_data_file(cphd_data_dir, "*.cphd")


@pytest.fixture
def require_crsd_file(crsd_data_dir):
    """CRSD data file."""
    return require_data_file(crsd_data_dir, "*.crsd")


@pytest.fixture
def require_sidd_file(sidd_data_dir):
    """SIDD NITF file."""
    result = find_data_file(sidd_data_dir, "*.nitf")
    if result is None:
        result = find_data_file(sidd_data_dir, "*.ntf")
    if result is None:
        readme_path = sidd_data_dir / "README.md"
        msg = (
            f"Data file '*.nitf or *.ntf' not found in {sidd_data_dir}. "
            f"Download instructions: {readme_path if readme_path.exists() else 'see data/ folder'}"
        )
        pytest.skip(msg)
    return result


@pytest.fixture
def require_sentinel1_file(sentinel1_data_dir):
    """Sentinel-1 SLC SAFE directory."""
    return require_data_file(sentinel1_data_dir, "*.SAFE")


@pytest.fixture
def require_aster_file(aster_data_dir):
    """ASTER L1T GeoTIFF file."""
    return require_data_file(aster_data_dir, "AST_L1T*.tif")


@pytest.fixture
def require_biomass_file(biomass_data_dir):
    """BIOMASS L1 GeoTIFF file."""
    return require_data_dir(biomass_data_dir, "BIO_S*")


@pytest.fixture
def require_dted_dir(dted_data_dir):
    """DTED tile directory with at least one .dt? file."""
    if not dted_data_dir.exists() or not list(dted_data_dir.glob("**/*.dt?")):
        pytest.skip(f"DTED data not found in {dted_data_dir}")
    return dted_data_dir


@pytest.fixture
def require_dem_file(dem_data_dir):
    """GeoTIFF DEM file."""
    return require_data_file(dem_data_dir, "*.tif")


@pytest.fixture
def require_geoid_file(geoid_data_dir):
    """EGM96 geoid model file."""
    return require_data_file(geoid_data_dir, "*.pgm")


@pytest.fixture
def require_nisar_file(nisar_data_dir):
    """NISAR RSLC/GSLC HDF5 file."""
    return require_data_file(nisar_data_dir, "NISAR*.h5")


@pytest.fixture
def require_eo_nitf_file(eo_nitf_data_dir):
    """EO NITF file with RPC/RSM geolocation."""
    result = find_data_file(eo_nitf_data_dir, "*.ntf")
    if result is None:
        result = find_data_file(eo_nitf_data_dir, "*.nitf")
    if result is None:
        readme_path = eo_nitf_data_dir / "README.md"
        msg = (
            f"Data file '*.ntf or *.nitf' not found in {eo_nitf_data_dir}. "
            f"Download instructions: {readme_path if readme_path.exists() else 'see data/ folder'}"
        )
        pytest.skip(msg)
    return result


@pytest.fixture
def require_terrasar_dir(terrasar_data_dir):
    """TerraSAR-X product directory (TSX1_* or TDX1_*)."""
    if not terrasar_data_dir.exists():
        pytest.skip(f"TerraSAR data directory not found: {terrasar_data_dir}")
    for candidate in sorted(terrasar_data_dir.iterdir()):
        if candidate.is_dir() and (
            candidate.name.startswith("TSX1_")
            or candidate.name.startswith("TDX1_")
        ):
            return candidate
    # Check for annotation XMLs directly
    xmls = sorted(terrasar_data_dir.glob("TSX1_SAR__*.xml"))
    if xmls:
        return terrasar_data_dir
    pytest.skip(f"No TSX1_*/TDX1_* product found in {terrasar_data_dir}")
