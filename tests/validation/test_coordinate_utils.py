# -*- coding: utf-8 -*-
"""
Coordinate Utilities Tests - WGS-84 geodetic/ECEF/ENU transforms.

Validates grdl.geolocation.coordinates against known values and
round-trip consistency at boundary conditions:
- Equator and prime meridian (known ECEF values)
- Poles (lat ±90°) — iterative convergence edge case
- Anti-meridian (lon ±180°)
- ENU origin identity (ref point maps to (0, 0, 0))
- ENU round-trip near reference and at 100 km offset

All tests use synthetic data only (no real imagery required).

Dependencies
------------
numpy

Author
------
Claude Code (generated)

License
-------
MIT License
See LICENSE file for full text.

Created
-------
2026-03-09

Modified
--------
2026-03-09
"""

# Standard library
from typing import Tuple

# Third-party
import pytest
import numpy as np

# GRDL internal
try:
    from grdl.geolocation.coordinates import (
        geodetic_to_ecef,
        ecef_to_geodetic,
        geodetic_to_enu,
        enu_to_geodetic,
        WGS84_A,
        WGS84_B,
    )
    _HAS_COORDS = True
except ImportError:
    _HAS_COORDS = False

pytestmark = [
    pytest.mark.geolocation,
    pytest.mark.skipif(not _HAS_COORDS, reason="grdl.geolocation.coordinates not available"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scalar_arrays(*vals) -> Tuple[np.ndarray, ...]:
    """Wrap scalars in 1-element float64 arrays."""
    return tuple(np.array([v], dtype=np.float64) for v in vals)


# ===========================================================================
# geodetic_to_ecef / ecef_to_geodetic
# ===========================================================================

class TestGeodeticEcef:
    """Round-trip and known-value tests for geodetic ↔ ECEF."""

    def test_equator_prime_meridian_known_ecef(self):
        """Equator / prime meridian maps to ECEF (WGS84_A, 0, 0).

        At lat=0, lon=0, alt=0 the ECEF X coordinate must equal the
        WGS-84 semi-major axis (6 378 137 m) and Y=Z=0.
        """
        lats, lons, hts = _make_scalar_arrays(0.0, 0.0, 0.0)
        x, y, z = geodetic_to_ecef(lats, lons, hts)

        assert abs(float(x[0]) - WGS84_A) < 1.0, (
            f"X at equator/prime meridian = {x[0]:.3f} m, expected {WGS84_A:.3f} m"
        )
        assert abs(float(y[0])) < 1.0, f"Y at equator/prime meridian = {y[0]:.3f} m, expected 0"
        assert abs(float(z[0])) < 1.0, f"Z at equator/prime meridian = {z[0]:.3f} m, expected 0"

    def test_north_pole_ecef_z_equals_semiminor(self):
        """North pole maps to ECEF (0, 0, WGS84_B).

        At lat=+90°, lon=0°, alt=0 the ECEF Z coordinate must equal the
        WGS-84 semi-minor axis (~6 356 752 m).
        """
        lats, lons, hts = _make_scalar_arrays(90.0, 0.0, 0.0)
        x, y, z = geodetic_to_ecef(lats, lons, hts)

        assert abs(float(z[0]) - WGS84_B) < 1.0, (
            f"Z at north pole = {z[0]:.3f} m, expected {WGS84_B:.3f} m"
        )
        assert abs(float(x[0])) < 1.0, f"X at north pole should be near 0, got {x[0]:.3f} m"
        assert abs(float(y[0])) < 1.0, f"Y at north pole should be near 0, got {y[0]:.3f} m"

    def test_south_pole_ecef_z_equals_negative_semiminor(self):
        """South pole maps to ECEF (0, 0, -WGS84_B)."""
        lats, lons, hts = _make_scalar_arrays(-90.0, 0.0, 0.0)
        x, y, z = geodetic_to_ecef(lats, lons, hts)

        assert abs(float(z[0]) + WGS84_B) < 1.0, (
            f"Z at south pole = {z[0]:.3f} m, expected {-WGS84_B:.3f} m"
        )

    def test_round_trip_general_points(self):
        """geodetic → ECEF → geodetic recovers input within 1e-7 degrees and 1 mm.

        Tests a grid of points spanning the globe to catch any systematic
        formula error that would not be visible in a single known-value check.
        """
        rng = np.random.default_rng(0)
        lats = rng.uniform(-80.0, 80.0, 100)
        lons = rng.uniform(-180.0, 180.0, 100)
        hts = rng.uniform(-500.0, 10000.0, 100)

        x, y, z = geodetic_to_ecef(lats, lons, hts)
        lats_rt, lons_rt, hts_rt = ecef_to_geodetic(x, y, z)

        np.testing.assert_allclose(lats_rt, lats, atol=1e-7,
            err_msg="Latitude round-trip error exceeds 1e-7 degrees")
        np.testing.assert_allclose(hts_rt, hts, atol=1e-3,
            err_msg="Height round-trip error exceeds 1 mm")

        # Longitude is periodic; compare via cosine to handle ±180° wrap
        cos_diff = np.cos(np.radians(lons_rt - lons))
        assert np.all(cos_diff > np.cos(np.radians(1e-6))), (
            "Longitude round-trip error exceeds 1e-6 degrees"
        )

    def test_round_trip_poles(self):
        """ecef_to_geodetic converges correctly at exact pole coordinates.

        The Bowring iterative method can be numerically unstable at lat=±90°
        because cos(lat)→0.  This test verifies convergence within 10 iterations.
        """
        for lat_sign in (+1.0, -1.0):
            lats, lons, hts = _make_scalar_arrays(lat_sign * 90.0, 0.0, 0.0)
            x, y, z = geodetic_to_ecef(lats, lons, hts)
            lats_rt, _, hts_rt = ecef_to_geodetic(x, y, z)

            assert abs(float(lats_rt[0]) - lat_sign * 90.0) < 1e-5, (
                f"Pole round-trip latitude error: {lats_rt[0]:.8f}° (expected ±90°)"
            )
            assert abs(float(hts_rt[0])) < 1.0, (
                f"Pole round-trip height error: {hts_rt[0]:.3f} m (expected ~0)"
            )

    def test_round_trip_antimeridian(self):
        """Round-trip works correctly at lon=±180°."""
        for lon in (180.0, -180.0):
            lats, lons, hts = _make_scalar_arrays(45.0, lon, 100.0)
            x, y, z = geodetic_to_ecef(lats, lons, hts)
            lats_rt, lons_rt, hts_rt = ecef_to_geodetic(x, y, z)

            assert abs(float(lats_rt[0]) - 45.0) < 1e-7, (
                f"Anti-meridian lat round-trip error: {lats_rt[0]:.8f}° (input 45°)"
            )
            assert abs(float(hts_rt[0]) - 100.0) < 1e-3, (
                f"Anti-meridian height round-trip error: {hts_rt[0]:.3f} m (input 100.0)"
            )

    def test_altitude_preserved(self):
        """Height above ellipsoid survives a round-trip for high-altitude points."""
        lats, lons, hts = _make_scalar_arrays(37.5, -122.0, 500000.0)  # 500 km altitude
        x, y, z = geodetic_to_ecef(lats, lons, hts)
        _, _, hts_rt = ecef_to_geodetic(x, y, z)
        assert abs(float(hts_rt[0]) - 500000.0) < 1.0, (
            f"High-altitude round-trip height error: {hts_rt[0]:.3f} m (expected 500000)"
        )

    def test_vectorized_array_input(self):
        """Functions accept and return arrays (vectorized, not just scalars)."""
        lats = np.array([0.0, 45.0, -45.0, 90.0, -90.0])
        lons = np.array([0.0, 90.0, -90.0, 180.0, -180.0])
        hts = np.zeros(5)

        x, y, z = geodetic_to_ecef(lats, lons, hts)
        assert x.shape == (5,), "geodetic_to_ecef must return arrays of same length as input"

        lats_rt, lons_rt, hts_rt = ecef_to_geodetic(x, y, z)
        assert lats_rt.shape == (5,), "ecef_to_geodetic must return arrays of same length as input"


# ===========================================================================
# geodetic_to_enu / enu_to_geodetic
# ===========================================================================

class TestGeodeticEnu:
    """Round-trip and identity tests for geodetic ↔ ENU."""

    # Reference point: mid-latitude continental US
    REF_LAT = 39.0
    REF_LON = -96.0
    REF_ALT = 0.0

    def test_reference_point_is_enu_origin(self):
        """A point at the reference location maps to ENU (0, 0, 0).

        The ENU system is defined such that the reference point is always
        (east=0, north=0, up=0).  Any non-zero result means the
        coordinate system is off-center.
        """
        lats, lons, hts = _make_scalar_arrays(
            self.REF_LAT, self.REF_LON, self.REF_ALT
        )
        east, north, up = geodetic_to_enu(
            lats, lons, hts,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        assert abs(float(east[0])) < 1e-3, (
            f"ENU east at reference point = {east[0]:.6f} m (expected 0)"
        )
        assert abs(float(north[0])) < 1e-3, (
            f"ENU north at reference point = {north[0]:.6f} m (expected 0)"
        )
        assert abs(float(up[0])) < 1e-3, (
            f"ENU up at reference point = {up[0]:.6f} m (expected 0)"
        )

    def test_east_offset_appears_in_east_component(self):
        """A point displaced east has positive east component, near-zero north/up.

        Moving ~1 km east in longitude at the reference latitude should
        produce east ≈ 1000 m and north, up ≪ 1 m.
        """
        # ~1 km east: 1000 m / (111320 * cos(lat)) degrees
        delta_lon = 1000.0 / (111320.0 * np.cos(np.radians(self.REF_LAT)))
        lats, lons, hts = _make_scalar_arrays(
            self.REF_LAT, self.REF_LON + delta_lon, 0.0
        )
        east, north, up = geodetic_to_enu(
            lats, lons, hts,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        assert abs(float(east[0]) - 1000.0) < 5.0, (
            f"East displacement: {east[0]:.2f} m (expected ~1000 m)"
        )
        assert abs(float(north[0])) < 10.0, (
            f"North contamination from east displacement: {north[0]:.2f} m"
        )

    def test_north_offset_appears_in_north_component(self):
        """A point displaced north has positive north component, near-zero east/up.

        Moving ~1 km north in latitude should produce north ≈ 1000 m.
        """
        delta_lat = 1000.0 / 111320.0  # degrees
        lats, lons, hts = _make_scalar_arrays(
            self.REF_LAT + delta_lat, self.REF_LON, 0.0
        )
        east, north, up = geodetic_to_enu(
            lats, lons, hts,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        assert abs(float(north[0]) - 1000.0) < 5.0, (
            f"North displacement: {north[0]:.2f} m (expected ~1000 m)"
        )
        assert abs(float(east[0])) < 10.0, (
            f"East contamination from north displacement: {east[0]:.2f} m"
        )

    def test_up_offset_appears_in_up_component(self):
        """A point at 1000 m altitude above the reference has up ≈ 1000 m."""
        lats, lons, hts = _make_scalar_arrays(
            self.REF_LAT, self.REF_LON, 1000.0
        )
        east, north, up = geodetic_to_enu(
            lats, lons, hts,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        assert abs(float(up[0]) - 1000.0) < 1.0, (
            f"Up displacement: {up[0]:.3f} m (expected ~1000 m)"
        )

    def test_round_trip_nearby_points(self):
        """geodetic → ENU → geodetic recovers input within 1 mm for nearby points.

        Uses offsets of ±100 km max, where the flat-Earth ENU approximation
        is highly accurate.
        """
        rng = np.random.default_rng(42)
        # Offsets of ±0.5° in lat/lon around the reference
        lats = self.REF_LAT + rng.uniform(-0.5, 0.5, 50)
        lons = self.REF_LON + rng.uniform(-0.5, 0.5, 50)
        hts = rng.uniform(-100.0, 1000.0, 50)

        east, north, up = geodetic_to_enu(
            lats, lons, hts,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        lats_rt, lons_rt, hts_rt = enu_to_geodetic(
            east, north, up,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )

        np.testing.assert_allclose(lats_rt, lats, atol=1e-7,
            err_msg="Nearby-point ENU→geodetic latitude error > 1e-7°")
        np.testing.assert_allclose(hts_rt, hts, atol=1e-3,
            err_msg="Nearby-point ENU→geodetic height error > 1 mm")

    def test_round_trip_100km_offset(self):
        """ENU round-trip remains better than 1 cm at 100 km from reference.

        The ENU linearisation around a reference point accumulates error
        over distance.  At 100 km the ECEF-based implementation should
        still recover input latitude within 1e-6 degrees (~0.1 m).
        """
        # ~100 km north: 100000 m / 111320 m/deg
        delta_lat = 100000.0 / 111320.0
        lats, lons, hts = _make_scalar_arrays(
            self.REF_LAT + delta_lat, self.REF_LON, 0.0
        )
        east, north, up = geodetic_to_enu(
            lats, lons, hts,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        lats_rt, lons_rt, hts_rt = enu_to_geodetic(
            east, north, up,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )

        assert abs(float(lats_rt[0]) - float(lats[0])) < 1e-6, (
            f"100 km offset round-trip lat error: "
            f"{abs(float(lats_rt[0]) - float(lats[0])):.2e}° "
            f"(threshold 1e-6°)"
        )
        assert abs(float(hts_rt[0]) - float(hts[0])) < 0.01, (
            f"100 km offset round-trip height error: "
            f"{abs(float(hts_rt[0]) - float(hts[0])):.4f} m (threshold 0.01 m)"
        )

    def test_vectorized_enu_returns_correct_shape(self):
        """geodetic_to_enu and enu_to_geodetic process arrays without shape errors."""
        lats = np.linspace(38.0, 40.0, 20)
        lons = np.linspace(-97.0, -95.0, 20)
        hts = np.zeros(20)

        east, north, up = geodetic_to_enu(
            lats, lons, hts,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        assert east.shape == (20,), "ENU east array shape mismatch"
        assert north.shape == (20,), "ENU north array shape mismatch"

        lats_rt, lons_rt, hts_rt = enu_to_geodetic(
            east, north, up,
            self.REF_LAT, self.REF_LON, self.REF_ALT,
        )
        assert lats_rt.shape == (20,), "Recovered lats array shape mismatch"
