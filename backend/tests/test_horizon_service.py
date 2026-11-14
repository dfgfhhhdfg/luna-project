"""
LUNA NAVIGATOR — Unit & Regression Tests for Horizon & Line-of-Sight Service (Member 3 Ownership)

Validates the 10 core scientific scenarios:
1. Flat terrain with no obstruction (returns clear).
2. Synthetic ridge that obstructs known direction (returns obstructed with correct distance).
3. Ridge that does not obstruct a higher target (returns clear).
4. Multiple azimuth directions (full profile).
5. NoData along ray (returns unknown if target above evaluated ridge).
6. Ray leaving DEM boundary before max distance (returns unknown).
7. Observer location outside DEM coverage (returns unknown).
8. Curvature drop verification at increasing distances (Delta_h = s^2 / 2R).
9. Azimuth wraparound (359° to 1° through North).
10. Analytically predictable geometric cases.
Additional tests:
11. Convenience methods on DemRaster.
12. API endpoints: /terrain/ray, /terrain/check-los, and /terrain/dem-horizon.
"""

import math
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.core.constants import MOON_MEAN_RADIUS_M
from app.services.geospatial.dem_loader import DemRaster, DemMetadata
from app.services.geospatial.horizon_service import (
    calculate_curvature_drop,
    planar_ray_waypoints,
    geodesic_ray_waypoints,
    sample_elevations_vectorized,
    sample_terrain_ray,
    check_line_of_sight,
    calculate_horizon_profile,
    get_horizon_elevation_at_azimuth,
    calculate_solar_irradiance,
    calculate_solar_disk_visibility,
    evaluate_solar_occultation_and_flux,
    calculate_fresnel_clearance
)


@pytest.fixture
def flat_dem() -> DemRaster:
    """Create a 100x100 flat DEM with 100m pixels centered at (0, 0), elevation = 1000.0m."""
    h, w = 100, 100
    elev = np.full((h, w), 1000.0, dtype=np.float32)
    meta = DemMetadata(
        dataset_name="flat_test_dem",
        data_source="Synthetic Test Fixture",
        file_path="memory://flat.tif",
        width_pixels=w,
        height_pixels=h,
        pixel_size_x_m=100.0,
        pixel_size_y_m=100.0,
        crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
        bounds_projected=(-5000.0, -5000.0, 5000.0, 5000.0),
        elevation_units="meters",
        reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
        nodata_value=-9999.0,
        valid_data_percent=100.0,
        min_elevation_m=1000.0,
        max_elevation_m=1000.0,
        mean_elevation_m=1000.0,
        is_synthetic=True
    )
    return DemRaster(elevation_array=elev, metadata=meta)


@pytest.fixture
def ridge_dem() -> DemRaster:
    """
    Create a 100x100 DEM centered at (0, 0) with:
    - Base elevation = 500.0m
    - Prominent East-West ridge located at y = 1000.0m (North of center):
      Row corresponding to y in [900, 1100] has elevation = 1000.0m (+500m prominence).
    """
    h, w = 100, 100
    elev = np.full((h, w), 500.0, dtype=np.float32)
    
    # In bounds (-5000, -5000, 5000, 5000):
    # y = top - (row + 0.5) * dy => 5000 - (row + 0.5) * 100
    # For y = 1000: row ~ (5000 - 1000) / 100 = 40.
    # Set rows 39, 40, 41 to 1000.0m
    elev[39:42, :] = 1000.0

    meta = DemMetadata(
        dataset_name="ridge_test_dem",
        data_source="Synthetic Test Fixture",
        file_path="memory://ridge.tif",
        width_pixels=w,
        height_pixels=h,
        pixel_size_x_m=100.0,
        pixel_size_y_m=100.0,
        crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
        bounds_projected=(-5000.0, -5000.0, 5000.0, 5000.0),
        elevation_units="meters",
        reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
        nodata_value=-9999.0,
        valid_data_percent=100.0,
        min_elevation_m=500.0,
        max_elevation_m=1000.0,
        mean_elevation_m=515.0,
        is_synthetic=True
    )
    return DemRaster(elevation_array=elev, metadata=meta)


@pytest.fixture
def nodata_dem() -> DemRaster:
    """Create a DEM with NoData cells introduced in a northern band."""
    h, w = 100, 100
    elev = np.full((h, w), 500.0, dtype=np.float32)
    # Put NoData cells at y > 1500m (rows 0 to 34)
    elev[0:35, :] = -9999.0

    meta = DemMetadata(
        dataset_name="nodata_test_dem",
        data_source="Synthetic Test Fixture",
        file_path="memory://nodata.tif",
        width_pixels=w,
        height_pixels=h,
        pixel_size_x_m=100.0,
        pixel_size_y_m=100.0,
        crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
        bounds_projected=(-5000.0, -5000.0, 5000.0, 5000.0),
        elevation_units="meters",
        reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
        nodata_value=-9999.0,
        valid_data_percent=65.0,
        min_elevation_m=500.0,
        max_elevation_m=500.0,
        mean_elevation_m=500.0,
        is_synthetic=True
    )
    return DemRaster(elevation_array=elev, metadata=meta)


# =============================================================================
# SCENARIO 1: Flat terrain with no obstruction (returns clear)
# =============================================================================

def test_scenario_1_flat_terrain_clear(flat_dem):
    """
    On a flat DEM with observer at height 2.0m, terrain angles are always negative
    (pointing downwards due to observer eye height and lunar curvature).
    Target at elevation >= 0.0° must return 'clear' out to max distance.
    """
    res = check_line_of_sight(
        dem=flat_dem,
        observer_coord=(0.0, 0.0),
        target_azimuth_deg=0.0,
        target_elevation_deg=0.5,
        max_distance_m=3000.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    assert res["visibility_state"] == "clear"
    assert res["is_obstructed"] is False
    assert res["elevation_margin_deg"] > 0.0
    assert res["obstructing_distance_m"] is None
    assert res["evaluated_distance_m"] == 3000.0


# =============================================================================
# SCENARIO 2: Synthetic ridge that obstructs known direction
# =============================================================================

def test_scenario_2_ridge_obstructs_known_direction(ridge_dem):
    """
    Observer at (0, 0) looking North (azimuth 0.0°).
    A ridge with elevation = 1000m exists at y ~ 1000m (+500m relative height).
    Elevation angle ~ arctan(500 / 1000) ~ 26.5°.
    Target at elevation = 5.0° must be obstructed at distance ~ 1000m.
    """
    res = check_line_of_sight(
        dem=ridge_dem,
        observer_coord=(0.0, 0.0),
        target_azimuth_deg=0.0,
        target_elevation_deg=5.0,
        max_distance_m=3000.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    assert res["visibility_state"] == "obstructed"
    assert res["is_obstructed"] is True
    assert 800.0 <= res["obstructing_distance_m"] <= 1100.0
    assert 900.0 <= res["highest_point"]["distance_m"] <= 1100.0
    assert res["obstructing_point"] is not None
    assert res["elevation_margin_deg"] <= 0.0


# =============================================================================
# SCENARIO 3: Ridge that does not obstruct a higher target (returns clear)
# =============================================================================

def test_scenario_3_ridge_does_not_obstruct_high_target(ridge_dem):
    """
    Same ridge has peak angle ~ 26.5°.
    Target at elevation = 45.0° is well above the ridge crest and must return 'clear'.
    """
    res = check_line_of_sight(
        dem=ridge_dem,
        observer_coord=(0.0, 0.0),
        target_azimuth_deg=0.0,
        target_elevation_deg=45.0,
        max_distance_m=3000.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    assert res["visibility_state"] == "clear"
    assert res["is_obstructed"] is False
    assert res["elevation_margin_deg"] > 15.0
    assert res["obstructing_distance_m"] is None
    assert res["evaluated_distance_m"] == 3000.0


# =============================================================================
# SCENARIO 4: Multiple azimuth directions (full profile)
# =============================================================================

def test_scenario_4_multiple_azimuth_profile(ridge_dem):
    """
    Compute full 360° horizon profile around (0, 0) with 10° steps (36 points).
    Azimuth 0° (North) intersects the ridge, while Azimuth 180° (South) is flat terrain.
    Max horizon elevation must occur around 0° (North).
    """
    profile = calculate_horizon_profile(
        dem=ridge_dem,
        observer_coord=(0.0, 0.0),
        max_distance_m=2500.0,
        azimuth_step_deg=10.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    assert len(profile["horizon_points"]) == 36
    assert profile["max_horizon_elevation_deg"] > 20.0
    assert profile["azimuth_of_max_elevation_deg"] == 0.0
    # Azimuth 180° (South) has flat terrain (negative elevation angle)
    south_pt = next(p for p in profile["horizon_points"] if p["azimuth_deg"] == 180.0)
    assert south_pt["horizon_elevation_deg"] < 0.0


# =============================================================================
# SCENARIO 5: NoData along ray (returns unknown if target above evaluated ridge)
# =============================================================================

def test_scenario_5_nodata_returns_unknown(nodata_dem):
    """
    Ray heads North into a NoData band starting at y = 1500m.
    Evaluated terrain before 1500m is flat (no obstruction for target at 5.0°).
    Because the ray enters NoData, line-of-sight clearance cannot be proven -> 'unknown'.
    """
    res = check_line_of_sight(
        dem=nodata_dem,
        observer_coord=(0.0, 0.0),
        target_azimuth_deg=0.0,
        target_elevation_deg=5.0,
        max_distance_m=3000.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    assert res["visibility_state"] == "unknown"
    assert res["is_obstructed"] is None
    assert "NoData" in res["reason"]
    assert res["evaluated_distance_m"] <= 1600.0


# =============================================================================
# SCENARIO 6: Ray leaving DEM boundary before max distance (returns unknown)
# =============================================================================

def test_scenario_6_boundary_departure_returns_unknown(flat_dem):
    """
    Observer at (0.0, 4000.0m) (1000m from northern boundary at 5000m).
    Ray heads North (azimuth 0.0°) with max_distance = 3000m.
    Ray exits the DEM at distance = 1000m.
    Target at 5.0° is not obstructed in the first 1000m -> must return 'unknown'.
    """
    res = check_line_of_sight(
        dem=flat_dem,
        observer_coord=(0.0, 4000.0),
        target_azimuth_deg=0.0,
        target_elevation_deg=5.0,
        max_distance_m=3000.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    assert res["visibility_state"] == "unknown"
    assert res["is_obstructed"] is None
    assert "boundary" in res["reason"].lower()
    assert res["evaluated_distance_m"] <= 1100.0


# =============================================================================
# SCENARIO 7: Observer location outside DEM coverage (returns unknown)
# =============================================================================

def test_scenario_7_observer_outside_dem(flat_dem):
    """
    Observer at (10000.0, 10000.0m) is outside DEM bounds (-5000 to 5000).
    Must return 'unknown' with descriptive reason.
    """
    res = check_line_of_sight(
        dem=flat_dem,
        observer_coord=(10000.0, 10000.0),
        target_azimuth_deg=0.0,
        target_elevation_deg=5.0,
        max_distance_m=2000.0,
        mode="projected"
    )
    assert res["visibility_state"] == "unknown"
    assert res["is_obstructed"] is None
    assert "outside" in res["reason"].lower()


# =============================================================================
# SCENARIO 8: Curvature drop verification at increasing distances
# =============================================================================

def test_scenario_8_curvature_drop_verification():
    """
    Verify Delta_h = s^2 / (2 * R_moon) with R_moon = 1,737,400.0m:
    - 10 km -> 28.78m
    - 30 km -> 259.01m
    - 60 km -> 1036.03m
    """
    drop_10km = calculate_curvature_drop(10000.0)
    drop_30km = calculate_curvature_drop(30000.0)
    drop_60km = calculate_curvature_drop(60000.0)

    assert pytest.approx(28.78, abs=0.05) == drop_10km
    assert pytest.approx(259.01, abs=0.1) == drop_30km
    assert pytest.approx(1036.03, abs=0.2) == drop_60km

    # Array input
    arr_dists = np.array([10000.0, 30000.0, 60000.0])
    arr_drops = calculate_curvature_drop(arr_dists)
    assert len(arr_drops) == 3
    assert pytest.approx(1036.03, abs=0.2) == arr_drops[2]

    # Negative distance error
    with pytest.raises(ValueError):
        calculate_curvature_drop(-500.0)


# =============================================================================
# SCENARIO 9: Azimuth wraparound (359° to 1° through North)
# =============================================================================

def test_scenario_9_azimuth_wraparound(flat_dem):
    """
    Test azimuth normalization and continuous periodic interpolation across North:
    - 359° and 1° rays must trace correctly.
    - -5° normalizes to 355°.
    - 365° normalizes to 5°.
    - Periodic interpolation smoothly transitions across 360°/0°.
    """
    ray_359 = sample_terrain_ray(flat_dem, observer_coord=(0.0, 0.0), azimuth_deg=359.0, max_distance_m=1000.0)
    ray_001 = sample_terrain_ray(flat_dem, observer_coord=(0.0, 0.0), azimuth_deg=1.0, max_distance_m=1000.0)
    ray_neg = sample_terrain_ray(flat_dem, observer_coord=(0.0, 0.0), azimuth_deg=-5.0, max_distance_m=1000.0)
    ray_over = sample_terrain_ray(flat_dem, observer_coord=(0.0, 0.0), azimuth_deg=365.0, max_distance_m=1000.0)

    assert ray_359["azimuth_deg"] == 359.0
    assert ray_001["azimuth_deg"] == 1.0
    assert ray_neg["azimuth_deg"] == 355.0
    assert ray_over["azimuth_deg"] == 5.0

    # Test periodic interpolation
    profile_data = {
        "horizon_points": [
            {"azimuth_deg": 0.0, "horizon_elevation_deg": 10.0},
            {"azimuth_deg": 90.0, "horizon_elevation_deg": 5.0},
            {"azimuth_deg": 180.0, "horizon_elevation_deg": 15.0},
            {"azimuth_deg": 270.0, "horizon_elevation_deg": 2.0},
        ]
    }
    el_359 = get_horizon_elevation_at_azimuth(profile_data, 359.0)
    el_001 = get_horizon_elevation_at_azimuth(profile_data, 1.0)
    # Both should be close to 10.0°
    assert abs(el_359 - 10.0) < 0.2
    assert abs(el_001 - 10.0) < 0.2


# =============================================================================
# SCENARIO 10: Analytically predictable geometric cases
# =============================================================================

def test_scenario_10_analytical_geometry():
    """
    Create a synthetic DEM with a single isolated obstacle of height 100.0m
    at exact distance s = 1000.0m along azimuth 90.0° (East).
    Ground is 0.0m, observer height is 0.0m.
    Formula: alpha = arctan((h - s^2/(2*R)) / s)
    At s = 1000m, s^2/(2*R) = 1e6 / (2 * 1737400) = 0.2878m.
    Apparent height = 100 - 0.2878 = 99.7122m.
    Expected alpha = arctan(99.7122 / 1000) = 5.6946°.
    Verify calculation matches analytical value within 0.01°.
    """
    h, w = 100, 100
    elev = np.zeros((h, w), dtype=np.float32)
    # Observer at (0, 0). Azimuth 90° points East (+X).
    # Distance 1000m -> col = 50 + 10 = 60, row = 50.
    elev[50, 60] = 100.0

    meta = DemMetadata(
        dataset_name="analytical_geom_dem",
        data_source="Synthetic Test Fixture",
        file_path="memory://analytical.tif",
        width_pixels=w,
        height_pixels=h,
        pixel_size_x_m=100.0,
        pixel_size_y_m=100.0,
        crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
        bounds_projected=(-5000.0, -5000.0, 5000.0, 5000.0),
        elevation_units="meters",
        reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
        nodata_value=-9999.0,
        valid_data_percent=100.0,
        min_elevation_m=0.0,
        max_elevation_m=100.0,
        mean_elevation_m=0.01,
        is_synthetic=True
    )
    dem = DemRaster(elevation_array=elev, metadata=meta)

    res = sample_terrain_ray(
        dem=dem,
        observer_coord=(0.0, 0.0),
        azimuth_deg=90.0,
        max_distance_m=1500.0,
        radial_step_m=100.0,
        observer_height_m=0.0,
        interpolation_method="nearest",
        mode="projected"
    )

    pt_1000 = next(p for p in res["samples"] if p["distance_m"] == 1000.0)
    expected_curv = 1000.0**2 / (2.0 * MOON_MEAN_RADIUS_M)
    expected_app_h = 100.0 - expected_curv
    expected_angle_deg = math.degrees(math.atan(expected_app_h / 1000.0))

    assert pytest.approx(expected_curv, abs=0.001) == pt_1000["curvature_drop_m"]
    assert pytest.approx(expected_angle_deg, abs=0.01) == pt_1000["elevation_angle_deg"]
    assert pytest.approx(5.69, abs=0.02) == pt_1000["elevation_angle_deg"]


# =============================================================================
# SCENARIO 11: Convenience methods on DemRaster
# =============================================================================

def test_dem_raster_convenience_methods(ridge_dem):
    """Verify sample_terrain_ray, check_line_of_sight, and calculate_horizon_profile on DemRaster."""
    # 1. Projected mode (ridge is at +Y, azimuth 0°)
    ray = ridge_dem.sample_terrain_ray(observer_coord=(0.0, 0.0), azimuth_deg=0.0, max_distance_m=2000.0, mode="projected")
    assert ray["max_horizon_angle_deg"] is not None

    los = ridge_dem.check_line_of_sight(
        target_azimuth_deg=0.0,
        target_elevation_deg=5.0,
        observer_coord=(0.0, 0.0),
        max_distance_m=2000.0,
        mode="projected"
    )
    assert los["visibility_state"] == "obstructed"

    prof = ridge_dem.calculate_horizon_profile(observer_coord=(0.0, 0.0), max_distance_m=2000.0, azimuth_step_deg=45.0, mode="projected")
    assert len(prof["horizon_points"]) == 8

    # 2. Spherical mode (ridge at +Y corresponds to meridian lambda=180°)
    los_sph = ridge_dem.check_line_of_sight(
        target_azimuth_deg=180.0,
        target_elevation_deg=5.0,
        observer_coord=(0.0, 0.0),
        max_distance_m=2000.0,
        mode="spherical"
    )
    assert los_sph["visibility_state"] == "obstructed"


# =============================================================================
# SCENARIO 12: REST API Endpoints Integration Tests
# =============================================================================

client = TestClient(app)

def test_api_terrain_ray():
    """Test POST /api/v1/terrain/ray."""
    payload = {
        "observer_x_m": 0.0,
        "observer_y_m": 0.0,
        "azimuth_deg": 45.0,
        "max_distance_m": 2000.0,
        "radial_step_m": 100.0,
        "observer_height_m": 2.0,
        "interpolation_method": "bilinear"
    }
    response = client.post("/api/v1/terrain/ray", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) == 20
    assert data["azimuth_deg"] == 45.0


def test_api_check_line_of_sight():
    """Test POST /api/v1/terrain/check-los."""
    payload = {
        "target_azimuth_deg": 90.0,
        "target_elevation_deg": 10.0,
        "observer_x_m": 0.0,
        "observer_y_m": 0.0,
        "max_distance_m": 2000.0,
        "radial_step_m": 100.0,
        "observer_height_m": 2.0
    }
    response = client.post("/api/v1/terrain/check-los", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["visibility_state"] in ("clear", "obstructed", "unknown")
    assert "elevation_margin_deg" in data


def test_api_dem_horizon_profile():
    """Test POST /api/v1/terrain/dem-horizon."""
    payload = {
        "observer_x_m": 0.0,
        "observer_y_m": 0.0,
        "max_distance_m": 1500.0,
        "azimuth_step_deg": 30.0,
        "radial_step_m": 100.0,
        "observer_height_m": 2.0
    }
    response = client.post("/api/v1/terrain/dem-horizon", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["horizon_points"]) == 12
    assert data["valid_coverage_percent"] > 0.0


def test_api_invalid_latitude_rejected():
    """Test that latitudes > -60° are rejected with HTTP 400."""
    payload = {
        "observer_lat": -45.0,
        "observer_lon": 0.0,
        "azimuth_deg": 0.0,
        "max_distance_m": 1000.0
    }
    response = client.post("/api/v1/terrain/ray", json=payload)
    assert response.status_code == 400
    assert "outside south-polar mission range" in response.json()["detail"]


# =============================================================================
# SCENARIO 13: Independent Scientific Review Supplementary Tests
# =============================================================================

def test_curvature_parabolic_vs_exact_secant():
    """
    Validate parabolic approximation Delta_h = s^2 / (2 * R_moon) against
    the exact spherical radial secant drop R_moon * (sec(s / R_moon) - 1).
    Verifies that the error bounds are within documented limits:
    - 10 km: error < 1 mm
    - 30 km: error < 50 mm (5 cm)
    - 60 km: error < 600 mm (60 cm)
    - 100 km: error < 4.5 m
    """
    r_moon = MOON_MEAN_RADIUS_M

    for s_km, max_err_m in [(10.0, 0.001), (30.0, 0.050), (60.0, 0.600), (100.0, 4.500)]:
        s = s_km * 1000.0
        parabolic_drop = calculate_curvature_drop(s)
        theta = s / r_moon
        exact_secant_drop = r_moon * (1.0 / math.cos(theta) - 1.0)
        discrepancy = exact_secant_drop - parabolic_drop

        assert discrepancy >= 0.0
        assert discrepancy < max_err_m


def test_narrow_ridge_profile_discretization_vs_direct_los():
    """
    Test a narrow obstacle located at azimuth 15.0° (between 0° and 30° samples).
    Demonstrates that coarse 30° horizon profiling misses the narrow ridge when
    interpolated, whereas direct check_line_of_sight(15.0°) catches the obstruction.
    """
    h, w = 100, 100
    elev = np.full((h, w), 500.0, dtype=np.float32)

    # Place an obstacle at distance ~1000m along azimuth 15°
    theta = math.radians(15.0)
    obs_x = 1000.0 * math.sin(theta)  # ~258.8m
    obs_y = 1000.0 * math.cos(theta)  # ~965.9m

    # Convert to row/col in bounds (-5000, -5000, 5000, 5000)
    col = int((obs_x - (-5000.0)) / 100.0)
    row = int((5000.0 - obs_y) / 100.0)
    elev[row, col] = 1200.0  # +700m peak

    meta = DemMetadata(
        dataset_name="narrow_obstacle_dem",
        data_source="Synthetic Test Fixture",
        file_path="memory://narrow.tif",
        width_pixels=w,
        height_pixels=h,
        pixel_size_x_m=100.0,
        pixel_size_y_m=100.0,
        crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
        bounds_projected=(-5000.0, -5000.0, 5000.0, 5000.0),
        elevation_units="meters",
        reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
        nodata_value=-9999.0,
        valid_data_percent=100.0,
        min_elevation_m=500.0,
        max_elevation_m=1200.0,
        mean_elevation_m=500.1,
        is_synthetic=True
    )
    dem = DemRaster(elev, meta)

    # 1. Coarse 30-degree profile evaluated at 0° and 30°
    profile = calculate_horizon_profile(
        dem=dem,
        observer_coord=(0.0, 0.0),
        max_distance_m=2000.0,
        azimuth_step_deg=30.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    # Looking up 15.0° from interpolated profile:
    el_interp = get_horizon_elevation_at_azimuth(profile, 15.0)
    # The coarse profile missed the narrow peak
    assert el_interp < 5.0

    # 2. Direct line-of-sight check at target_azimuth = 15.0°
    los = check_line_of_sight(
        dem=dem,
        target_azimuth_deg=15.0,
        target_elevation_deg=10.0,
        observer_coord=(0.0, 0.0),
        max_distance_m=2000.0,
        radial_step_m=100.0,
        observer_height_m=2.0,
        mode="projected"
    )
    # Direct LOS caught the narrow obstruction!
    assert los["visibility_state"] == "obstructed"
    assert los["is_obstructed"] is True
    assert 900.0 <= los["obstructing_distance_m"] <= 1100.0


def test_spherical_vs_projected_mode_selection_and_true_north():
    """
    Test geodetic True North alignment vs. Cartesian Grid North.
    Observer at (x0 = 0.0, y0 = -151617.7m) (corresponding to lat = -85.0°, lon = 0.0°).
    - In geodetic coordinates, True North (azimuth 0.0°) points toward the equator (away from pole).
      Along meridian lambda = 0°, latitude increases (e.g. -85.0° -> -84.7°), so y becomes more negative.
    - In planar mode, Grid North (+Y) points toward the South Pole (latitude decreases).
    Verify that mode='auto' and mode='spherical' correctly march North (more negative y, increasing lat),
    while mode='projected' explicitly marches along +Y.
    """
    from app.services.geospatial.projection import lola_polar_projection

    h, w = 100, 100
    elev = np.zeros((h, w), dtype=np.float32)
    meta = DemMetadata(
        dataset_name="true_north_test_dem",
        data_source="Test",
        file_path="memory://test.tif",
        width_pixels=w,
        height_pixels=h,
        pixel_size_x_m=1000.0,
        pixel_size_y_m=1000.0,
        crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
        bounds_projected=(-50000.0, -200000.0, 50000.0, -100000.0),
        elevation_units="meters",
        reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
        nodata_value=-9999.0,
        valid_data_percent=100.0,
        min_elevation_m=0.0,
        max_elevation_m=0.0,
        mean_elevation_m=0.0,
        is_synthetic=False
    )
    dem = DemRaster(elev, meta)

    x0 = 0.0
    y0 = -151617.7  # lat -85.0, lon 0.0

    # 1. Test mode='auto' with projected coordinates (must resolve to spherical / True North)
    ray_auto = sample_terrain_ray(
        dem=dem,
        observer_x_m=x0,
        observer_y_m=y0,
        azimuth_deg=0.0,
        max_distance_m=10000.0,
        radial_step_m=5000.0,
        mode="auto"
    )
    assert ray_auto["mode"] == "spherical"
    # In spherical mode, True North increases latitude (more negative y in polar stereo)
    for sample in ray_auto["samples"]:
        assert sample["latitude_deg"] > -85.0  # latitude increased toward equator
        assert sample["projected_y_m"] < y0    # y became more negative (away from pole)

    # 2. Test explicit mode='projected' (preserves Cartesian +Y Grid North)
    ray_proj = sample_terrain_ray(
        dem=dem,
        observer_x_m=x0,
        observer_y_m=y0,
        azimuth_deg=0.0,
        max_distance_m=10000.0,
        radial_step_m=5000.0,
        mode="projected"
    )
    assert ray_proj["mode"] == "projected"
    for sample in ray_proj["samples"]:
        assert sample["projected_y_m"] > y0  # y increased along +Y axis (Grid North)


# =============================================================================
# PHASE 4 TESTS: Solar Occultation, Penumbra, Irradiance, and Fresnel Clearance
# =============================================================================

def test_solar_disk_penumbra_four_states():
    """
    Test 4-state solar disk occultation using the circular-segment formula:
    1. Full illumination: solar center El > H + R_sun -> f_disk = 1.0, state = 'full'
    2. Partial occultation:
       - El = H (center exactly on horizon) -> u = 0 -> f_disk = 0.50 (half disk)
       - H - R_sun < El < H + R_sun -> 0 < f_disk < 1, state = 'partial'
    3. Total occultation: El < H - R_sun -> f_disk = 0.0, state = 'occulted'
    4. Unknown horizon: is_ray_valid = False -> f_disk = 0.0, state = 'unknown'
    """
    delta_sun = 0.2667  # 16 arcmin semi-diameter
    h = 2.0  # Horizon elevation angle 2.0 deg

    # 1. Full
    res_full = calculate_solar_disk_visibility(
        sun_elevation_deg=h + delta_sun + 0.5,
        horizon_elevation_deg=h,
        solar_semi_diameter_deg=delta_sun,
        is_ray_valid=True
    )
    assert res_full["visibility_state"] == "full"
    assert res_full["visible_fraction"] == 1.0

    # 2. Partial: center exactly on horizon -> f_disk = 0.50
    res_half = calculate_solar_disk_visibility(
        sun_elevation_deg=h,
        horizon_elevation_deg=h,
        solar_semi_diameter_deg=delta_sun,
        is_ray_valid=True
    )
    assert res_half["visibility_state"] == "partial"
    assert pytest.approx(res_half["visible_fraction"], abs=1e-4) == 0.50

    # 2b. Partial: limb grazing above horizon
    res_graze = calculate_solar_disk_visibility(
        sun_elevation_deg=h + 0.1,
        horizon_elevation_deg=h,
        solar_semi_diameter_deg=delta_sun,
        is_ray_valid=True
    )
    assert res_graze["visibility_state"] == "partial"
    assert 0.50 < res_graze["visible_fraction"] < 1.0

    # 3. Occulted
    res_occ = calculate_solar_disk_visibility(
        sun_elevation_deg=h - delta_sun - 0.5,
        horizon_elevation_deg=h,
        solar_semi_diameter_deg=delta_sun,
        is_ray_valid=True
    )
    assert res_occ["visibility_state"] == "occulted"
    assert res_occ["visible_fraction"] == 0.0

    # 4. Unknown
    res_unk = calculate_solar_disk_visibility(
        sun_elevation_deg=5.0,
        horizon_elevation_deg=None,
        solar_semi_diameter_deg=delta_sun,
        is_ray_valid=False
    )
    assert res_unk["visibility_state"] == "unknown"
    assert res_unk["is_illuminated"] is False
    assert res_unk["visible_fraction"] == 0.0
    assert len(res_unk["warnings"]) > 0


def test_solar_irradiance_distance_scaling():
    """
    Test inverse-square distance scaling of solar irradiance:
    S(r) = S_0 * (1 AU / r)^2
    - At 1.0 AU: S = 1361.0 W/m^2
    - At perihelion (0.98329 AU): S ~ 1407.6 W/m^2
    - At aphelion (1.01671 AU): S ~ 1316.6 W/m^2
    - At None: fallback to 1361.0 with provenance 'ASSUMPTION'
    - Non-positive distance raises ValueError
    """
    au_km = 149597870.7
    irr_1au = calculate_solar_irradiance(sun_distance_km=au_km)
    assert irr_1au["solar_irradiance_w_m2"] == 1361.0
    assert irr_1au["is_measured"] is True

    irr_peri = calculate_solar_irradiance(sun_distance_km=0.98329 * au_km)
    expected_peri = 1361.0 / (0.98329 ** 2)
    assert pytest.approx(irr_peri["solar_irradiance_w_m2"], abs=0.01) == round(expected_peri, 2)
    assert irr_peri["solar_irradiance_w_m2"] > 1400.0

    irr_aph = calculate_solar_irradiance(sun_distance_km=1.01671 * au_km)
    expected_aph = 1361.0 / (1.01671 ** 2)
    assert pytest.approx(irr_aph["solar_irradiance_w_m2"], abs=0.01) == round(expected_aph, 2)
    assert irr_aph["solar_irradiance_w_m2"] < 1320.0

    irr_none = calculate_solar_irradiance(sun_distance_km=None)
    assert irr_none["solar_irradiance_w_m2"] == 1361.0
    assert irr_none["is_measured"] is False
    assert "ASSUMPTION" in irr_none["provenance"]

    with pytest.raises(ValueError):
        calculate_solar_irradiance(sun_distance_km=0.0)
    with pytest.raises(ValueError):
        calculate_solar_irradiance(sun_distance_km=-1000.0)


def test_evaluate_solar_occultation_and_flux_integration(flat_dem):
    """Test full integration from DEM ray trace to solar flux calculation."""
    res = evaluate_solar_occultation_and_flux(
        dem=flat_dem,
        observer_coord=(0.0, 0.0),
        sun_azimuth_deg=45.0,
        sun_elevation_deg=5.0,
        max_distance_m=2000.0
    )
    assert "visibility_state" in res
    assert "visible_fraction" in res
    assert "solar_irradiance_w_m2" in res
    assert "direct_normal_flux_w_m2" in res
    assert "horizontal_flux_w_m2" in res
    assert "horizon_elevation_deg" in res
    assert res["visibility_state"] == "full"
    assert res["direct_normal_flux_w_m2"] > 1300.0


def test_fresnel_radius_benchmark_exact():
    """
    Validate full two-endpoint first Fresnel-zone radius equation against analytical benchmarks:
    r_F1 = sqrt(lambda * d1 * d2 / (d1 + d2))
    Benchmark distance:
      d1 = 10,000 m (10 km to obstacle)
      d2 = 384,400,000 m - 10,000 m
    Frequencies:
      S-band: f = 2.2 GHz (lambda = 0.136269 m) -> r_F1 = 36.914 m, 0.6*r_F1 = 22.148 m
      Ka-band: f = 32.0 GHz (lambda = 0.0093685 m) -> r_F1 = 9.678 m, 0.6*r_F1 = 5.807 m
    """
    d1 = 10000.0
    d_earth = 384400000.0

    # S-band 2.2 GHz
    s_band = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=d_earth,
        frequency_hz=2.2e9,
        earth_elevation_deg=1.0,
        obstacle_elevation_angle_deg=0.0
    )
    assert pytest.approx(s_band["fresnel_radius_r1_m"], abs=0.01) == 36.914
    assert pytest.approx(s_band["required_clearance_m"], abs=0.01) == 22.148
    assert s_band["fresnel_screening_pass"] is True

    # Ka-band 32.0 GHz
    ka_band = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=d_earth,
        frequency_hz=32.0e9,
        earth_elevation_deg=1.0,
        obstacle_elevation_angle_deg=0.0
    )
    assert pytest.approx(ka_band["fresnel_radius_r1_m"], abs=0.01) == 9.678
    assert pytest.approx(ka_band["required_clearance_m"], abs=0.01) == 5.807
    assert ka_band["fresnel_screening_pass"] is True


def test_fresnel_equal_elevation_zero_clearance():
    """
    When Earth elevation and obstacle elevation are identical (measured from the common local horizontal),
    vertical and perpendicular clearances must be exactly zero:
    Delta_z = d1 * (tan(theta) - tan(theta)) = 0.0
    """
    d1 = 8000.0
    res = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=384400000.0,
        frequency_hz=2.2e9,
        earth_elevation_deg=3.5,
        obstacle_elevation_angle_deg=3.5
    )
    assert pytest.approx(res["vertical_clearance_m"], abs=1e-5) == 0.0
    assert pytest.approx(res["perpendicular_clearance_m"], abs=1e-5) == 0.0
    assert res["clearance_margin_m"] == -res["required_clearance_m"]
    assert res["fresnel_screening_pass"] is False


def test_fresnel_obstacle_below_direct_path_positive_clearance():
    """
    When the obstacle is below the line of sight (theta_Earth > theta_obs),
    clearances are strictly positive.
    d1 = 5,000m, theta_Earth = 4.0°, theta_obs = 1.0°:
    Delta_z = 5000 * (tan(4.0°) - tan(1.0°)) = 5000 * (0.0699268 - 0.0174551) = 262.359m
    Delta_h_perp = 262.359 * cos(4.0°) = 261.720m
    """
    d1 = 5000.0
    res = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=384400000.0,
        frequency_hz=2.2e9,
        earth_elevation_deg=4.0,
        obstacle_elevation_angle_deg=1.0
    )
    assert res["vertical_clearance_m"] > 0.0
    assert res["perpendicular_clearance_m"] > 0.0
    assert pytest.approx(res["vertical_clearance_m"], abs=0.01) == 262.36
    assert pytest.approx(res["perpendicular_clearance_m"], abs=0.01) == 261.72
    assert res["fresnel_screening_pass"] is True
    assert res["clearance_margin_m"] > 0.0


def test_fresnel_obstacle_above_direct_path_negative_clearance():
    """
    When the obstacle is above the line of sight (theta_Earth < theta_obs),
    clearance is negative (knife-edge impingement).
    """
    d1 = 10000.0
    res = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=384400000.0,
        frequency_hz=2.2e9,
        earth_elevation_deg=1.0,
        obstacle_elevation_angle_deg=3.0
    )
    assert res["vertical_clearance_m"] < 0.0
    assert res["perpendicular_clearance_m"] < 0.0
    assert res["clearance_margin_m"] < 0.0
    assert res["fresnel_screening_pass"] is False


def test_fresnel_low_angle_exact_vs_tangent_difference_approximation():
    """
    Compare exact expression Delta_z = d1 * (tan(alpha) - tan(beta))
    against previous approximation Delta_z_approx = d1 * tan(alpha - beta).
    At d1 = 10,000m, alpha = 5.0°, beta = 3.0°:
    - Exact: 10000 * (tan(5.0°) - tan(3.0°)) = 10000 * (0.08748866 - 0.05240778) = 350.809m
    - Approximation: 10000 * tan(5.0° - 3.0°) = 10000 * tan(2.0°) = 349.208m
    - Geometric discrepancy: 350.809m - 349.208m = 1.601m
    Verify that the implementation produces the exact value (350.81m) and eliminates the 1.60m error.
    """
    d1 = 10000.0
    alpha = 5.0
    beta = 3.0
    res = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=384400000.0,
        frequency_hz=2.2e9,
        earth_elevation_deg=alpha,
        obstacle_elevation_angle_deg=beta
    )
    exact_val = d1 * (math.tan(math.radians(alpha)) - math.tan(math.radians(beta)))
    approx_val = d1 * math.tan(math.radians(alpha - beta))

    assert pytest.approx(res["vertical_clearance_m"], abs=0.01) == 350.81
    assert pytest.approx(res["vertical_clearance_m"], abs=0.01) == exact_val
    assert abs(res["vertical_clearance_m"] - approx_val) > 1.5
    assert pytest.approx(res["vertical_clearance_m"] - approx_val, abs=0.05) == 1.60


def test_fresnel_clearance_screening():
    """
    Test clearance screening categories:
    - perpendicular clearance >= 0.6*r_F1 -> screening pass True
    - 0 <= perpendicular clearance < 0.6*r_F1 -> screening pass False
    - clearance < 0 -> screening pass False (direct knife-edge impingement)
    """
    d1 = 10000.0
    d_earth = 384400000.0
    f = 2.2e9  # r_F1 = 36.914m, required_clearance = 22.148m

    # Case 1: Sub-60% clearance (e.g. diff produces ~15m clearance)
    # tan(theta_earth) - tan(1.0°) = 15.0 / 10000.0 = 0.00150
    # tan(1.0°) = 0.01745506 -> tan(theta_earth) = 0.01895506 -> theta_earth = 1.08592°
    sub_60 = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=d_earth,
        frequency_hz=f,
        earth_elevation_deg=1.08592,
        obstacle_elevation_angle_deg=1.0
    )
    assert pytest.approx(sub_60["vertical_clearance_m"], abs=0.1) == 15.0
    assert pytest.approx(sub_60["perpendicular_clearance_m"], abs=0.1) == 15.0
    assert sub_60["fresnel_screening_pass"] is False
    assert sub_60["clearance_margin_m"] < 0.0

    # Case 2: Direct knife-edge impingement (negative clearance, obstacle higher than Earth ray)
    knife = calculate_fresnel_clearance(
        obstacle_distance_m=d1,
        earth_distance_m=d_earth,
        frequency_hz=f,
        earth_elevation_deg=0.5,
        obstacle_elevation_angle_deg=1.0
    )
    assert knife["vertical_clearance_m"] < 0.0
    assert knife["perpendicular_clearance_m"] < 0.0
    assert knife["fresnel_screening_pass"] is False


def test_separate_earth_lunar_masks(flat_dem):
    """
    Verify that Earth-side DSN station elevation mask is kept strictly separate
    from lunar terrain horizon line-of-sight screening.
    - Earth is at lunar local elevation El_lunar = 2.0 deg, azimuth 0.0 deg.
    - Lunar horizon H(Az) is negative on flat DEM with observer height and curvature.
    - Clearance is positive: lunar terrain LOS is CLEAR.
    - A hypothetical 5.0 deg DSN station elevation mask on Earth does NOT mask the lunar transmitter.
    """
    los = check_line_of_sight(
        dem=flat_dem,
        observer_coord=(0.0, 0.0),
        target_azimuth_deg=0.0,
        target_elevation_deg=2.0,
        max_distance_m=3000.0,
        observer_height_m=2.0,
        mode="projected"
    )
    assert los["visibility_state"] == "clear"
    assert los["is_obstructed"] is False
    assert los["elevation_margin_deg"] > 1.9
    assert los["target_elevation_deg"] == 2.0
