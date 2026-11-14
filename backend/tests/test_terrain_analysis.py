"""
Unit and integration tests for Lunar Terrain Analysis (terrain_analysis.py).
Member 3: GIS & Scientific Computing.

Validates Phase 2 and Scientific Regression Patch:
1. Flat terrain produces approximately zero slope.
2. A known planar surface produces the expected slope.
3. Different X and Y pixel resolutions (dx != dy) are scaled accurately.
4. Diagonal slope on anisotropic grids (dx != dy, sloping in both X and Y).
5. Synthetic Gaussian hill produces nonzero slope.
6. Roughness is approximately zero on flat terrain.
7. Rough surface produces higher roughness than flat terrain.
8. Physical roughness baselines for anisotropic non-square pixels.
9. Terrain profiles return correctly ordered distances and elevations.
10. Bilinear interpolation on a known planar surface.
11. Bilinear interpolation around nodata cells (strict nodata policy, no bleeding).
12. Invalid parameters raise clear errors.
13. Geographic CRS rejection (DemCoordinateError).
14. Projected CRS with non-meter units rejection (DemCoordinateError).
15. Rotated or sheared raster transform rejection (DemCoordinateError).
16. Nodata regions handled safely without false gradient spikes.
17. Raster boundaries do not create misleading edge artifacts.
18. Landing circle fully inside DEM coverage (100% valid coverage).
19. Landing circle partly outside DEM coverage (strictly normalized coverage < 100%).
20. Landing circle entirely outside DEM coverage (0% coverage, warning).
21. API endpoint /terrain/profile valid and invalid (HTTP 400) coordinates.
22. API endpoint /terrain/assess-landing-area valid and invalid (HTTP 400) coordinates.
"""

import os
import math
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_application
from app.services.geospatial.dem_loader import DemLoader, DemRaster, DemMetadata
from app.services.geospatial.terrain_analysis import (
    DemCoordinateError,
    calculate_slope,
    calculate_slope_statistics,
    calculate_roughness,
    calculate_roughness_statistics,
    calculate_elevation_statistics,
    calculate_terrain_profile,
    assess_landing_area,
    calculate_disjoint_slope_categories,
    calculate_cliff_buffer_distance,
    evaluate_site_terrain_safety
)

# Reference synthetic DEM path
SYNTHETIC_DEM_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "dem", "synthetic_test_dem.tif")
)


class TestTerrainSlope:
    """Validate terrain slope calculations using controlled synthetic surfaces."""

    def test_flat_terrain_zero_slope(self):
        """Flat terrain produces approximately zero slope across all cells."""
        flat = np.full((30, 30), 250.0, dtype=np.float64)
        slope = calculate_slope(flat, pixel_size_x_m=100.0, pixel_size_y_m=100.0)

        assert slope.shape == (30, 30)
        assert np.nanmax(slope) < 1e-6
        assert np.nanmin(slope) >= 0.0

        stats = calculate_slope_statistics(slope)
        assert stats["mean_slope_deg"] == 0.0
        assert stats["max_slope_deg"] == 0.0
        assert stats["threshold_coverage_pct"]["lte_5deg"] == 100.0

    def test_known_planar_surface_slope(self):
        """A known planar surface produces the exact analytical slope."""
        dx, dy = 100.0, 100.0
        x = np.arange(25) * dx
        y = np.arange(25) * dy
        xx, yy = np.meshgrid(x, y)
        plane = 0.1 * xx

        slope = calculate_slope(plane, pixel_size_x_m=dx, pixel_size_y_m=dy)
        expected_deg = math.degrees(math.atan(0.1))

        # Check interior cells (excluding outer border)
        interior_slope = slope[1:-1, 1:-1]
        assert np.allclose(interior_slope, expected_deg, atol=1e-4)

        # 45-degree planar incline: dz/dx = 1.0 -> arctan(1.0) = 45.0 degrees
        plane_45 = 1.0 * xx
        slope_45 = calculate_slope(plane_45, pixel_size_x_m=dx, pixel_size_y_m=dy)
        assert np.allclose(slope_45[1:-1, 1:-1], 45.0, atol=1e-4)

    def test_differing_pixel_resolutions(self):
        """Different X and Y pixel resolutions (dx != dy) are scaled accurately."""
        dx = 50.0
        dy = 200.0
        cols, rows = 20, 20
        x = np.arange(cols) * dx
        y = np.arange(rows) * dy
        xx, yy = np.meshgrid(x, y)

        plane_x = 0.2 * xx
        slope_x = calculate_slope(plane_x, pixel_size_x_m=dx, pixel_size_y_m=dy)
        expected_deg_x = math.degrees(math.atan(0.2))
        assert np.allclose(slope_x[1:-1, 1:-1], expected_deg_x, atol=1e-4)

        plane_y = 0.2 * yy
        slope_y = calculate_slope(plane_y, pixel_size_x_m=dx, pixel_size_y_m=dy)
        expected_deg_y = math.degrees(math.atan(0.2))
        assert np.allclose(slope_y[1:-1, 1:-1], expected_deg_y, atol=1e-4)

    def test_diagonal_slope_anisotropic_resolution(self):
        """Diagonal slope on anisotropic grid (dx != dy, sloping in both X and Y)."""
        dx = 60.0
        dy = 120.0
        cols, rows = 25, 25
        x = np.arange(cols) * dx
        y = np.arange(rows) * dy
        xx, yy = np.meshgrid(x, y)

        # z = 0.10 * x + 0.15 * y
        plane_diag = 0.10 * xx + 0.15 * yy
        slope_diag = calculate_slope(plane_diag, pixel_size_x_m=dx, pixel_size_y_m=dy)

        # Analytical gradient magnitude: sqrt(0.10^2 + 0.15^2) = sqrt(0.01 + 0.0225) = sqrt(0.0325) ~ 0.180277
        expected_grad = math.sqrt(0.10**2 + 0.15**2)
        expected_deg = math.degrees(math.atan(expected_grad))

        interior_slope = slope_diag[1:-1, 1:-1]
        assert np.allclose(interior_slope, expected_deg, atol=1e-3)

    def test_synthetic_hill_nonzero_slope(self):
        """A synthetic Gaussian hill produces nonzero slope along flanks and ~0 at summit."""
        cols, rows = 41, 41
        dx, dy = 100.0, 100.0
        x = (np.arange(cols) - 20) * dx
        y = (np.arange(rows) - 20) * dy
        xx, yy = np.meshgrid(x, y)
        r2 = xx**2 + yy**2

        hill = 500.0 * np.exp(-r2 / (2.0 * (800.0**2)))
        slope = calculate_slope(hill, pixel_size_x_m=dx, pixel_size_y_m=dy)

        assert slope[20, 20] < 0.1
        flank_slopes = slope[20, 28]  # 800m East
        assert flank_slopes == pytest.approx(20.7, abs=1.5)


class TestCrsAndTransformValidation:
    """Validate CRS and affine transform inspection and rejection."""

    def test_geographic_crs_rejection(self):
        """Reject DEMs with geographic angular units (degrees) with DemCoordinateError."""
        flat = np.full((20, 20), 100.0, dtype=np.float32)
        meta = DemMetadata(
            dataset_name="geographic_test",
            data_source="Test",
            file_path="",
            width_pixels=20,
            height_pixels=20,
            pixel_size_x_m=0.01,  # 0.01 degrees!
            pixel_size_y_m=0.01,
            crs_repr="GEOGCS[\"GCS_Moon_2000\",DATUM[\"D_Moon_2000\",SPHEROID[\"Moon_2000_IAU_IAG\",1737400,0]],UNIT[\"Degree\",0.0174532925199433]]",
            bounds_projected=(-10.0, -10.0, 10.0, 10.0),
            elevation_units="meters",
            reference_datum="Moon",
            nodata_value=-9999.0,
            valid_data_percent=100.0,
            min_elevation_m=100.0,
            max_elevation_m=100.0,
            mean_elevation_m=100.0,
            is_synthetic=True
        )
        raster = DemRaster(flat, meta)

        with pytest.raises(DemCoordinateError) as exc_info:
            calculate_slope(raster)
        assert "geographic" in str(exc_info.value).lower()

    def test_projected_crs_non_meter_units_rejection(self):
        """Reject DEMs with non-meter linear units (e.g. feet)."""
        flat = np.full((20, 20), 100.0, dtype=np.float32)
        meta = DemMetadata(
            dataset_name="feet_test",
            data_source="Test",
            file_path="",
            width_pixels=20,
            height_pixels=20,
            pixel_size_x_m=300.0,
            pixel_size_y_m=300.0,
            crs_repr="PROJCS[\"Stereo_Feet\",UNIT[\"foot\",0.3048]]",
            bounds_projected=(-1000.0, -1000.0, 1000.0, 1000.0),
            elevation_units="meters",
            reference_datum="Moon",
            nodata_value=-9999.0,
            valid_data_percent=100.0,
            min_elevation_m=100.0,
            max_elevation_m=100.0,
            mean_elevation_m=100.0,
            is_synthetic=True
        )
        raster = DemRaster(flat, meta)

        with pytest.raises(DemCoordinateError) as exc_info:
            calculate_slope(raster)
        assert "non-meter" in str(exc_info.value).lower() or "foot" in str(exc_info.value).lower()

    def test_rotated_or_sheared_raster_rejection(self):
        """Reject sheared or rotated raster transforms with DemCoordinateError."""
        flat = np.full((20, 20), 100.0, dtype=np.float32)
        meta = DemMetadata(
            dataset_name="rotated_test",
            data_source="Test",
            file_path="",
            width_pixels=20,
            height_pixels=20,
            pixel_size_x_m=100.0,
            pixel_size_y_m=100.0,
            crs_repr="PROJCS[\"Polar_Stereographic\",UNIT[\"metre\",1]]",
            bounds_projected=(-1000.0, -1000.0, 1000.0, 1000.0),
            elevation_units="meters",
            reference_datum="Moon",
            nodata_value=-9999.0,
            valid_data_percent=100.0,
            min_elevation_m=100.0,
            max_elevation_m=100.0,
            mean_elevation_m=100.0,
            is_synthetic=True
        )
        # Transform with rotation: b = 15.0, d = 15.0
        sheared_transform = (100.0, 15.0, 0.0, 15.0, -100.0, 0.0)
        raster = DemRaster(flat, meta, transform=sheared_transform)

        with pytest.raises(DemCoordinateError) as exc_info:
            calculate_slope(raster)
        assert "rotated or sheared" in str(exc_info.value).lower()


class TestTerrainRoughness:
    """Validate terrain roughness calculations and physical baselines."""

    def test_roughness_zero_on_flat_terrain(self):
        flat = np.full((25, 25), 350.0, dtype=np.float64)
        rough_std = calculate_roughness(flat, window_size=5, metric="std")
        assert np.nanmax(rough_std) < 1e-6

        rough_relief = calculate_roughness(flat, window_size=5, metric="relief")
        assert np.nanmax(rough_relief) < 1e-6

    def test_roughness_rough_vs_flat(self):
        np.random.seed(42)
        flat = np.full((30, 30), 100.0, dtype=np.float64)
        noise = np.random.normal(0.0, 15.0, flat.shape)
        rough = flat + noise

        r_flat = calculate_roughness(flat, window_size=5, metric="std")
        r_rough = calculate_roughness(rough, window_size=5, metric="std")

        assert float(np.nanmean(r_flat)) == 0.0
        assert float(np.nanmean(r_rough)) == pytest.approx(15.0, abs=2.5)

    def test_roughness_physical_baseline_anisotropic(self):
        """Verify physical baseline reporting for anisotropic non-square pixels."""
        flat = np.full((20, 20), 100.0, dtype=np.float64)
        stats = calculate_roughness_statistics(
            flat,
            metric="std",
            window_size_pixels=5,
            pixel_size_x_m=50.0,
            pixel_size_y_m=150.0
        )
        assert stats["physical_baseline_x_m"] == 250.0  # 5 * 50m
        assert stats["physical_baseline_y_m"] == 750.0  # 5 * 150m
        assert "anisotropic" in stats["baseline_convention"].lower()


class TestElevationInterpolationAndProfile:
    """Validate elevation interpolation (bilinear vs nearest) and profile transects."""

    def test_bilinear_interpolation_planar_surface(self):
        """Bilinear interpolation at fractional coordinates on a known plane is exact."""
        dx, dy = 100.0, 100.0
        cols, rows = 10, 10
        left, top = 0.0, 1000.0
        # Continuous function: z(x, y) = 500.0 + 0.05 * x - 0.02 * y
        x_grid = left + (np.arange(cols) + 0.5) * dx
        y_grid = top - (np.arange(rows) + 0.5) * dy
        xx, yy = np.meshgrid(x_grid, y_grid)
        elev_plane = 500.0 + 0.05 * xx - 0.02 * yy

        meta = DemMetadata(
            dataset_name="plane_test",
            data_source="Test",
            file_path="",
            width_pixels=cols,
            height_pixels=rows,
            pixel_size_x_m=dx,
            pixel_size_y_m=dy,
            crs_repr="PROJCS[\"Polar_Stereographic\",UNIT[\"metre\",1]]",
            bounds_projected=(left, top - rows * dy, left + cols * dx, top),
            elevation_units="meters",
            reference_datum="Moon",
            nodata_value=-9999.0,
            valid_data_percent=100.0,
            min_elevation_m=float(np.min(elev_plane)),
            max_elevation_m=float(np.max(elev_plane)),
            mean_elevation_m=float(np.mean(elev_plane)),
            is_synthetic=True
        )
        raster = DemRaster(elev_plane.astype(np.float32), meta)

        # Test arbitrary sub-pixel points
        test_points = [
            (250.0, 750.0),
            (325.4, 612.8),
            (115.0, 890.0),
            (540.2, 420.7)
        ]
        for tx, ty in test_points:
            z_bilinear = raster.get_elevation_at_projected(tx, ty, method="bilinear")
            z_expected = 500.0 + 0.05 * tx - 0.02 * ty
            assert z_bilinear == pytest.approx(z_expected, abs=1e-4)

    def test_bilinear_interpolation_around_nodata(self):
        """Bilinear interpolation returns NaN if any interpolation corner is nodata (no bleeding)."""
        data = np.full((10, 10), 100.0, dtype=np.float32)
        data[4, 4] = -9999.0  # nodata cell

        meta = DemMetadata(
            dataset_name="nodata_interp",
            data_source="Test",
            file_path="",
            width_pixels=10,
            height_pixels=10,
            pixel_size_x_m=100.0,
            pixel_size_y_m=100.0,
            crs_repr="PROJCS[\"Polar_Stereographic\",UNIT[\"metre\",1]]",
            bounds_projected=(0.0, 0.0, 1000.0, 1000.0),
            elevation_units="meters",
            reference_datum="Moon",
            nodata_value=-9999.0,
            valid_data_percent=99.0,
            min_elevation_m=100.0,
            max_elevation_m=100.0,
            mean_elevation_m=100.0,
            is_synthetic=True
        )
        raster = DemRaster(data, meta)

        # A point between cell (4, 4) and cell (4, 5)
        # Pixel (4, 4) is at row 4, col 4: x around 450, y around 550
        z_near_nodata = raster.get_elevation_at_projected(450.0, 550.0, method="bilinear")
        assert math.isnan(z_near_nodata)

    def test_terrain_profile_ordered_distances(self):
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        profile = calculate_terrain_profile(
            raster,
            start_coord=(-3000.0, 0.0),
            end_coord=(3000.0, 0.0),
            sample_step_m=100.0,
            interpolation_method="bilinear"
        )
        assert profile["profile_length_m"] == 6000.0
        assert profile["interpolation_method"] == "bilinear"
        pts = profile["points"]
        distances = [p["distance_m"] for p in pts]
        assert distances == sorted(distances)
        assert distances[0] == 0.0
        assert distances[-1] == 6000.0


class TestBoundaryAndNodataHandling:
    """Validate boundary conditions and edge cases."""

    def test_invalid_parameters_raise_errors(self):
        flat = np.full((10, 10), 100.0)
        with pytest.raises(ValueError):
            calculate_slope(flat, pixel_size_x_m=-50.0)
        with pytest.raises(ValueError):
            calculate_roughness(flat, window_size=1)

    def test_nodata_regions_handled_safely(self):
        flat = np.full((20, 20), 100.0, dtype=np.float64)
        flat[8:12, 8:12] = -9999.0
        slope = calculate_slope(flat, nodata_value=-9999.0, pixel_size_x_m=100.0, pixel_size_y_m=100.0)
        assert np.all(np.isnan(slope[8:12, 8:12]))
        valid_slopes = slope[~np.isnan(slope)]
        assert np.nanmax(valid_slopes) < 1e-4

    def test_raster_boundaries_no_artifacts(self):
        plane = np.arange(10, dtype=np.float64).reshape((1, 10)).repeat(10, axis=0) * 10.0
        slope = calculate_slope(plane, pixel_size_x_m=100.0, pixel_size_y_m=100.0)
        expected_deg = math.degrees(math.atan(0.1))
        assert slope[0, 5] == pytest.approx(expected_deg, abs=1e-3)
        assert slope[9, 5] == pytest.approx(expected_deg, abs=1e-3)


class TestLandingAreaCoverageNormalization:
    """Validate rigorous landing area coverage normalization."""

    def test_landing_circle_fully_inside_dem(self):
        """Landing circle fully covered by valid DEM cells yields 100% coverage."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        # Center at (0, 0), radius 800m is entirely inside [-5000, 5000]
        res = assess_landing_area(raster, center_projected=(0.0, 0.0), radius_m=800.0)
        assert res["valid_data_percent"] == 100.0
        assert len(res["warnings"]) == 0

    def test_landing_circle_partly_outside_dem(self):
        """Landing circle overlapping DEM edge has strictly normalized coverage < 100%."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        # Right boundary of synthetic DEM is +5000m
        # Placing center at x=4500m with radius=1000m means 500m of the circle extends outside!
        res = assess_landing_area(raster, center_projected=(4500.0, 0.0), radius_m=1000.0)

        # Coverage must be strictly less than 100% and greater than 40%
        assert res["valid_data_percent"] < 85.0
        assert res["valid_data_percent"] > 40.0
        assert any("outside" in w.lower() for w in res["warnings"])

    def test_landing_circle_entirely_outside_dem(self):
        """Landing circle completely outside DEM coverage yields 0% coverage and warning."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        res = assess_landing_area(raster, center_projected=(15000.0, 15000.0), radius_m=1000.0)
        assert res["valid_data_percent"] == 0.0
        assert any("completely outside" in w.lower() for w in res["warnings"])


class TestApiIntegrationAndErrorHandling:
    """Validate FastAPI endpoints, error handling, and HTTP status codes."""

    def test_api_terrain_profile_valid(self):
        app = create_application()
        client = TestClient(app)
        payload = {
            "start_x_m": -1000.0,
            "start_y_m": 0.0,
            "end_x_m": 1000.0,
            "end_y_m": 0.0,
            "sample_step_m": 100.0,
            "interpolation_method": "bilinear"
        }
        res = client.post("/api/v1/terrain/profile", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["profile_length_m"] == 2000.0
        assert data["interpolation_method"] == "bilinear"

    def test_api_terrain_profile_invalid_coordinate_http_400(self):
        """Coordinates outside south-polar region (e.g. Equator) return HTTP 400 Bad Request."""
        app = create_application()
        client = TestClient(app)
        payload = {
            "start_lat": 0.0,  # Invalid: Equator is outside polar projection range
            "start_lon": 0.0,
            "end_lat": -89.0,
            "end_lon": 0.0
        }
        res = client.post("/api/v1/terrain/profile", json=payload)
        assert res.status_code == 400
        assert "invalid start coordinate" in res.json()["detail"].lower()

    def test_api_landing_area_assessment_valid(self):
        app = create_application()
        client = TestClient(app)
        payload = {
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "radius_m": 1000.0,
            "slope_thresholds": [5.0, 10.0, 12.0, 15.0]
        }
        res = client.post("/api/v1/terrain/assess-landing-area", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert "physical_baseline_m" in data["roughness_stats"]
        assert "baseline_convention" in data["roughness_stats"]

    def test_api_landing_area_invalid_coordinate_http_400(self):
        """Invalid center coordinate (e.g. northern hemisphere) returns HTTP 400 Bad Request."""
        app = create_application()
        client = TestClient(app)
        payload = {
            "center_lat": 10.0,
            "center_lon": 0.0,
            "radius_m": 1000.0
        }
        res = client.post("/api/v1/terrain/assess-landing-area", json=payload)
        assert res.status_code == 400
        assert "invalid center coordinate" in res.json()["detail"].lower()


class TestDisjointSlopeCategories:
    """Validate disjoint slope partition [0-5], (5-10], (10-15], >15 and valid-cell normalization."""

    def test_disjoint_slope_categories_sum_to_100_uniform_flat(self):
        """Flat terrain: 100% in [0-5 deg], others 0%, score = 100.0, sum = 100%."""
        slopes = np.zeros((50, 50), dtype=np.float32)
        res = calculate_disjoint_slope_categories(slopes)
        assert res["p_0_to_5_deg"] == 100.0
        assert res["p_5_to_10_deg"] == 0.0
        assert res["p_10_to_15_deg"] == 0.0
        assert res["p_gt_15_deg"] == 0.0
        total_sum = (
            res["p_0_to_5_deg"] + res["p_5_to_10_deg"] +
            res["p_10_to_15_deg"] + res["p_gt_15_deg"]
        )
        assert pytest.approx(total_sum, abs=1e-4) == 100.0
        assert res["slope_distribution_score"] == 100.0
        assert res["valid_cells"] == 2500

    def test_disjoint_slope_categories_sum_to_100_steep(self):
        """Steep terrain: 100% in >15 deg, others 0%, score = 0.0, sum = 100%."""
        slopes = np.full((50, 50), 25.0, dtype=np.float32)
        res = calculate_disjoint_slope_categories(slopes)
        assert res["p_0_to_5_deg"] == 0.0
        assert res["p_5_to_10_deg"] == 0.0
        assert res["p_10_to_15_deg"] == 0.0
        assert res["p_gt_15_deg"] == 100.0
        total_sum = (
            res["p_0_to_5_deg"] + res["p_5_to_10_deg"] +
            res["p_10_to_15_deg"] + res["p_gt_15_deg"]
        )
        assert pytest.approx(total_sum, abs=1e-4) == 100.0
        assert res["slope_distribution_score"] == 0.0

    def test_disjoint_slope_categories_sum_to_100_multibucket_and_formula(self):
        """Known distribution across all 4 buckets verifies normalization and S_slope formula."""
        arr = np.concatenate([
            np.full(160, 3.0),
            np.full(120, 8.0),
            np.full(80, 12.0),
            np.full(40, 20.0)
        ]).astype(np.float32)

        res = calculate_disjoint_slope_categories(arr)
        assert pytest.approx(res["p_0_to_5_deg"], abs=1e-4) == 40.0
        assert pytest.approx(res["p_5_to_10_deg"], abs=1e-4) == 30.0
        assert pytest.approx(res["p_10_to_15_deg"], abs=1e-4) == 20.0
        assert pytest.approx(res["p_gt_15_deg"], abs=1e-4) == 10.0
        total_sum = (
            res["p_0_to_5_deg"] + res["p_5_to_10_deg"] +
            res["p_10_to_15_deg"] + res["p_gt_15_deg"]
        )
        assert pytest.approx(total_sum, abs=1e-4) == 100.0

        # Formula: 100.0 * (1.00*(0.40) + 0.65*(0.30) + 0.20*(0.20) + 0.00*(0.10)) = 63.5
        expected_score = round(100.0 * (0.40 * 1.00 + 0.30 * 0.65 + 0.20 * 0.20 + 0.10 * 0.00), 2)
        assert pytest.approx(res["slope_distribution_score"], abs=1e-2) == expected_score

    def test_disjoint_slope_categories_nan_masking_normalization(self):
        """Valid-cell normalization: NaNs (e.g. outside DEM or NoData) are excluded and valid cells sum to 100%."""
        arr = np.array([2.0, 7.0, np.nan, 14.0, np.nan, 22.0], dtype=np.float32)
        res = calculate_disjoint_slope_categories(arr)
        assert res["valid_cells"] == 4
        assert pytest.approx(res["p_0_to_5_deg"], abs=1e-4) == 25.0
        assert pytest.approx(res["p_5_to_10_deg"], abs=1e-4) == 25.0
        assert pytest.approx(res["p_10_to_15_deg"], abs=1e-4) == 25.0
        assert pytest.approx(res["p_gt_15_deg"], abs=1e-4) == 25.0
        total_sum = (
            res["p_0_to_5_deg"] + res["p_5_to_10_deg"] +
            res["p_10_to_15_deg"] + res["p_gt_15_deg"]
        )
        assert pytest.approx(total_sum, abs=1e-4) == 100.0

    def test_assess_landing_area_includes_phase4_disjoint_stats(self):
        """assess_landing_area includes disjoint_coverage_pct and slope_distribution_score."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        res = assess_landing_area(raster, center_projected=(0.0, 0.0), radius_m=500.0)
        assert "disjoint_coverage_pct" in res["slope_stats"]
        disjoint = res["slope_stats"]["disjoint_coverage_pct"]
        total_sum = (
            disjoint["p_0_to_5_deg"] + disjoint["p_5_to_10_deg"] +
            disjoint["p_10_to_15_deg"] + disjoint["p_gt_15_deg"]
        )
        assert pytest.approx(total_sum, abs=1e-4) == 100.0
        assert 0.0 <= res["slope_stats"]["slope_distribution_score"] <= 100.0


class TestCliffBufferDistance:
    """Validate cliff-buffer distance extraction using raster pixel spacing."""

    def test_cliff_detected_at_known_distance(self):
        """Create a synthetic cliff wall (>=20 deg) at a known distance and verify detection."""
        w, h = 101, 101
        elev = np.zeros((h, w), dtype=np.float32)
        elev[:, 57:] = 300.0

        meta = DemMetadata(
            dataset_name="cliff_test_dem",
            data_source="Synthetic Test",
            file_path="memory://cliff_test.tif",
            width_pixels=w,
            height_pixels=h,
            pixel_size_x_m=100.0,
            pixel_size_y_m=100.0,
            crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
            bounds_projected=(-5050.0, -5050.0, 5050.0, 5050.0),
            elevation_units="meters",
            reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
            nodata_value=-9999.0,
            valid_data_percent=100.0,
            min_elevation_m=0.0,
            max_elevation_m=300.0,
            mean_elevation_m=50.0,
            is_synthetic=True
        )
        dem = DemRaster(elev, meta)

        res = calculate_cliff_buffer_distance(
            dem=dem,
            center_projected=(0.0, 0.0),
            cliff_slope_threshold_deg=20.0,
            max_search_radius_m=1500.0
        )
        assert res["status"] == "CLIFF_DETECTED"
        assert res["has_cliff_in_radius"] is True
        assert res["cliff_distance_m"] is not None
        assert 500.0 <= res["cliff_distance_m"] <= 800.0
        assert res["nearest_cliff_coord"] is not None
        assert res["nearest_cliff_slope_deg"] >= 20.0

    def test_no_cliff_within_search_radius(self):
        """DEM with modest slopes (<15 deg) returns NO_CLIFF_WITHIN_SEARCH_RADIUS."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        res = calculate_cliff_buffer_distance(
            dem=raster,
            center_projected=(0.0, 0.0),
            cliff_slope_threshold_deg=20.0,
            max_search_radius_m=300.0
        )
        assert res["status"] == "NO_CLIFF_WITHIN_SEARCH_RADIUS"
        assert res["has_cliff_in_radius"] is False
        assert res["cliff_distance_m"] is None

    def test_center_outside_dem(self):
        """Center outside DEM boundaries returns CENTER_OUTSIDE_DEM."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        res = calculate_cliff_buffer_distance(
            dem=raster,
            center_projected=(20000.0, 20000.0),
            max_search_radius_m=1000.0
        )
        assert res["status"] == "CENTER_OUTSIDE_DEM"
        assert res["has_cliff_in_radius"] is False
        assert res["cliff_distance_m"] is None

    def test_dem_raster_convenience_method(self):
        """DemRaster.calculate_cliff_buffer_distance matches standalone function."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        res = raster.calculate_cliff_buffer_distance(center_projected=(0.0, 0.0), max_search_radius_m=500.0)
        assert "status" in res
        assert "search_radius_m" in res


class TestEvaluateSiteTerrainSafety:
    """Validate GIS-to-MCDA metrics adapter evaluate_site_terrain_safety."""

    def test_evaluate_site_terrain_safety_compliant(self):
        """Gentle terrain satisfies all hard safety constraints."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        assessment = evaluate_site_terrain_safety(
            dem=raster,
            center_coord=(0.0, 0.0),
            landing_radius_m=500.0,
            max_acceptable_slope_deg=15.0,
            min_dem_coverage_percent=75.0,
            cliff_search_radius_m=500.0
        )
        assert assessment["provenance"]["terrain_elevation"] == "MEASURED (NASA LOLA DEM)"
        assert assessment["units"]["slope"] == "degrees"
        assert assessment["units"]["distances"] == "meters"
        assert assessment["is_safety_gated"] is True
        assert assessment["is_slope_compliant"] is True
        assert assessment["is_coverage_compliant"] is True
        assert assessment["dem_coverage_percent"] == 100.0
        assert assessment["terrain_safety_score"] is not None

    def test_evaluate_site_terrain_safety_slope_violation(self):
        """Steep terrain fails slope constraint."""
        w, h = 50, 50
        elev = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            elev[i, :] = i * 100.0 * math.tan(math.radians(25.0))

        meta = DemMetadata(
            dataset_name="steep_dem",
            data_source="Test",
            file_path="memory://steep.tif",
            width_pixels=w,
            height_pixels=h,
            pixel_size_x_m=100.0,
            pixel_size_y_m=100.0,
            crs_repr="PROJCS[\"Lunar_South_Pole_Stereographic\",UNIT[\"metre\",1]]",
            bounds_projected=(-2500.0, -2500.0, 2500.0, 2500.0),
            elevation_units="meters",
            reference_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
            nodata_value=-9999.0,
            valid_data_percent=100.0,
            min_elevation_m=0.0,
            max_elevation_m=float(np.max(elev)),
            mean_elevation_m=float(np.mean(elev)),
            is_synthetic=True
        )
        dem = DemRaster(elev, meta)

        assessment = evaluate_site_terrain_safety(
            dem=dem,
            center_coord=(0.0, 0.0),
            landing_radius_m=500.0,
            max_acceptable_slope_deg=15.0
        )
        assert assessment["is_slope_compliant"] is False
        assert assessment["is_safety_gated"] is False
        assert len(assessment["violations"]) > 0

    def test_evaluate_site_terrain_safety_low_coverage(self):
        """Site located near DEM edge with < 75% coverage fails coverage constraint."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        assessment = evaluate_site_terrain_safety(
            dem=raster,
            center_coord=(4800.0, 0.0),
            landing_radius_m=1000.0,
            min_dem_coverage_percent=75.0
        )
        assert assessment["is_coverage_compliant"] is False
        assert assessment["is_safety_gated"] is False
        assert any("coverage" in v.lower() for v in assessment["violations"])

