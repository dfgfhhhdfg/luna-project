"""
Unit & Integration Tests for LUNA NAVIGATOR — Terrain & GIS Engine (Member 3)
Tests polar stereographic projections, LOLA DEM sampling, 360° ray-tracing horizon masks,
solar/Earth topographic occultations, and surface-to-surface line-of-sight viewshed analysis.
"""

import pytest
import math
from fastapi.testclient import TestClient
from app.main import create_application
from app.core.constants import MOON_MEAN_RADIUS_M
from app.core.coordinates import lunar_curvature_drop
from app.services.geospatial.projection import LunarPolarStereographic, lola_polar_projection
from app.services.geospatial.dem_sampler import dem_model, LolaSouthPoleDemModel
from app.engines.terrain.horizon import terrain_engine
from app.schemas.terrain import (
    TerrainHorizonRequest,
    SolarObstructionRequest,
    EarthObstructionRequest,
    SurfaceViewshedRequest
)

app = create_application()
client = TestClient(app)


class TestLunarPolarStereographic:
    """Validate conformal polar stereographic projection equations."""

    def test_pole_maps_to_origin(self):
        proj = LunarPolarStereographic()
        x, y = proj.forward(-90.0, 0.0)
        assert abs(x) < 1e-3
        assert abs(y) < 1e-3

    def test_round_trip_accuracy(self):
        proj = LunarPolarStereographic()
        test_points = [
            (-89.78, 125.0),   # Shackleton Crater Rim
            (-89.44, 222.5),   # Connecting Ridge
            (-85.99, 2.91),    # Malapert Mountain Massif
            (-88.52, 271.4),   # de Gerlache Rim 1
            (-87.30, 77.0),    # Faustini Rim
        ]
        for lat, lon in test_points:
            x, y = proj.forward(lat, lon)
            lat_rec, lon_rec = proj.inverse(x, y)
            assert pytest.approx(lat, abs=1e-5) == lat_rec
            assert pytest.approx(lon % 360.0, abs=1e-5) == (lon_rec % 360.0)

    def test_northern_hemisphere_rejection(self):
        proj = LunarPolarStereographic()
        with pytest.raises(ValueError):
            proj.forward(10.0, 45.0)


class TestLolaDemSampler:
    """Validate LOLA digital elevation model sampling and geomorphic metrics."""

    def test_shackleton_rim_elevation(self):
        # Shackleton Rim is calibrated at ~1240 meters above datum
        elev = dem_model.get_elevation_m(-89.78, 125.0)
        assert 1100.0 <= elev <= 1400.0

    def test_malapert_massif_peak_elevation(self):
        # Malapert Mountain is one of the highest peaks near the pole (~5050 m)
        elev = dem_model.get_elevation_m(-85.99, 2.91)
        assert elev >= 4500.0

    def test_connecting_ridge_elevation(self):
        elev = dem_model.get_elevation_m(-89.44, 222.5)
        assert 700.0 <= elev <= 1000.0

    def test_terrain_metrics_slope_and_roughness(self):
        metrics = dem_model.sample_terrain_metrics(-89.78, 125.0)
        assert metrics.slope_deg >= 0.0
        assert 0.0 <= metrics.aspect_deg <= 360.0
        assert metrics.roughness_m >= 0.0
        assert len(metrics.terrain_classification) > 0

    def test_feature_identification(self):
        feat = dem_model.identify_feature(-89.78, 125.0)
        assert "Shackleton" in feat or "Highlands" in feat

        malapert_feat = dem_model.identify_feature(-85.99, 2.91)
        assert "Malapert" in malapert_feat


class TestTerrainHorizonRayTracing:
    """Validate 360-degree radial ray-tracing and obstruction detection."""

    def test_lunar_curvature_drop_benchmark(self):
        # Ground truth: Delta_h = s^2 / (2 * R_moon)
        # At s = 30 km: (30,000)^2 / (2 * 1,737,400) = ~259.0 meters
        drop_30km = lunar_curvature_drop(30000.0)
        assert pytest.approx(259.0, abs=1.0) == drop_30km

        # At s = 60 km: (60,000)^2 / (2 * 1,737,400) = ~1036.0 meters
        drop_60km = lunar_curvature_drop(60000.0)
        assert pytest.approx(1036.0, abs=2.0) == drop_60km

    def test_generate_horizon_profile_points_count(self):
        req = TerrainHorizonRequest(
            latitude=-89.78,
            longitude=125.0,
            elevation_m=1240.0,
            max_distance_km=40.0,
            azimuth_step_deg=5.0,
            observer_height_m=2.0
        )
        res = terrain_engine.generate_horizon_profile(req)
        # 360 / 5 = 72 azimuth points
        assert len(res.horizon_points) == 72
        assert res.elevation_m == 1240.0
        assert res.max_horizon_elevation_deg >= res.min_horizon_elevation_deg
        # Local polar horizon angles should stay within reasonable bounds (-5° to +10°)
        assert -5.0 <= res.min_horizon_elevation_deg <= 5.0
        assert 0.0 <= res.max_horizon_elevation_deg <= 12.0

    def test_horizon_interpolation(self):
        req = TerrainHorizonRequest(
            latitude=-89.78,
            longitude=125.0,
            elevation_m=1240.0,
            max_distance_km=40.0,
            azimuth_step_deg=2.0
        )
        horizon = terrain_engine.generate_horizon_profile(req)
        el_at_45 = terrain_engine.get_horizon_elevation_at_azimuth(horizon, 45.0)
        el_at_45_5 = terrain_engine.get_horizon_elevation_at_azimuth(horizon, 45.5)
        # Continuous interpolation: difference between 45.0° and 45.5° should be small
        assert abs(el_at_45 - el_at_45_5) < 1.0


class TestSolarAndEarthObstruction:
    """Validate solar occultation and Earth telecommunications line-of-sight checks."""

    @pytest.fixture
    def sample_horizon(self):
        req = TerrainHorizonRequest(
            latitude=-89.78,
            longitude=125.0,
            elevation_m=1240.0,
            max_distance_km=30.0,
            azimuth_step_deg=5.0
        )
        return terrain_engine.generate_horizon_profile(req)

    def test_solar_obstruction_full_sun(self, sample_horizon):
        # High solar elevation (+10°) will clear all terrain
        result = terrain_engine.calculate_solar_obstruction(sample_horizon, 90.0, 10.0)
        assert result.illumination_state == "SUNLIT"
        assert result.illumination_fraction == 1.0
        assert result.effective_flux_w_m2 > 0.0
        assert result.blocking_feature is None

    def test_solar_obstruction_total_shadow(self, sample_horizon):
        # Deep negative solar elevation (-10°) is totally shadowed
        result = terrain_engine.calculate_solar_obstruction(sample_horizon, 90.0, -10.0)
        assert result.illumination_state == "SHADOW"
        assert result.illumination_fraction == 0.0
        assert result.effective_flux_w_m2 == 0.0

    def test_earth_obstruction_clear_vs_blocked(self, sample_horizon):
        # Earth high above horizon (+12°)
        clear_res = terrain_engine.calculate_earth_obstruction(sample_horizon, 180.0, 12.0)
        assert clear_res.los_state == "CLEAR"
        assert "NOMINAL" in clear_res.link_margin_status

        # Earth below horizon (-5°)
        blocked_res = terrain_engine.calculate_earth_obstruction(sample_horizon, 180.0, -5.0)
        assert blocked_res.los_state == "BLOCKED"
        assert "OCCULTED" in blocked_res.link_margin_status


class TestSurfaceViewshed:
    """Validate line-of-sight ray-tracing between two surface stations."""

    def test_viewshed_between_close_points(self):
        req = SurfaceViewshedRequest(
            site_a_lat=-89.78,
            site_a_lon=125.0,
            mast_height_a_m=3.0,
            site_b_lat=-89.77,
            site_b_lon=125.1,
            mast_height_b_m=2.0
        )
        res = terrain_engine.calculate_viewshed(req)
        assert res.total_distance_km > 0.0
        assert len(res.profile) > 0
        assert isinstance(res.is_line_of_sight_clear, bool)


class TestTerrainEndpointsIntegration:
    """Verify FastAPI integration for Member 3 terrain endpoints."""

    def test_post_terrain_horizon_api(self):
        payload = {
            "latitude": -89.78,
            "longitude": 125.0,
            "elevation_m": 1240.0,
            "max_distance_km": 30.0,
            "azimuth_step_deg": 10.0,
            "observer_height_m": 2.0
        }
        res = client.post("/api/terrain/horizon", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["horizon_points"]) == 36
        assert "elevation_datum" in data
        assert "site_slope_deg" in data

    def test_post_solar_obstruction_api(self):
        payload = {
            "latitude": -89.78,
            "longitude": 125.0,
            "elevation_m": 1240.0,
            "observer_height_m": 2.0,
            "sun_azimuth_deg": 45.0,
            "sun_elevation_deg": 2.5,
            "max_distance_km": 30.0
        }
        res = client.post("/api/terrain/solar-obstruction", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["illumination_state"] in ["SUNLIT", "PENUMBRA", "SHADOW"]
        assert "effective_flux_w_m2" in data

    def test_post_earth_obstruction_api(self):
        payload = {
            "latitude": -89.78,
            "longitude": 125.0,
            "elevation_m": 1240.0,
            "observer_height_m": 2.0,
            "earth_azimuth_deg": 310.0,
            "earth_elevation_deg": 5.0,
            "max_distance_km": 30.0
        }
        res = client.post("/api/terrain/earth-obstruction", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["los_state"] in ["CLEAR", "PARTIAL_GRAZING", "BLOCKED"]
        assert "link_margin_status" in data

    def test_post_surface_viewshed_api(self):
        payload = {
            "site_a_lat": -89.78,
            "site_a_lon": 125.0,
            "mast_height_a_m": 3.0,
            "site_b_lat": -89.77,
            "site_b_lon": 125.1,
            "mast_height_b_m": 2.0
        }
        res = client.post("/api/terrain/viewshed", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "is_line_of_sight_clear" in data
        assert len(data["profile"]) > 0

    def test_get_terrain_sample_api(self):
        res = client.get("/api/terrain/sample?latitude=-89.78&longitude=125.0")
        assert res.status_code == 200
        data = res.json()
        assert "metrics" in data
        assert "slope_deg" in data["metrics"]
        assert "closest_feature" in data
