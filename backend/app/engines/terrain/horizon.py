"""
LUNA NAVIGATOR — Terrain Horizon & Line-of-Sight Ray Tracing Engine (Member 3 Ownership)

Computes 360-degree local terrain horizon elevation profile, obstruction masks,
and line-of-sight ray-tracing for Sun and Earth visibility using NASA LOLA DEM data
and spherical lunar curvature correction.

Mathematical Formulation:
- Lunar Curvature Drop: Delta_h = s^2 / (2 * R_moon) where R_moon = 1,737,400 meters.
- Local Horizon Angle: H(theta) = max_{s in [s_min, s_max]} arctan((h(s) - h_obs - Delta_h) / s)
- Solar Semi-diameter: delta_sun = 0.267° (16 arcmin).
- Earth Semi-diameter: delta_earth = 0.95° (~1.9° full angular disk).
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from app.core.constants import (
    MOON_MEAN_RADIUS_M,
    MOON_MEAN_RADIUS_KM,
    SOLAR_CONSTANT_1AU_W_M2
)
from app.core.coordinates import (
    lunar_curvature_drop,
    great_circle_distance_m
)
from app.services.geospatial.dem_sampler import (
    dem_model,
    LolaSouthPoleDemModel,
    TerrainMetrics
)
from app.schemas.terrain import (
    TerrainHorizonRequest,
    HorizonPoint,
    TerrainHorizonResponse,
    SolarObstructionResult,
    EarthObstructionResult,
    SurfaceViewshedRequest,
    SurfaceViewshedResponse,
    ViewshedProfilePoint
)


class TerrainHorizonEngine:
    """
    High-precision Lunar Terrain Ray-Tracing Engine.
    Samples digital elevation models along radial azimuth vectors with curvature correction.
    """

    def __init__(self, dem: Optional[LolaSouthPoleDemModel] = None):
        self.dem = dem or dem_model

    def calculate_ray_waypoint(
        self,
        lat0_deg: float,
        lon0_deg: float,
        azimuth_deg: float,
        distance_m: float
    ) -> Tuple[float, float]:
        """
        Calculates the selenographic coordinates (Lat, Lon) of a point at distance_m
        along a given azimuth_deg from the observer using spherical geodesy.
        """
        phi0 = math.radians(lat0_deg)
        lam0 = math.radians(lon0_deg)
        theta = math.radians(azimuth_deg)
        delta = distance_m / MOON_MEAN_RADIUS_M

        sin_phi = math.sin(phi0) * math.cos(delta) + math.cos(phi0) * math.sin(delta) * math.cos(theta)
        phi = math.asin(max(-1.0, min(1.0, sin_phi)))

        y = math.sin(theta) * math.sin(delta) * math.cos(phi0)
        x = math.cos(delta) - math.sin(phi0) * math.sin(phi)
        lam = lam0 + math.atan2(y, x)

        lat_res = math.degrees(phi)
        lon_res = math.degrees(lam) % 360.0

        return (lat_res, lon_res)

    def generate_horizon_profile(self, request: TerrainHorizonRequest) -> TerrainHorizonResponse:
        """
        Computes 360-degree local terrain horizon elevation profile and ray-tracing obstruction masks
        from NASA LOLA elevation models with lunar curvature correction.
        """
        obs_elev = request.elevation_m
        if obs_elev == 0.0 or obs_elev is None:
            # Sample actual LOLA DEM elevation if default/unspecified
            obs_elev = self.dem.get_elevation_m(request.latitude, request.longitude)

        total_obs_height = obs_elev + request.observer_height_m

        # Dynamic ray-stepping resolution
        max_dist_m = request.max_distance_km * 1000.0
        # Adaptive step: 250m step provides high fidelity across 60-100km radii
        radial_step_m = max(150.0, min(500.0, max_dist_m / 240.0))

        distances = np.arange(radial_step_m, max_dist_m + 1.0, radial_step_m)
        curv_drops = (distances ** 2) / (2.0 * MOON_MEAN_RADIUS_M)

        horizon_points: List[HorizonPoint] = []
        azimuth = 0.0

        phi0 = math.radians(request.latitude)
        lam0 = math.radians(request.longitude)
        sin_phi0 = math.sin(phi0)
        cos_phi0 = math.cos(phi0)
        deltas = distances / MOON_MEAN_RADIUS_M
        cos_deltas = np.cos(deltas)
        sin_deltas = np.sin(deltas)

        while azimuth < 360.0:
            theta = math.radians(azimuth)
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            sin_phi = sin_phi0 * cos_deltas + cos_phi0 * sin_deltas * cos_theta
            phis = np.arcsin(np.clip(sin_phi, -1.0, 1.0))
            ys = sin_theta * sin_deltas * cos_phi0
            xs = cos_deltas - sin_phi0 * sin_phi
            lams = (lam0 + np.arctan2(ys, xs)) % (2.0 * math.pi)

            w_lats = np.degrees(phis)
            w_lons = np.degrees(lams)

            h_terrains = self.dem.get_elevations_vectorized(w_lats, w_lons)
            elev_diffs = h_terrains - total_obs_height - curv_drops
            angles_rad = np.arctan2(elev_diffs, distances)

            max_idx = int(np.argmax(angles_rad))
            max_angle_rad = angles_rad[max_idx]
            best_dist_km = distances[max_idx] / 1000.0
            best_feature = self.dem.identify_feature(float(w_lats[max_idx]), float(w_lons[max_idx]))

            horizon_elev_deg = round(float(np.degrees(max_angle_rad)), 2)
            point = HorizonPoint(
                azimuth_deg=round(azimuth, 1),
                horizon_elevation_deg=horizon_elev_deg,
                obstruction_distance_km=round(best_dist_km, 1),
                blocking_feature=best_feature
            )
            horizon_points.append(point)
            azimuth += request.azimuth_step_deg

        elevations = [p.horizon_elevation_deg for p in horizon_points]
        max_el = max(elevations) if elevations else 0.0
        min_el = min(elevations) if elevations else 0.0
        avg_el = round(sum(elevations) / len(elevations), 2) if elevations else 0.0

        # Sample local slope and geomorphic characteristics at the candidate site
        metrics = self.dem.sample_terrain_metrics(request.latitude, request.longitude)

        return TerrainHorizonResponse(
            latitude=request.latitude,
            longitude=request.longitude,
            elevation_m=round(obs_elev, 2),
            observer_height_m=request.observer_height_m,
            elevation_datum="Lunar Sphere R=1737.4 km (NASA LOLA)",
            horizon_points=horizon_points,
            max_horizon_elevation_deg=round(max_el, 2),
            min_horizon_elevation_deg=round(min_el, 2),
            average_horizon_elevation_deg=avg_el,
            site_slope_deg=round(metrics.slope_deg, 2),
            site_aspect_deg=round(metrics.aspect_deg, 1),
            site_roughness_m=round(metrics.roughness_m, 2),
            terrain_classification=metrics.terrain_classification
        )

    def get_horizon_elevation_at_azimuth(
        self,
        horizon_response: TerrainHorizonResponse,
        azimuth_deg: float
    ) -> float:
        """
        Fast lookup of horizon obstruction elevation angle at a specific azimuth angle
        with linear interpolation.
        """
        az_normalized = azimuth_deg % 360.0
        points = horizon_response.horizon_points
        if not points:
            return 0.0

        step = points[1].azimuth_deg - points[0].azimuth_deg if len(points) > 1 else 1.0
        idx_lower = int(az_normalized / step) % len(points)
        idx_upper = (idx_lower + 1) % len(points)

        p1 = points[idx_lower]
        p2 = points[idx_upper]

        # Linear interpolation between nearest angular azimuth steps
        fraction = (az_normalized - p1.azimuth_deg) / step if step > 0 else 0.0
        interp_elevation = p1.horizon_elevation_deg + fraction * (p2.horizon_elevation_deg - p1.horizon_elevation_deg)

        return float(interp_elevation)

    def calculate_solar_obstruction(
        self,
        horizon_response: TerrainHorizonResponse,
        sun_azimuth_deg: float,
        sun_elevation_deg: float
    ) -> SolarObstructionResult:
        """
        Calculates terrain-induced solar occultation accounting for the Sun's finite angular disk.
        Solar angular semi-diameter = 0.267° (16 arcminutes).
        """
        SOLAR_SEMI_DIAMETER_DEG = 0.2667  # 16 arcmin

        horizon_el = self.get_horizon_elevation_at_azimuth(horizon_response, sun_azimuth_deg)
        clearance_deg = sun_elevation_deg - horizon_el

        if clearance_deg >= SOLAR_SEMI_DIAMETER_DEG:
            # Entire solar disk is above terrain horizon
            state = "SUNLIT"
            illum_fraction = 1.0
        elif clearance_deg <= -SOLAR_SEMI_DIAMETER_DEG:
            # Entire solar disk is occulted behind terrain (Umbra)
            state = "SHADOW"
            illum_fraction = 0.0
        else:
            # Partial occultation: Sun disk intersects ridge crest (Penumbra)
            state = "PENUMBRA"
            # Linear fraction approximation across solar diameter
            illum_fraction = (clearance_deg + SOLAR_SEMI_DIAMETER_DEG) / (2.0 * SOLAR_SEMI_DIAMETER_DEG)
            illum_fraction = max(0.0, min(1.0, illum_fraction))

        # Effective incident solar flux on horizontal surface
        # S_eff = S0 * fraction * sin(elevation)
        sin_el = max(0.0, math.sin(math.radians(max(0.0, sun_elevation_deg))))
        effective_flux_w_m2 = SOLAR_CONSTANT_1AU_W_M2 * illum_fraction * sin_el

        # Find closest point on horizon for feature citation
        step = horizon_response.horizon_points[1].azimuth_deg - horizon_response.horizon_points[0].azimuth_deg
        idx = int((sun_azimuth_deg % 360.0) / step) % len(horizon_response.horizon_points)
        blocking_feature = horizon_response.horizon_points[idx].blocking_feature if state != "SUNLIT" else None

        return SolarObstructionResult(
            sun_azimuth_deg=round(sun_azimuth_deg, 2),
            sun_elevation_deg=round(sun_elevation_deg, 2),
            horizon_elevation_deg=round(horizon_el, 2),
            clearance_deg=round(clearance_deg, 2),
            illumination_state=state,
            illumination_fraction=round(illum_fraction, 3),
            effective_flux_w_m2=round(effective_flux_w_m2, 2),
            blocking_feature=blocking_feature
        )

    def calculate_earth_obstruction(
        self,
        horizon_response: TerrainHorizonResponse,
        earth_azimuth_deg: float,
        earth_elevation_deg: float
    ) -> EarthObstructionResult:
        """
        Calculates terrain occultation for Earth communication line-of-sight.
        Earth angular semi-diameter = 0.95° (~1.9° full disk).
        """
        EARTH_SEMI_DIAMETER_DEG = 0.95

        horizon_el = self.get_horizon_elevation_at_azimuth(horizon_response, earth_azimuth_deg)
        clearance_deg = earth_elevation_deg - horizon_el

        if clearance_deg >= EARTH_SEMI_DIAMETER_DEG:
            state = "CLEAR"
            link_margin = "NOMINAL (Unobstructed Direct-to-Earth LOS)"
        elif clearance_deg <= -EARTH_SEMI_DIAMETER_DEG:
            state = "BLOCKED"
            link_margin = "OCCULTED (Terrain Masking, Relay Required)"
        else:
            state = "PARTIAL_GRAZING"
            link_margin = "MARGINAL (Topographic Grazing / Multipath Risk)"

        step = horizon_response.horizon_points[1].azimuth_deg - horizon_response.horizon_points[0].azimuth_deg
        idx = int((earth_azimuth_deg % 360.0) / step) % len(horizon_response.horizon_points)
        blocking_feature = horizon_response.horizon_points[idx].blocking_feature if state != "CLEAR" else None

        return EarthObstructionResult(
            earth_azimuth_deg=round(earth_azimuth_deg, 2),
            earth_elevation_deg=round(earth_elevation_deg, 2),
            horizon_elevation_deg=round(horizon_el, 2),
            clearance_deg=round(clearance_deg, 2),
            los_state=state,
            link_margin_status=link_margin,
            blocking_feature=blocking_feature
        )

    def calculate_viewshed(self, req: SurfaceViewshedRequest) -> SurfaceViewshedResponse:
        """
        Calculates line-of-sight clearance between two surface stations (e.g. lander to rover,
        or surface asset to relay mast) across intermediate lunar terrain.
        """
        # Great-circle distance between Station A and Station B
        total_dist_m = great_circle_distance_m(
            req.site_a_lat, req.site_a_lon, req.site_b_lat, req.site_b_lon
        )

        h_a = self.dem.get_elevation_m(req.site_a_lat, req.site_a_lon) + req.mast_height_a_m
        h_b = self.dem.get_elevation_m(req.site_b_lat, req.site_b_lon) + req.mast_height_b_m

        steps = max(20, min(200, int(total_dist_m / 250.0)))
        fractions = np.linspace(0.0, 1.0, num=steps)

        profile: List[ViewshedProfilePoint] = []
        is_clear = True
        min_clearance = float("inf")
        blocking_dist_km = None
        blocking_feature = None

        # Sample terrain and compute line-of-sight clearance along the transect
        for frac in fractions:
            curr_dist_m = frac * total_dist_m
            
            # Intermediate point coordinates on great circle
            lat_interp = req.site_a_lat + frac * (req.site_b_lat - req.site_a_lat)
            lon_interp = req.site_a_lon + frac * (req.site_b_lon - req.site_a_lon)

            # Ground elevation at waypoint
            terrain_h = self.dem.get_elevation_m(lat_interp, lon_interp)

            # Spherical chord LOS height between Station A and Station B:
            # Linear interpolation between endpoint heights minus curvature drop along the chord
            los_nominal_h = h_a + frac * (h_b - h_a)
            # Curvature drop relative to the chord: Delta_h = s * (D - s) / (2 * R)
            chord_curvature_drop = (curr_dist_m * (total_dist_m - curr_dist_m)) / (2.0 * MOON_MEAN_RADIUS_M)
            los_actual_h = los_nominal_h - chord_curvature_drop

            clearance_m = los_actual_h - terrain_h

            if clearance_m < min_clearance:
                min_clearance = clearance_m

            if clearance_m < 0.0:
                is_clear = False
                if blocking_dist_km is None:
                    blocking_dist_km = round(curr_dist_m / 1000.0, 2)
                    blocking_feature = self.dem.identify_feature(lat_interp, lon_interp)

            profile.append(
                ViewshedProfilePoint(
                    distance_km=round(curr_dist_m / 1000.0, 2),
                    terrain_elevation_m=round(terrain_h, 2),
                    los_elevation_m=round(los_actual_h, 2),
                    clearance_m=round(clearance_m, 2)
                )
            )

        return SurfaceViewshedResponse(
            is_line_of_sight_clear=is_clear,
            total_distance_km=round(total_dist_m / 1000.0, 2),
            min_clearance_m=round(min_clearance, 2),
            blocking_distance_km=blocking_dist_km,
            blocking_feature=blocking_feature,
            profile=profile
        )


# Singleton engine instance
terrain_engine = TerrainHorizonEngine()
