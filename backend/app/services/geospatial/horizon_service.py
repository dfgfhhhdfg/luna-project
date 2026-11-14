"""
LUNA NAVIGATOR — Lunar Terrain Horizon & Line-of-Sight Ray Tracing Engine (Member 3 Ownership)

Scientific computing service for:
1. Lunar spherical curvature drop calculation (Delta_h = s^2 / (2 * R_moon)).
2. Vectorized terrain ray tracing along geodetic (great circle) and projected planar trajectories.
3. Continuous bilinear and nearest elevation interpolation with strict NoData / boundary tracking.
4. Tri-state line-of-sight visibility evaluation:
   - 'obstructed': Confirmed topographic obstruction along the ray.
   - 'clear': Ray fully traversed to max search distance with all terrain below target elevation.
   - 'unknown': Ray truncated by DEM boundary or NoData before proving clearance.
5. 360-degree local terrain horizon profiling with angular extrema and periodic azimuth interpolation.

Conventions:
- Horizontal Coordinate Reference: Lunar South Pole Polar Stereographic (meters)
- Elevation Datum: Mean Lunar Sphere R_moon = 1,737,400.0 meters (NASA LOLA standard)
- Topocentric Frame: Azimuth measured clockwise from Lunar True North [0°, 360°)
  (0° = North, 90° = East, 180° = South, 270° = West)
- Elevation Angle: Measured in degrees above the local horizontal tangent plane [-90°, +90°]
"""

import math
from typing import Tuple, Dict, Any, Optional, List, Union, Sequence
import numpy as np

from app.core.constants import MOON_MEAN_RADIUS_M
from app.services.geospatial.projection import lola_polar_projection
from app.services.geospatial.dem_loader import DemRaster


class HorizonServiceError(Exception):
    """Base exception for horizon and line-of-sight service operations."""
    pass


# =============================================================================
# 1. Lunar Spherical Curvature Drop
# =============================================================================

def calculate_curvature_drop(
    distance_m: Union[float, np.ndarray],
    radius_m: float = MOON_MEAN_RADIUS_M
) -> Union[float, np.ndarray]:
    """
    Calculate the elevation drop (in meters) due to lunar spherical curvature
    using the second-order parabolic geodesy approximation:
    
    Formula: Delta_h = s^2 / (2 * R_moon)
    
    This approximates the exact radial secant drop R * (sec(s/R) - 1):
    - At 10 km: Parabolic = 28.7786m, Exact = 28.7790m (discrepancy: 0.4 mm)
    - At 30 km: Parabolic = 259.0077m, Exact = 259.0399m (discrepancy: 32.2 mm)
    - At 60 km: Parabolic = 1036.0309m, Exact = 1036.5459m (discrepancy: 51.5 cm)
    - At 100 km: Parabolic = 2877.8635m, Exact = 2881.8413m (discrepancy: 3.98 m)
    
    Args:
        distance_m: Ground arc distance in meters (scalar or numpy array).
        radius_m: Lunar mean radius in meters (default: 1,737,400.0 m).
        
    Returns:
        Curvature drop in meters.
    """
    if radius_m <= 0.0:
        raise ValueError(f"Lunar radius must be strictly positive: {radius_m}")
    
    s = np.asarray(distance_m, dtype=np.float64)
    if np.any(s < 0.0):
        raise ValueError("Ground distance cannot be negative.")
        
    drop = (s ** 2) / (2.0 * radius_m)
    if np.isscalar(distance_m):
        return float(drop)
    return drop


# =============================================================================
# 2. Ray Waypoint Trajectory Generation
# =============================================================================

def planar_ray_waypoints(
    x0_m: float,
    y0_m: float,
    azimuth_deg: float,
    distances_m: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate planar Cartesian coordinates (X, Y) in meters along a grid azimuth direction.
    
    NOTE ON GRID AZIMUTH VS TRUE GEODETIC NORTH:
    In planar mode, azimuth 0° aligns with the Cartesian +Y axis ("Grid North").
    In Lunar South Pole Polar Stereographic projection, the meridian lambda=0° lies along -Y
    and lambda=180° lies along +Y. Therefore, Grid North (+Y) differs from geodetic True
    North by the meridian convergence angle. For True North geodetic alignment (matching
    Member 2's ephemeris), use mode='spherical' (the default in mode='auto').
    
    Formula:
        x(s) = x0 + s * sin(theta_grid)
        y(s) = y0 + s * cos(theta_grid)
    """
    theta = math.radians(float(azimuth_deg) % 360.0)
    dists = np.asarray(distances_m, dtype=np.float64)
    
    xs = x0_m + dists * math.sin(theta)
    ys = y0_m + dists * math.cos(theta)
    return xs, ys


def geodesic_ray_waypoints(
    lat0_deg: float,
    lon0_deg: float,
    azimuth_deg: float,
    distances_m: np.ndarray,
    radius_m: float = MOON_MEAN_RADIUS_M
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate great-circle spherical waypoints along the lunar sphere and project
    them to Lunar South Polar Stereographic coordinates (X, Y in meters).
    
    Azimuth is measured clockwise from Lunar True North [0°, 360°).
    
    Returns:
        (xs_m, ys_m, lats_deg, lons_deg)
    """
    dists = np.asarray(distances_m, dtype=np.float64)
    az_norm = float(azimuth_deg) % 360.0
    theta = math.radians(az_norm)
    delta = dists / radius_m
    
    # Handle pole singularity (latitude = -90.0°)
    if abs(lat0_deg - (-90.0)) < 1e-6:
        # At south pole, every direction is North; azimuth directly specifies longitude
        lats_deg = -90.0 + np.degrees(delta)
        lons_deg = np.full_like(dists, az_norm)
    else:
        phi0 = math.radians(lat0_deg)
        lam0 = math.radians(lon0_deg)
        sin_phi0 = math.sin(phi0)
        cos_phi0 = math.cos(phi0)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)
        
        sin_phi = sin_phi0 * cos_delta + cos_phi0 * sin_delta * cos_theta
        phis = np.arcsin(np.clip(sin_phi, -1.0, 1.0))
        y_sph = sin_theta * sin_delta * cos_phi0
        x_sph = cos_delta - sin_phi0 * sin_phi
        lams = (lam0 + np.arctan2(y_sph, x_sph)) % (2.0 * math.pi)
        
        lats_deg = np.degrees(phis)
        lons_deg = np.degrees(lams)
        
    # Project spherical coordinates to Lunar Polar Stereographic (meters)
    colats_rad = np.radians(90.0 + lats_deg)
    rhos = 2.0 * radius_m * np.tan(colats_rad / 2.0)
    lams_proj_rad = np.radians(lons_deg) - lola_polar_projection.central_meridian_rad
    
    xs_m = rhos * np.sin(lams_proj_rad)
    ys_m = -rhos * np.cos(lams_proj_rad)
    
    return xs_m, ys_m, lats_deg, lons_deg


# =============================================================================
# 3. Vectorized Elevation Sampling with Strict NoData Handling
# =============================================================================

def sample_elevations_vectorized(
    dem: DemRaster,
    xs_m: np.ndarray,
    ys_m: np.ndarray,
    method: str = "bilinear"
) -> np.ndarray:
    """
    Vectorized elevation sampling across multiple planar coordinates.
    
    Supports:
    - 'bilinear': Continuous 4-point bilinear interpolation with strict NoData propagation
      (if any required neighbor is NoData or NaN, output is strictly NaN).
    - 'nearest': Discrete grid cell center lookup.
    
    Args:
        dem: Loaded DemRaster instance.
        xs_m: 1D array of projected Easting coordinates (meters).
        ys_m: 1D array of projected Northing coordinates (meters).
        method: 'bilinear' or 'nearest'.
        
    Returns:
        1D array of elevation values in meters (NaN for out-of-bounds or NoData).
    """
    xs = np.asarray(xs_m, dtype=np.float64)
    ys = np.asarray(ys_m, dtype=np.float64)
    out = np.full(xs.shape, np.nan, dtype=np.float64)
    
    left, bottom, right, top = dem.metadata.bounds_projected
    in_bounds = (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)
    if not np.any(in_bounds):
        return out
        
    w = dem.metadata.width_pixels
    h = dem.metadata.height_pixels
    
    # Affine transformation to continuous column (u) and row (v) pixel coordinates
    if dem.transform is not None:
        try:
            inv_t = ~dem.transform
            u = inv_t.a * xs + inv_t.b * ys + inv_t.c
            v = inv_t.d * xs + inv_t.e * ys + inv_t.f
        except Exception:
            u = (xs - left) / dem.metadata.pixel_size_x_m
            v = (top - ys) / dem.metadata.pixel_size_y_m
    else:
        u = (xs - left) / dem.metadata.pixel_size_x_m
        v = (top - ys) / dem.metadata.pixel_size_y_m
        
    elev = dem.elevation
    nodata = dem.metadata.nodata_value

    if method == "nearest":
        cols = np.floor(u).astype(np.int64)
        rows = np.floor(v).astype(np.int64)
        valid = in_bounds & (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        if np.any(valid):
            vals = elev[rows[valid], cols[valid]].astype(np.float64)
            if nodata is not None:
                vals[vals == nodata] = np.nan
            out[valid] = vals

    elif method == "bilinear":
        u_c = u - 0.5
        v_c = v - 0.5
        c0 = np.floor(u_c).astype(np.int64)
        r0 = np.floor(v_c).astype(np.int64)
        c1 = c0 + 1
        r1 = r0 + 1
        
        valid = in_bounds & (c0 >= 0) & (c1 < w) & (r0 >= 0) & (r1 < h)
        if np.any(valid):
            c0_v = c0[valid]
            c1_v = c1[valid]
            r0_v = r0[valid]
            r1_v = r1[valid]
            
            z00 = elev[r0_v, c0_v].astype(np.float64)
            z10 = elev[r0_v, c1_v].astype(np.float64)
            z01 = elev[r1_v, c0_v].astype(np.float64)
            z11 = elev[r1_v, c1_v].astype(np.float64)
            
            bad = np.isnan(z00) | np.isnan(z10) | np.isnan(z01) | np.isnan(z11)
            if nodata is not None:
                bad = bad | (z00 == nodata) | (z10 == nodata) | (z01 == nodata) | (z11 == nodata)
                
            alpha = u_c[valid] - c0_v
            beta = v_c[valid] - r0_v
            
            z_interp = (
                (1.0 - alpha) * (1.0 - beta) * z00 +
                alpha * (1.0 - beta) * z10 +
                (1.0 - alpha) * beta * z01 +
                alpha * beta * z11
            )
            z_interp[bad] = np.nan
            out[valid] = z_interp
    else:
        raise ValueError(f"Unsupported interpolation method '{method}'. Supported: 'bilinear', 'nearest'")
        
    return out


# =============================================================================
# 4. Terrain Ray Sampling
# =============================================================================

def sample_terrain_ray(
    dem: DemRaster,
    observer_coord: Optional[Tuple[float, float]] = None,
    azimuth_deg: float = 0.0,
    max_distance_m: float = 30000.0,
    radial_step_m: float = 100.0,
    observer_height_m: float = 2.0,
    observer_lat: Optional[float] = None,
    observer_lon: Optional[float] = None,
    observer_x_m: Optional[float] = None,
    observer_y_m: Optional[float] = None,
    interpolation_method: str = "bilinear",
    mode: str = "auto"
) -> Dict[str, Any]:
    """
    Sample a radial transect across lunar terrain from an observer along an azimuth direction.
    
    Computes:
    - Distance-dependent lunar curvature drop Delta_h = s^2 / (2 * R_moon).
    - Apparent elevation z_apparent = z(s) - Delta_h.
    - Topographic inclination angle alpha(s) = arctan((z_apparent(s) - z_obs) / s).
    - Peak horizon obstruction angle and defining terrain distance.
    - Explicit detection of NoData cells and boundary departures.
    
    Args:
        dem: Loaded DemRaster instance.
        observer_coord: Optional (x, y) or (lat, lon) tuple.
        azimuth_deg: Azimuth angle clockwise from True North [0°, 360°).
        max_distance_m: Maximum ray distance in meters (must be > 0).
        radial_step_m: Radial step increment in meters (must be > 0).
        observer_height_m: Instrument/mast height above ground in meters (>= 0).
        observer_lat: Selenographic latitude in degrees [-90°, 0°].
        observer_lon: Selenographic longitude in degrees [0°, 360°).
        observer_x_m: Projected Easting in meters.
        observer_y_m: Projected Northing in meters.
        interpolation_method: 'bilinear' or 'nearest'.
        mode: 'auto', 'projected', or 'spherical'.
        
    Returns:
        Dictionary with samples, observer metrics, peak horizon angle, and coverage flags.
    """
    if max_distance_m <= 0.0:
        raise ValueError(f"max_distance_m must be strictly positive: {max_distance_m}")
    if radial_step_m <= 0.0:
        raise ValueError(f"radial_step_m must be strictly positive: {radial_step_m}")
    if observer_height_m < 0.0:
        raise ValueError(f"observer_height_m must be non-negative: {observer_height_m}")

    az_norm = float(azimuth_deg) % 360.0

    # 1. Resolve observer position and determine trajectory mode
    lat0, lon0 = None, None
    if observer_lat is not None and observer_lon is not None:
        lat0, lon0 = float(observer_lat), float(observer_lon)
        x0, y0 = lola_polar_projection.forward(lat0, lon0)
        if mode == "auto":
            mode = "spherical"
    elif observer_x_m is not None and observer_y_m is not None:
        x0, y0 = float(observer_x_m), float(observer_y_m)
        try:
            lat0, lon0 = lola_polar_projection.inverse(x0, y0)
            if mode == "auto":
                mode = "spherical"
        except Exception:
            if mode == "auto":
                mode = "projected"
    elif observer_coord is not None:
        c1, c2 = float(observer_coord[0]), float(observer_coord[1])
        b_left, b_bottom, b_right, b_top = dem.metadata.bounds_projected
        # Check if coordinates are geographic latitude/longitude in southern domain
        if -90.0 <= c1 <= 0.0 and 0.0 <= c2 <= 360.0 and (c1 < b_left or c1 > b_right):
            lat0, lon0 = c1, c2
            x0, y0 = lola_polar_projection.forward(lat0, lon0)
            if mode == "auto":
                mode = "spherical"
        else:
            x0, y0 = c1, c2
            try:
                lat0, lon0 = lola_polar_projection.inverse(x0, y0)
                if mode == "auto":
                    mode = "spherical"
            except Exception:
                if mode == "auto":
                    mode = "projected"
    else:
        raise ValueError("Observer coordinate must be provided.")

    warnings: List[str] = []

    # 2. Check observer surface elevation
    z_ground = dem.get_elevation_at_projected(x0, y0, method=interpolation_method)
    if math.isnan(z_ground):
        warnings.append("Observer coordinate is outside DEM coverage or on NoData.")
        return {
            "observer_projected": (round(x0, 2), round(y0, 2)),
            "observer_elevation_m": None,
            "observer_height_m": observer_height_m,
            "azimuth_deg": round(az_norm, 2),
            "max_distance_m": round(max_distance_m, 1),
            "radial_step_m": round(radial_step_m, 1),
            "max_horizon_angle_deg": None,
            "max_horizon_distance_m": None,
            "max_horizon_point": None,
            "samples": [],
            "evaluated_distance_m": 0.0,
            "has_boundary_exit": True,
            "has_nodata": True,
            "warnings": warnings,
            "mode": mode,
            "elevation_datum": dem.metadata.reference_datum
        }

    z_obs = z_ground + observer_height_m

    # 3. Generate radial distances
    distances = np.arange(radial_step_m, max_distance_m + 1e-4, radial_step_m)
    if len(distances) == 0:
        distances = np.array([max_distance_m], dtype=np.float64)

    # 4. Generate waypoints along ray
    lats, lons = None, None
    if mode == "spherical":
        if lat0 is None or lon0 is None:
            lat0, lon0 = lola_polar_projection.inverse(x0, y0)
        xs, ys, lats, lons = geodesic_ray_waypoints(lat0, lon0, az_norm, distances)
    else:
        xs, ys = planar_ray_waypoints(x0, y0, az_norm, distances)

    # 5. Vectorized elevation sampling
    elevs = sample_elevations_vectorized(dem, xs, ys, method=interpolation_method)
    curv_drops = calculate_curvature_drop(distances)
    app_elevs = elevs - curv_drops
    angles_rad = np.arctan2(app_elevs - z_obs, distances)
    angles_deg = np.degrees(angles_rad)

    # 6. Build sample points and track coverage
    samples = []
    has_boundary_exit = False
    has_nodata = False
    first_invalid_dist = None

    left, bottom, right, top = dem.metadata.bounds_projected
    nodata_val = dem.metadata.nodata_value

    for i in range(len(distances)):
        d = float(distances[i])
        px = float(xs[i])
        py = float(ys[i])
        curv = float(curv_drops[i])
        raw_z = elevs[i]

        out_of_bounds = not (left <= px <= right and bottom <= py <= top)
        if out_of_bounds:
            has_boundary_exit = True
            if first_invalid_dist is None:
                first_invalid_dist = d
        elif math.isnan(raw_z):
            has_nodata = True
            if first_invalid_dist is None:
                first_invalid_dist = d

        is_valid = not math.isnan(raw_z)
        if lats is not None:
            plat = float(lats[i])
            plon = float(lons[i])
        else:
            try:
                plat_inv, plon_inv = lola_polar_projection.inverse(px, py)
                plat = float(plat_inv)
                plon = float(plon_inv)
            except Exception:
                plat, plon = None, None

        samples.append({
            "distance_m": round(d, 1),
            "projected_x_m": round(px, 1),
            "projected_y_m": round(py, 1),
            "latitude_deg": round(plat, 5) if plat is not None else None,
            "longitude_deg": round(plon, 5) if plon is not None else None,
            "terrain_elevation_m": round(float(raw_z), 2) if is_valid else None,
            "apparent_elevation_m": round(float(app_elevs[i]), 2) if is_valid else None,
            "elevation_angle_deg": round(float(angles_deg[i]), 4) if is_valid else None,
            "curvature_drop_m": round(curv, 3),
            "is_valid": is_valid
        })

    # 7. Identify maximum horizon obstruction angle
    valid_mask = ~np.isnan(angles_deg)
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        max_sub_idx = int(np.argmax(angles_deg[valid_mask]))
        max_idx = valid_indices[max_sub_idx]
        
        max_angle = round(float(angles_deg[max_idx]), 4)
        max_dist = round(float(distances[max_idx]), 1)
        max_pt = {
            "distance_m": max_dist,
            "projected_x_m": round(float(xs[max_idx]), 1),
            "projected_y_m": round(float(ys[max_idx]), 1),
            "terrain_elevation_m": round(float(elevs[max_idx]), 2),
            "elevation_angle_deg": max_angle
        }
    else:
        max_angle = None
        max_dist = None
        max_pt = None

    if has_boundary_exit:
        warnings.append(f"Ray exited DEM boundary at distance {first_invalid_dist:.1f}m.")
    elif has_nodata:
        warnings.append(f"Ray encountered NoData cells at distance {first_invalid_dist:.1f}m.")

    eval_dist = float(first_invalid_dist) if first_invalid_dist is not None else float(distances[-1])

    return {
        "observer_projected": (round(x0, 2), round(y0, 2)),
        "observer_elevation_m": round(z_ground, 2),
        "observer_height_m": observer_height_m,
        "azimuth_deg": round(az_norm, 2),
        "max_distance_m": round(max_distance_m, 1),
        "radial_step_m": round(radial_step_m, 1),
        "max_horizon_angle_deg": max_angle,
        "max_horizon_distance_m": max_dist,
        "max_horizon_point": max_pt,
        "samples": samples,
        "evaluated_distance_m": round(eval_dist, 1),
        "has_boundary_exit": has_boundary_exit,
        "has_nodata": has_nodata,
        "warnings": warnings,
        "mode": mode,
        "elevation_datum": dem.metadata.reference_datum
    }


# =============================================================================
# 5. Tri-State Line-of-Sight Obstruction Evaluation
# =============================================================================

def check_line_of_sight(
    dem: DemRaster,
    target_azimuth_deg: float,
    target_elevation_deg: float,
    observer_coord: Optional[Tuple[float, float]] = None,
    max_distance_m: float = 60000.0,
    radial_step_m: float = 100.0,
    observer_height_m: float = 2.0,
    observer_lat: Optional[float] = None,
    observer_lon: Optional[float] = None,
    observer_x_m: Optional[float] = None,
    observer_y_m: Optional[float] = None,
    interpolation_method: str = "bilinear",
    mode: str = "auto"
) -> Dict[str, Any]:
    """
    Determine whether lunar terrain obstructs a target line-of-sight in a specified direction.
    
    Tri-state visibility logic:
    - 'obstructed': Terrain angle alpha(s) >= target_elevation_deg at some distance s.
    - 'clear': Ray reached max_distance_m entirely within valid DEM coverage and all alpha(s) < target_elevation_deg.
    - 'unknown': No obstruction found in evaluated terrain, but ray exited DEM boundary,
                 crossed NoData, or observer is outside DEM.
                 
    Args:
        dem: DemRaster container.
        target_azimuth_deg: Direction to target body (Sun, Earth, rover) clockwise from North [0°, 360°).
        target_elevation_deg: Angle of target body above observer horizontal tangent plane [-90°, +90°].
        observer_coord: Optional coordinate tuple.
        max_distance_m: Maximum search distance in meters.
        radial_step_m: Sampling resolution step in meters.
        observer_height_m: Instrument height above surface in meters.
        observer_lat: Observer selenographic latitude.
        observer_lon: Observer selenographic longitude.
        observer_x_m: Observer projected Easting.
        observer_y_m: Observer projected Northing.
        interpolation_method: 'bilinear' or 'nearest'.
        mode: 'auto', 'projected', or 'spherical'.
        
    Returns:
        Dictionary matching LineOfSightObstructionResponse schema.
    """
    ray_result = sample_terrain_ray(
        dem=dem,
        observer_coord=observer_coord,
        azimuth_deg=target_azimuth_deg,
        max_distance_m=max_distance_m,
        radial_step_m=radial_step_m,
        observer_height_m=observer_height_m,
        observer_lat=observer_lat,
        observer_lon=observer_lon,
        observer_x_m=observer_x_m,
        observer_y_m=observer_y_m,
        interpolation_method=interpolation_method,
        mode=mode
    )

    warnings = list(ray_result.get("warnings", []))
    samples = ray_result["samples"]

    # Handle observer outside DEM
    if ray_result["observer_elevation_m"] is None:
        return {
            "visibility_state": "unknown",
            "is_obstructed": None,
            "target_azimuth_deg": round(float(target_azimuth_deg) % 360.0, 2),
            "target_elevation_deg": round(float(target_elevation_deg), 2),
            "horizon_elevation_deg": None,
            "elevation_margin_deg": None,
            "obstructing_distance_m": None,
            "obstructing_point": None,
            "highest_point": None,
            "evaluated_distance_m": 0.0,
            "max_distance_m": round(max_distance_m, 1),
            "reason": "Observer location is outside DEM coverage or on NoData; line-of-sight cannot be evaluated.",
            "warnings": warnings,
            "elevation_datum": dem.metadata.reference_datum
        }

    first_obstruction_point = None
    first_obstruction_dist = None
    first_obstruction_angle = None

    for pt in samples:
        if pt["is_valid"] and pt["elevation_angle_deg"] is not None:
            if pt["elevation_angle_deg"] >= target_elevation_deg:
                first_obstruction_dist = pt["distance_m"]
                first_obstruction_angle = pt["elevation_angle_deg"]
                first_obstruction_point = {
                    "distance_m": pt["distance_m"],
                    "projected_x_m": pt["projected_x_m"],
                    "projected_y_m": pt["projected_y_m"],
                    "terrain_elevation_m": pt["terrain_elevation_m"],
                    "elevation_angle_deg": pt["elevation_angle_deg"]
                }
                break

    highest_angle = ray_result["max_horizon_angle_deg"]
    highest_pt = ray_result["max_horizon_point"]
    margin = round(target_elevation_deg - highest_angle, 4) if highest_angle is not None else None

    # Condition 1: Confirmed obstruction
    if first_obstruction_dist is not None:
        return {
            "visibility_state": "obstructed",
            "is_obstructed": True,
            "target_azimuth_deg": round(float(target_azimuth_deg) % 360.0, 2),
            "target_elevation_deg": round(float(target_elevation_deg), 2),
            "horizon_elevation_deg": highest_angle,
            "elevation_margin_deg": margin,
            "obstructing_distance_m": first_obstruction_dist,
            "obstructing_point": first_obstruction_point,
            "highest_point": highest_pt,
            "evaluated_distance_m": first_obstruction_dist,
            "max_distance_m": round(max_distance_m, 1),
            "reason": (
                f"Line of sight is obstructed by terrain at distance {first_obstruction_dist:.1f}m "
                f"(terrain angle: {first_obstruction_angle:+.2f}° >= target: {target_elevation_deg:+.2f}°)."
            ),
            "warnings": warnings,
            "elevation_datum": dem.metadata.reference_datum
        }

    # Condition 2: Truncated by DEM boundary or NoData without confirming obstruction
    if ray_result["has_boundary_exit"] or ray_result["has_nodata"]:
        eval_dist = ray_result["evaluated_distance_m"]
        return {
            "visibility_state": "unknown",
            "is_obstructed": None,
            "target_azimuth_deg": round(float(target_azimuth_deg) % 360.0, 2),
            "target_elevation_deg": round(float(target_elevation_deg), 2),
            "horizon_elevation_deg": highest_angle,
            "elevation_margin_deg": margin,
            "obstructing_distance_m": None,
            "obstructing_point": None,
            "highest_point": highest_pt,
            "evaluated_distance_m": eval_dist,
            "max_distance_m": round(max_distance_m, 1),
            "reason": (
                f"Ray traversed {eval_dist:.1f}m before encountering DEM boundary or NoData. "
                f"No obstruction was encountered in evaluated terrain (peak: {highest_angle:+.2f}°), "
                f"but missing or unmapped terrain beyond {eval_dist:.1f}m could obstruct line-of-sight."
            ),
            "warnings": warnings,
            "elevation_datum": dem.metadata.reference_datum
        }

    # Condition 3: Full ray traversed in valid DEM without obstruction
    return {
        "visibility_state": "clear",
        "is_obstructed": False,
        "target_azimuth_deg": round(float(target_azimuth_deg) % 360.0, 2),
        "target_elevation_deg": round(float(target_elevation_deg), 2),
        "horizon_elevation_deg": highest_angle,
        "elevation_margin_deg": margin,
        "obstructing_distance_m": None,
        "obstructing_point": None,
        "highest_point": highest_pt,
        "evaluated_distance_m": round(max_distance_m, 1),
        "max_distance_m": round(max_distance_m, 1),
        "reason": (
            f"Line of sight is confirmed clear out to {max_distance_m:.1f}m "
            f"(elevation margin: {margin:+.2f}° above highest terrain at {highest_angle:+.2f}°)."
        ),
        "warnings": warnings,
        "elevation_datum": dem.metadata.reference_datum
    }


# =============================================================================
# 6. 360-Degree Horizon Profile
# =============================================================================

def calculate_horizon_profile(
    dem: DemRaster,
    observer_coord: Optional[Tuple[float, float]] = None,
    max_distance_m: float = 30000.0,
    azimuth_step_deg: float = 1.0,
    radial_step_m: float = 100.0,
    observer_height_m: float = 2.0,
    observer_lat: Optional[float] = None,
    observer_lon: Optional[float] = None,
    observer_x_m: Optional[float] = None,
    observer_y_m: Optional[float] = None,
    interpolation_method: str = "bilinear",
    mode: str = "auto"
) -> Dict[str, Any]:
    """
    Calculate full 360-degree local terrain horizon elevation profile and summary metrics.
    
    Args:
        dem: Loaded DemRaster instance.
        observer_coord: Optional observer coordinate tuple.
        max_distance_m: Maximum search radius in meters.
        azimuth_step_deg: Angular sampling resolution step in degrees [0.1° to 45°].
        radial_step_m: Radial step along each ray in meters.
        observer_height_m: Height of instrument array above surface in meters.
        observer_lat: Selenographic latitude in degrees.
        observer_lon: Selenographic longitude in degrees.
        observer_x_m: Projected Easting in meters.
        observer_y_m: Projected Northing in meters.
        interpolation_method: 'bilinear' or 'nearest'.
        mode: 'auto', 'projected', or 'spherical'.
        
    Returns:
        Dictionary with horizon_points, min/max/mean horizon elevation, and azimuth extrema.
    """
    if azimuth_step_deg <= 0.0 or azimuth_step_deg > 45.0:
        raise ValueError(f"azimuth_step_deg must be in range (0.0, 45.0]: {azimuth_step_deg}")

    azimuths = np.arange(0.0, 360.0, azimuth_step_deg)
    horizon_points = []
    angles = []
    fully_valid_count = 0

    observer_info = None

    for az in azimuths:
        ray = sample_terrain_ray(
            dem=dem,
            observer_coord=observer_coord,
            azimuth_deg=float(az),
            max_distance_m=max_distance_m,
            radial_step_m=radial_step_m,
            observer_height_m=observer_height_m,
            observer_lat=observer_lat,
            observer_lon=observer_lon,
            observer_x_m=observer_x_m,
            observer_y_m=observer_y_m,
            interpolation_method=interpolation_method,
            mode=mode
        )
        if observer_info is None:
            observer_info = ray

        el = ray["max_horizon_angle_deg"]
        dist = ray["max_horizon_distance_m"]
        pt = ray["max_horizon_point"]
        has_exit = ray["has_boundary_exit"]
        has_nodata = ray["has_nodata"]

        if not (has_exit or has_nodata):
            fully_valid_count += 1

        if el is not None:
            angles.append(el)

        horizon_points.append({
            "azimuth_deg": round(float(az), 2),
            "horizon_elevation_deg": el,
            "obstruction_distance_m": dist,
            "highest_point": pt,
            "has_boundary_exit": has_exit,
            "has_nodata": has_nodata
        })

    total_rays = len(azimuths)
    valid_cov_pct = round(float(fully_valid_count / total_rays * 100.0), 2) if total_rays > 0 else 0.0

    if angles:
        arr_angles = np.array(angles, dtype=np.float64)
        min_el = round(float(np.min(arr_angles)), 4)
        max_el = round(float(np.max(arr_angles)), 4)
        mean_el = round(float(np.mean(arr_angles)), 4)
        
        idx_max = int(np.argmax(arr_angles))
        idx_min = int(np.argmin(arr_angles))
        az_max = horizon_points[idx_max]["azimuth_deg"]
        az_min = horizon_points[idx_min]["azimuth_deg"]
    else:
        min_el, max_el, mean_el = None, None, None
        az_max, az_min = None, None

    return {
        "observer_projected": observer_info["observer_projected"] if observer_info else None,
        "observer_elevation_m": observer_info["observer_elevation_m"] if observer_info else None,
        "observer_height_m": observer_height_m,
        "max_distance_m": round(max_distance_m, 1),
        "azimuth_step_deg": round(azimuth_step_deg, 2),
        "radial_step_m": round(radial_step_m, 1),
        "horizon_points": horizon_points,
        "min_horizon_elevation_deg": min_el,
        "max_horizon_elevation_deg": max_el,
        "mean_horizon_elevation_deg": mean_el,
        "azimuth_of_max_elevation_deg": az_max,
        "azimuth_of_min_elevation_deg": az_min,
        "valid_coverage_percent": valid_cov_pct,
        "elevation_datum": dem.metadata.reference_datum
    }


def get_horizon_elevation_at_azimuth(
    horizon_profile: Dict[str, Any],
    azimuth_deg: float
) -> float:
    """
    Continuous lookup of horizon obstruction elevation angle at an arbitrary azimuth
    with 360-degree periodic linear interpolation.
    
    Args:
        horizon_profile: Output dictionary from calculate_horizon_profile.
        azimuth_deg: Target azimuth angle in degrees.
        
    Returns:
        Interpolated horizon elevation angle in degrees.
    """
    pts = horizon_profile.get("horizon_points", [])
    if not pts:
        return 0.0

    az_norm = float(azimuth_deg) % 360.0
    azs = [p["azimuth_deg"] for p in pts if p.get("horizon_elevation_deg") is not None]
    els = [p["horizon_elevation_deg"] for p in pts if p.get("horizon_elevation_deg") is not None]

    if not azs:
        return 0.0

    azs_arr = np.array(azs, dtype=np.float64)
    els_arr = np.array(els, dtype=np.float64)

    # Periodic wraparound: append 360.0 with the value at the first azimuth (0.0)
    azs_ext = np.append(azs_arr, 360.0)
    els_ext = np.append(els_arr, els_arr[0])

    return float(np.interp(az_norm, azs_ext, els_ext))


# =============================================================================
# 8. Four-State Solar Disk Occultation & Scaled Irradiance
# =============================================================================

def calculate_solar_irradiance(
    sun_distance_km: Optional[float] = None,
    solar_constant_1au_w_m2: float = 1361.0
) -> Dict[str, Any]:
    """
    Calculate instantaneous top-of-atmosphere solar irradiance at the Moon scaled by
    inverse-square ephemeris distance:
      S(r) = S0 * (1.0 AU / r_sun_moon)^2
      
    Where:
      1.0 AU = 149,597,870.7 km (IAU standard astronomical unit).
      S0 = 1361.0 W/m² (WMO/NASA nominal total solar irradiance at 1 AU).
      
    If sun_distance_km is not provided (None), defaults to 1361.0 W/m² and explicitly
    labels the data provenance as 'ASSUMPTION (Fixed 1 AU Solar Constant)'.
    """
    AU_KM = 149597870.7

    if sun_distance_km is not None:
        dist_km = float(sun_distance_km)
        if dist_km <= 0.0:
            raise ValueError(f"sun_distance_km must be strictly positive: {sun_distance_km}")
        ratio = AU_KM / dist_km
        irradiance_w_m2 = float(solar_constant_1au_w_m2 * (ratio ** 2))
        provenance = "MEASURED (JPL Ephemeris Distance)"
        is_measured = True
    else:
        dist_km = AU_KM
        irradiance_w_m2 = float(solar_constant_1au_w_m2)
        provenance = "ASSUMPTION (Fixed 1 AU Solar Constant)"
        is_measured = False

    return {
        "solar_irradiance_w_m2": round(irradiance_w_m2, 2),
        "sun_distance_km": round(dist_km, 1),
        "distance_au": round(float(dist_km / AU_KM), 6),
        "is_measured": is_measured,
        "provenance": provenance
    }


def calculate_solar_disk_visibility(
    sun_elevation_deg: float,
    horizon_elevation_deg: Optional[float],
    solar_semi_diameter_deg: float = 0.2667,
    is_ray_valid: bool = True
) -> Dict[str, Any]:
    """
    Calculate finite-diameter solar disk occultation using circular segment geometry.
    Classifies illumination into four mutually exclusive states:
      - 'full': Sun disk completely above terrain horizon (visible_fraction = 1.0)
      - 'partial': Sun disk intersects terrain horizon crest (0.0 < visible_fraction < 1.0)
      - 'occulted': Sun disk completely below terrain horizon in umbra (visible_fraction = 0.0)
      - 'unknown': Inconclusive ray / NoData / boundary departure (visible_fraction = 0.0, uncertain)
      
    Circular Segment Geometry:
      Let clearance Delta = sun_elevation - horizon_elevation.
      Normalized chord offset u = clamp((horizon_elevation - sun_elevation) / solar_semi_diameter, -1.0, 1.0)
      Visible disk fraction:
        f_disk = (1 / pi) * (arccos(u) - u * sqrt(1 - u^2))
        
    Args:
        sun_elevation_deg: Solar elevation angle in degrees above local tangent plane.
        horizon_elevation_deg: Maximum terrain obstacle elevation angle in degrees (or None).
        solar_semi_diameter_deg: Solar angular semi-diameter in degrees (default: 0.2667° = 16 arcmin).
        is_ray_valid: False if ray exited DEM boundary or crossed NoData before confirming obstruction.
        
    Returns:
        Dictionary with visibility state, visible disk fraction, clearance angle, and flags.
    """
    if solar_semi_diameter_deg <= 0.0:
        raise ValueError(f"solar_semi_diameter_deg must be strictly positive: {solar_semi_diameter_deg}")

    delta_sun = float(solar_semi_diameter_deg)
    sun_el = float(sun_elevation_deg)

    # 1. Unknown / Inconclusive ray handling (Never interpret unknown as clear!)
    if horizon_elevation_deg is None or not is_ray_valid:
        return {
            "visibility_state": "unknown",
            "visible_fraction": 0.0,
            "is_illuminated": False,
            "sun_elevation_deg": round(sun_el, 3),
            "horizon_elevation_deg": round(float(horizon_elevation_deg), 3) if horizon_elevation_deg is not None else None,
            "clearance_deg": None,
            "solar_semi_diameter_deg": round(delta_sun, 4),
            "warnings": ["Inconclusive terrain coverage; ray exited DEM boundary or crossed NoData."]
        }

    horiz_el = float(horizon_elevation_deg)
    clearance = sun_el - horiz_el

    # 2. State classification
    if clearance >= delta_sun:
        # Full disk above terrain
        state = "full"
        f_disk = 1.0
        is_illum = True
    elif clearance <= -delta_sun:
        # Full disk below terrain (umbra)
        state = "occulted"
        f_disk = 0.0
        is_illum = False
    else:
        # Partial penumbra: limb occultation
        state = "partial"
        is_illum = True
        u = float(np.clip((horiz_el - sun_el) / delta_sun, -1.0, 1.0))
        # Circular segment area fraction: (1/pi) * (arccos(u) - u * sqrt(1 - u^2))
        area_frac = (math.acos(u) - u * math.sqrt(max(0.0, 1.0 - u * u))) / math.pi
        f_disk = float(np.clip(area_frac, 0.0, 1.0))

    return {
        "visibility_state": state,
        "visible_fraction": round(f_disk, 4),
        "is_illuminated": is_illum,
        "sun_elevation_deg": round(sun_el, 3),
        "horizon_elevation_deg": round(horiz_el, 3),
        "clearance_deg": round(clearance, 3),
        "solar_semi_diameter_deg": round(delta_sun, 4),
        "warnings": []
    }


def evaluate_solar_occultation_and_flux(
    dem: DemRaster,
    observer_coord: Optional[Tuple[float, float]] = None,
    sun_azimuth_deg: float = 0.0,
    sun_elevation_deg: float = 0.0,
    sun_distance_km: Optional[float] = None,
    max_distance_m: float = 60000.0,
    radial_step_m: float = 100.0,
    observer_height_m: float = 2.0,
    solar_semi_diameter_deg: float = 0.2667,
    mode: str = "auto"
) -> Dict[str, Any]:
    """
    Integrated evaluation of solar occultation and incident solar flux on lunar surface.
    Combines ray tracing, finite-disk penumbra circular segment geometry, and distance-scaled irradiance.
    """
    irradiance_data = calculate_solar_irradiance(sun_distance_km=sun_distance_km)
    s_flux = irradiance_data["solar_irradiance_w_m2"]

    ray = sample_terrain_ray(
        dem=dem,
        observer_coord=observer_coord,
        azimuth_deg=sun_azimuth_deg,
        max_distance_m=max_distance_m,
        radial_step_m=radial_step_m,
        observer_height_m=observer_height_m,
        mode=mode
    )

    peak_horizon = ray.get("max_horizon_angle_deg")
    is_valid_ray = not (ray.get("has_boundary_exit") or ray.get("has_nodata"))

    vis = calculate_solar_disk_visibility(
        sun_elevation_deg=sun_elevation_deg,
        horizon_elevation_deg=peak_horizon,
        solar_semi_diameter_deg=solar_semi_diameter_deg,
        is_ray_valid=is_valid_ray
    )

    f_disk = vis["visible_fraction"]
    sin_el = max(0.0, math.sin(math.radians(max(0.0, sun_elevation_deg))))
    direct_normal_flux = s_flux * f_disk
    horizontal_flux = direct_normal_flux * sin_el

    return {
        "visibility_state": vis["visibility_state"],
        "visible_fraction": vis["visible_fraction"],
        "is_illuminated": vis["is_illuminated"],
        "sun_azimuth_deg": round(float(sun_azimuth_deg) % 360.0, 2),
        "sun_elevation_deg": round(float(sun_elevation_deg), 3),
        "horizon_elevation_deg": vis["horizon_elevation_deg"],
        "clearance_deg": vis["clearance_deg"],
        "solar_irradiance_w_m2": s_flux,
        "direct_normal_flux_w_m2": round(direct_normal_flux, 2),
        "horizontal_flux_w_m2": round(horizontal_flux, 2),
        "is_distance_scaled": irradiance_data["is_measured"],
        "sun_distance_km": irradiance_data["sun_distance_km"],
        "provenance": {
            "irradiance": irradiance_data["provenance"],
            "terrain_horizon": "MEASURED (NASA LOLA DEM)",
            "solar_disk_geometry": "MEASURED (Circular Segment Penumbra)"
        },
        "warnings": list(ray.get("warnings", [])) + list(vis.get("warnings", []))
    }


# =============================================================================
# 9. Full First Fresnel-Zone & Topographic Obstruction Clearance
# =============================================================================

def calculate_fresnel_clearance(
    obstacle_distance_m: float,
    earth_distance_m: float = 384400000.0,
    frequency_hz: float = 2.2e9,
    earth_elevation_deg: float = 0.0,
    obstacle_elevation_angle_deg: float = 0.0,
    screening_fraction: float = 0.6
) -> Dict[str, Any]:
    """
    Calculate full first Fresnel-zone radius and physical terrain obstruction clearance:
      r_F1 = sqrt(lambda * d1 * d2 / (d1 + d2))
      
    Where:
      d1: Distance from lunar surface antenna to obstructing terrain ridge (meters).
      d2: Distance from terrain ridge to Earth receiving station (meters, d_Earth - d1).
      lambda: Carrier wavelength c / frequency_hz (meters).
      Delta_z_vert: Exact vertical clearance above obstacle knife-edge (meters):
                    Delta_z_vert = d1 * (tan(theta_Earth) - tan(theta_obs))
      Delta_h_perp: Perpendicular distance from obstacle knife-edge to direct line-of-sight ray:
                    Delta_h_perp = Delta_z_vert * cos(theta_Earth)
      z_req: Screening clearance threshold screening_fraction * r_F1 (default 0.6 * r_F1).
      
    Geometric Conventions:
      - theta_Earth and theta_obs are measured from the common local horizontal tangent plane
        at the lunar observer/transmitter.
      - At horizontal distance d1, the ray altitude is d1 * tan(theta_Earth) and the obstacle
        altitude is d1 * tan(theta_obs).
      - The exact vertical clearance is Delta_z = d1 * (tan(theta_Earth) - tan(theta_obs)).
      - The perpendicular distance from the obstacle to the line of sight is:
        Delta_h_perp = Delta_z * cos(theta_Earth).
      - Screening criterion: Evaluated against perpendicular clearance (ITU-R P.526 knife-edge convention),
        which agrees with the orientation of the circular first Fresnel zone transverse to the propagation path.
        Passes screening if Delta_h_perp >= z_req.
      - Total path distance: d_Earth = d1 + d2, valid because d1 <= 100 km << d_Earth ~ 384,400 km
        (deviation < 10^-7).
      
    NOTE: Evaluates lunar terrain Fresnel screening only.
    Earth-side DSN station elevation mask is a separate station-visibility criterion.
    """
    c_speed = 299792458.0  # Speed of light in vacuum (m/s)

    d1 = float(obstacle_distance_m)
    d_earth = float(earth_distance_m)
    f_hz = float(frequency_hz)
    frac = float(screening_fraction)

    if d1 <= 0.0:
        raise ValueError(f"obstacle_distance_m must be strictly positive: {obstacle_distance_m}")
    if d_earth <= d1:
        raise ValueError(f"earth_distance_m ({d_earth}) must be strictly greater than obstacle_distance_m ({d1}).")
    if f_hz <= 0.0:
        raise ValueError(f"frequency_hz must be strictly positive: {frequency_hz}")
    if frac <= 0.0:
        raise ValueError(f"screening_fraction must be strictly positive: {screening_fraction}")

    d2 = d_earth - d1
    wavelength_m = c_speed / f_hz

    # Full two-endpoint first Fresnel radius
    r_f1 = math.sqrt((wavelength_m * d1 * d2) / (d1 + d2))
    required_clearance_m = frac * r_f1

    # Exact vertical ray clearance above peak obstacle knife-edge:
    # At horizontal distance d1 from common origin:
    # z_ray = d1 * tan(theta_Earth), z_obs = d1 * tan(theta_obs)
    # Delta_z_vert = d1 * (tan(theta_Earth) - tan(theta_obs))
    rad_earth = math.radians(float(earth_elevation_deg))
    rad_obs = math.radians(float(obstacle_elevation_angle_deg))
    delta_z_m = d1 * (math.tan(rad_earth) - math.tan(rad_obs))

    # Perpendicular distance from obstacle knife-edge to direct line-of-sight ray:
    # Delta_h_perp = Delta_z_vert * cos(theta_Earth)
    h_perp_m = delta_z_m * math.cos(rad_earth)

    # Clearance margin and screening pass against perpendicular Fresnel radius
    clearance_margin_m = h_perp_m - required_clearance_m
    is_screening_pass = bool(h_perp_m >= required_clearance_m)

    return {
        "obstacle_distance_m": round(d1, 2),
        "earth_distance_m": round(d_earth, 1),
        "frequency_hz": f_hz,
        "wavelength_m": round(wavelength_m, 6),
        "fresnel_radius_r1_m": round(r_f1, 3),
        "screening_fraction": frac,
        "required_clearance_m": round(required_clearance_m, 3),
        "vertical_clearance_m": round(delta_z_m, 3),
        "perpendicular_clearance_m": round(h_perp_m, 3),
        "clearance_margin_m": round(clearance_margin_m, 3),
        "fresnel_screening_pass": is_screening_pass,
        "earth_elevation_deg": round(float(earth_elevation_deg), 3),
        "obstacle_elevation_angle_deg": round(float(obstacle_elevation_angle_deg), 3),
        "units": {
            "distances": "meters",
            "frequency": "Hertz",
            "wavelength": "meters",
            "vertical_clearance": "meters (vertical elevation difference at distance d1)",
            "perpendicular_clearance": "meters (perpendicular distance to direct line-of-sight ray)",
            "angles": "degrees above local horizontal tangent plane at lunar observer"
        },
        "provenance": {
            "fresnel_model": "MEASURED (Full two-endpoint Fresnel formulation)",
            "clearance_geometry": "MEASURED (Exact geometric elevation difference: d1 * (tan(El_Earth) - tan(El_obs)))",
            "perpendicular_projection": "MEASURED (Delta_z * cos(El_Earth))",
            "screening_threshold": "ASSUMPTION (0.6 * r_F1 standard screening rule)"
        }
    }
