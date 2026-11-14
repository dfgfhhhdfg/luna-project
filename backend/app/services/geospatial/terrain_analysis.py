"""
LUNA NAVIGATOR — Lunar Slope, Roughness & Terrain Analysis Engine (Member 3 Ownership)

Provides scientific computing modules for:
1. Elevation extraction & statistical aggregation.
2. Lunar topographic slope calculation via Horn's weighted 8-neighbor stencil
   with non-square pixel support (dx != dy) and robust nodata boundary handling.
3. Moving-window terrain roughness (local elevation standard deviation & relief).
4. Linear terrain profile transects with data gap detection.
5. Landing area terrain assessment for landing site recommendation (Member 4 integration).

Conventions:
- Horizontal Coordinate Reference: Lunar South Pole Polar Stereographic (meters)
- Elevation Datum: Mean Lunar Sphere R_moon = 1,737,400.0 meters (NASA LOLA standard)
- Elevation Units: Meters relative to the reference sphere
- Slope Units: Degrees [0°, 90°]
- Roughness Units: Meters
"""

import math
from typing import Tuple, Dict, Any, Optional, List, Sequence, Union
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

from app.services.geospatial.dem_loader import DemRaster


class TerrainAnalysisError(Exception):
    """Base exception for terrain analysis operations."""
    pass


class DemCoordinateError(TerrainAnalysisError):
    """Raised when DEM coordinate reference system or affine transform is invalid for slope analysis."""
    pass


# =============================================================================
# 1. Elevation Statistics
# =============================================================================

def calculate_elevation_statistics(
    elevation_array: np.ndarray,
    nodata_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute rigorous statistical metrics across an elevation raster or subset.
    
    Args:
        elevation_array: 1D or 2D array of elevation values in meters.
        nodata_value: Optional sentinel value representing nodata (e.g. -9999.0).
        
    Returns:
        Dictionary containing min, max, mean, median, standard deviation,
        valid data percentage, total cells, and valid cells.
    """
    arr = np.asarray(elevation_array, dtype=np.float64)
    total_cells = int(arr.size)
    
    if total_cells == 0:
        return {
            "min_elevation_m": None,
            "max_elevation_m": None,
            "mean_elevation_m": None,
            "median_elevation_m": None,
            "std_elevation_m": None,
            "valid_data_percent": 0.0,
            "total_cells": 0,
            "valid_cells": 0
        }

    # Filter out nodata and NaNs
    if nodata_value is not None:
        valid_mask = ~np.isnan(arr) & (arr != nodata_value)
    else:
        valid_mask = ~np.isnan(arr)

    valid_cells = int(np.count_nonzero(valid_mask))
    valid_pct = float(valid_cells / total_cells * 100.0)

    if valid_cells == 0:
        return {
            "min_elevation_m": None,
            "max_elevation_m": None,
            "mean_elevation_m": None,
            "median_elevation_m": None,
            "std_elevation_m": None,
            "valid_data_percent": 0.0,
            "total_cells": total_cells,
            "valid_cells": 0
        }

    valid_vals = arr[valid_mask]
    return {
        "min_elevation_m": round(float(np.min(valid_vals)), 2),
        "max_elevation_m": round(float(np.max(valid_vals)), 2),
        "mean_elevation_m": round(float(np.mean(valid_vals)), 2),
        "median_elevation_m": round(float(np.median(valid_vals)), 2),
        "std_elevation_m": round(float(np.std(valid_vals)), 2),
        "valid_data_percent": round(valid_pct, 2),
        "total_cells": total_cells,
        "valid_cells": valid_cells
    }


# =============================================================================
# 2. Lunar Terrain Slope (Horn's 8-Neighbor Stencil with dx != dy)
# =============================================================================

def calculate_slope(
    dem: Union[np.ndarray, DemRaster],
    nodata_value: Optional[float] = None,
    pixel_size_x_m: float = 100.0,
    pixel_size_y_m: float = 100.0
) -> np.ndarray:
    """
    Calculate terrain slope in degrees using Horn's weighted 8-neighbor central difference
    stencil, explicitly accounting for unequal pixel spacing (dx != dy) and nodata masking.
    
    Mathematical Formulation:
      dz/dx = ((z_ne + 2*z_e + z_se) - (z_nw + 2*z_w + z_sw)) / (8 * dx)
      dz/dy = ((z_nw + 2*z_n + z_ne) - (z_sw + 2*z_s + z_se)) / (8 * dy)
      gradient = sqrt((dz/dx)^2 + (dz/dy)^2)
      slope_deg = degrees(arctan(gradient))
      
    Edge and Nodata Fallback:
      For boundary pixels or cells adjacent to nodata, one-sided or 2-point central differences
      are applied to prevent false ~89° slope artifacts along nodata margins.
      Nodata and NaN cells are strictly preserved as NaN.
      
    Args:
        dem: DemRaster object or 2D numpy array of elevations in meters.
        nodata_value: Nodata sentinel (extracted from DemRaster if supplied).
        pixel_size_x_m: Easting pixel resolution in meters (extracted from DemRaster if supplied).
        pixel_size_y_m: Northing pixel resolution in meters (extracted from DemRaster if supplied).
        
    Returns:
        2D numpy array of topographic slopes in degrees [0.0, 90.0] with NaNs at nodata positions.
    """
    if isinstance(dem, DemRaster):
        elev = dem.elevation
        dx = float(dem.metadata.pixel_size_x_m)
        dy = float(dem.metadata.pixel_size_y_m)
        nodata = dem.metadata.nodata_value if nodata_value is None else nodata_value

        # Inspect and validate Coordinate Reference System (CRS)
        crs_str = str(dem.metadata.crs_repr).lower() if dem.metadata.crs_repr else ""
        geographic_indicators = ["degree", "degrees", "geogcs", "+proj=latlong", "+proj=longlat", "epsg:4326"]
        if any(ind in crs_str for ind in geographic_indicators):
            raise DemCoordinateError(
                f"DEM coordinate system is geographic (angular units): '{dem.metadata.crs_repr}'. "
                "Topographic slope calculation requires a conformal projected coordinate system in meters "
                "(such as Lunar South Pole Polar Stereographic)."
            )

        # Validate that projected coordinates use meters
        non_meter_indicators = ["foot", "feet", "ft", "+units=us-ft", "+units=ft"]
        if any(ind in crs_str for ind in non_meter_indicators):
            raise DemCoordinateError(
                f"Unsupported non-meter linear units in DEM CRS: '{dem.metadata.crs_repr}'. "
                "Slope calculation requires horizontal coordinates in meters."
            )

        # Inspect and validate affine transform for shear / rotation
        if dem.transform is not None:
            try:
                b = float(dem.transform[1])
                d = float(dem.transform[3])
                if abs(b) > 1e-6 or abs(d) > 1e-6:
                    raise DemCoordinateError(
                        f"Rotated or sheared raster grids (transform b={b:.6f}, d={d:.6f}) are not supported "
                        "for grid-aligned finite difference stencils. DEM must be aligned with projected coordinate axes."
                    )
            except DemCoordinateError:
                raise
            except Exception:
                pass
    else:
        elev = dem
        dx = float(pixel_size_x_m)
        dy = float(pixel_size_y_m)
        nodata = nodata_value

    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"Pixel resolutions must be strictly positive: dx={dx}, dy={dy}")

    if not isinstance(dem, DemRaster):
        if dx < 0.1 or dy < 0.1:
            raise DemCoordinateError(
                f"Pixel resolution (dx={dx}m, dy={dy}m) is less than 0.1m. "
                "Verify that horizontal grid coordinates are in meters and not in angular degrees."
            )

    z = np.copy(elev).astype(np.float64)
    if nodata is not None:
        z[z == nodata] = np.nan

    h, w = z.shape
    if h < 2 or w < 2:
        return np.full_like(z, np.nan)

    # Pad array with NaNs to compute neighbor stencils
    pad = np.pad(z, 1, mode="constant", constant_values=np.nan)
    nw = pad[:-2, :-2]
    n  = pad[:-2, 1:-1]
    ne = pad[:-2, 2:]
    w_ = pad[1:-1, :-2]
    c  = pad[1:-1, 1:-1]
    e_ = pad[1:-1, 2:]
    sw = pad[2:, :-2]
    s  = pad[2:, 1:-1]
    se = pad[2:, 2:]

    # Mask of cells where all 8 neighbors and center are valid
    valid_8 = (
        ~np.isnan(nw) & ~np.isnan(n) & ~np.isnan(ne) &
        ~np.isnan(w_) & ~np.isnan(c) & ~np.isnan(e_) &
        ~np.isnan(sw) & ~np.isnan(s) & ~np.isnan(se)
    )

    dz_dx = np.full_like(z, np.nan)
    dz_dy = np.full_like(z, np.nan)

    # Full 8-neighbor Horn stencil
    dz_dx[valid_8] = (
        (ne[valid_8] + 2.0 * e_[valid_8] + se[valid_8]) -
        (nw[valid_8] + 2.0 * w_[valid_8] + sw[valid_8])
    ) / (8.0 * dx)

    # Note: row index increases downwards (South), so Northern row (n) has larger Northing Y
    dz_dy[valid_8] = (
        (nw[valid_8] + 2.0 * n[valid_8] + ne[valid_8]) -
        (sw[valid_8] + 2.0 * s[valid_8] + se[valid_8])
    ) / (8.0 * dy)

    # Fallback for partial / boundary cells where center is valid but not all 8 neighbors are valid
    partial = ~valid_8 & ~np.isnan(c)

    # X gradient fallback (Easting)
    has_ew = partial & ~np.isnan(e_) & ~np.isnan(w_)
    dz_dx[has_ew] = (e_[has_ew] - w_[has_ew]) / (2.0 * dx)
    has_e_only = partial & ~has_ew & ~np.isnan(e_)
    dz_dx[has_e_only] = (e_[has_e_only] - c[has_e_only]) / dx
    has_w_only = partial & ~has_ew & ~np.isnan(w_)
    dz_dx[has_w_only] = (c[has_w_only] - w_[has_w_only]) / dx
    has_neither_x = partial & ~has_ew & ~has_e_only & ~has_w_only
    dz_dx[has_neither_x] = 0.0

    # Y gradient fallback (Northing)
    has_ns = partial & ~np.isnan(n) & ~np.isnan(s)
    dz_dy[has_ns] = (n[has_ns] - s[has_ns]) / (2.0 * dy)
    has_n_only = partial & ~has_ns & ~np.isnan(n)
    dz_dy[has_n_only] = (n[has_n_only] - c[has_n_only]) / dy
    has_s_only = partial & ~has_ns & ~np.isnan(s)
    dz_dy[has_s_only] = (c[has_s_only] - s[has_s_only]) / dy
    has_neither_y = partial & ~has_ns & ~has_n_only & ~has_s_only
    dz_dy[has_neither_y] = 0.0

    # Slope angle in degrees: arctan(gradient magnitude)
    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_deg = np.degrees(np.arctan(grad_mag))
    slope_deg[np.isnan(c)] = np.nan

    return slope_deg


def calculate_slope_statistics(
    slope_array: np.ndarray,
    thresholds: Sequence[float] = (5.0, 10.0, 12.0, 15.0)
) -> Dict[str, Any]:
    """
    Compute distribution metrics and threshold coverage for a slope raster.
    
    Args:
        slope_array: 1D or 2D array of slopes in degrees.
        thresholds: List/tuple of slope thresholds in degrees (e.g. 5°, 10°, 12°, 15°).
        
    Returns:
        Dictionary with min, max, mean, median, std, valid data percent,
        and threshold coverage percentages.
    """
    arr = np.asarray(slope_array, dtype=np.float64)
    total_cells = int(arr.size)
    valid_mask = ~np.isnan(arr)
    valid_cells = int(np.count_nonzero(valid_mask))

    if valid_cells == 0:
        return {
            "min_slope_deg": None,
            "max_slope_deg": None,
            "mean_slope_deg": None,
            "median_slope_deg": None,
            "std_slope_deg": None,
            "threshold_coverage_pct": {f"lte_{t:g}deg": 0.0 for t in thresholds},
            "valid_data_percent": 0.0
        }

    valid_vals = arr[valid_mask]
    cov_dict = {}
    for t in thresholds:
        count_lte = np.count_nonzero(valid_vals <= t)
        cov_dict[f"lte_{t:g}deg"] = round(float(count_lte / valid_cells * 100.0), 2)

    disjoint = calculate_disjoint_slope_categories(valid_vals)

    return {
        "min_slope_deg": round(float(np.min(valid_vals)), 2),
        "max_slope_deg": round(float(np.max(valid_vals)), 2),
        "mean_slope_deg": round(float(np.mean(valid_vals)), 2),
        "median_slope_deg": round(float(np.median(valid_vals)), 2),
        "std_slope_deg": round(float(np.std(valid_vals)), 2),
        "threshold_coverage_pct": cov_dict,
        "disjoint_coverage_pct": {
            "p_0_to_5_deg": disjoint["p_0_to_5_deg"],
            "p_5_to_10_deg": disjoint["p_5_to_10_deg"],
            "p_10_to_15_deg": disjoint["p_10_to_15_deg"],
            "p_gt_15_deg": disjoint["p_gt_15_deg"]
        },
        "slope_distribution_score": disjoint["slope_distribution_score"],
        "valid_data_percent": round(float(valid_cells / total_cells * 100.0), 2)
    }


def calculate_disjoint_slope_categories(
    slope_array: np.ndarray,
    nodata_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Partition valid slope values into four strictly disjoint, non-overlapping categories:
      p1: [0.0°, 5.0°]   (Gentle / Nominal, score weight 1.00)
      p2: (5.0°, 10.0°]  (Moderate / Acceptable, score weight 0.65)
      p3: (10.0°, 15.0°] (Steep / Marginal, score weight 0.20)
      p4: > 15.0°        (Hazardous / Severe, score weight 0.00)
      
    Valid-Cell Normalization:
      All percentages are calculated strictly relative to valid non-NaN / non-nodata cells.
      Identity: p1 + p2 + p3 + p4 == 100.0% (within floating point precision).
      
    Returns:
      Dictionary with disjoint percentages, raw cell counts, valid count,
      and normalized slope distribution score S_slope in [0.0, 100.0].
    """
    arr = np.asarray(slope_array, dtype=np.float64)
    if nodata_value is not None:
        valid_mask = ~np.isnan(arr) & (arr != nodata_value)
    else:
        valid_mask = ~np.isnan(arr)

    valid_cells = int(np.count_nonzero(valid_mask))
    if valid_cells == 0:
        return {
            "p_0_to_5_deg": 0.0,
            "p_5_to_10_deg": 0.0,
            "p_10_to_15_deg": 0.0,
            "p_gt_15_deg": 0.0,
            "cell_counts": {
                "c_0_to_5_deg": 0,
                "c_5_to_10_deg": 0,
                "c_10_to_15_deg": 0,
                "c_gt_15_deg": 0
            },
            "valid_cells": 0,
            "total_cells": int(arr.size),
            "slope_distribution_score": 0.0,
            "is_valid": False
        }

    valid_vals = arr[valid_mask]
    c1 = int(np.count_nonzero((valid_vals >= 0.0) & (valid_vals <= 5.0)))
    c2 = int(np.count_nonzero((valid_vals > 5.0) & (valid_vals <= 10.0)))
    c3 = int(np.count_nonzero((valid_vals > 10.0) & (valid_vals <= 15.0)))
    c4 = int(np.count_nonzero(valid_vals > 15.0))

    # Exact percentages relative to valid cells
    p1 = float(c1 / valid_cells * 100.0)
    p2 = float(c2 / valid_cells * 100.0)
    p3 = float(c3 / valid_cells * 100.0)
    p4 = float(c4 / valid_cells * 100.0)

    # Score: S_slope = 1.00*p1 + 0.65*p2 + 0.20*p3 + 0.00*p4 in [0, 100]
    score_slope = 1.00 * (c1 / valid_cells) + 0.65 * (c2 / valid_cells) + 0.20 * (c3 / valid_cells)
    score_slope_pct = round(float(score_slope * 100.0), 2)

    return {
        "p_0_to_5_deg": round(p1, 4),
        "p_5_to_10_deg": round(p2, 4),
        "p_10_to_15_deg": round(p3, 4),
        "p_gt_15_deg": round(p4, 4),
        "cell_counts": {
            "c_0_to_5_deg": c1,
            "c_5_to_10_deg": c2,
            "c_10_to_15_deg": c3,
            "c_gt_15_deg": c4
        },
        "valid_cells": valid_cells,
        "total_cells": int(arr.size),
        "slope_distribution_score": score_slope_pct,
        "is_valid": True
    }


# =============================================================================
# 3. Terrain Roughness (Moving-Window Standard Deviation & Local Relief)
# =============================================================================

def calculate_roughness(
    dem: Union[np.ndarray, DemRaster],
    window_size: int = 5,
    metric: str = "std",
    nodata_value: Optional[float] = None,
    min_valid_fraction: float = 0.5
) -> np.ndarray:
    """
    Calculate surface roughness across the DEM within a moving rectangular window.
    
    Supported Metrics:
      - 'std' (default): Local standard deviation of elevation (meters).
      - 'relief': Local relief, i.e., max(z) - min(z) within the window (meters).
      
    Nodata and Missing Cell Handling:
      - Nodata/NaN cells are ignored in window statistics.
      - If the count of valid cells in a window is less than min_valid_fraction * window_size^2,
        or if the center cell itself is nodata, the roughness is assigned NaN.
        
    Args:
        dem: DemRaster or 2D numpy array.
        window_size: Dimension of square kernel in pixels (must be >= 3, odd integer preferred).
        metric: 'std' or 'relief'.
        nodata_value: Nodata sentinel.
        min_valid_fraction: Minimum fraction of valid cells required in window [0.0, 1.0].
        
    Returns:
        2D numpy array of roughness values in meters with NaNs at nodata locations.
    """
    if isinstance(dem, DemRaster):
        elev = dem.elevation
        nodata = dem.metadata.nodata_value if nodata_value is None else nodata_value
    else:
        elev = dem
        nodata = nodata_value

    w = int(window_size)
    if w < 2:
        raise ValueError(f"Window size must be at least 2 pixels: {w}")

    z = np.copy(elev).astype(np.float64)
    if nodata is not None:
        z[z == nodata] = np.nan

    mask = (~np.isnan(z)).astype(np.float64)
    z_clean = np.where(np.isnan(z), 0.0, z)
    n_total = float(w * w)
    min_count = max(2.0, min_valid_fraction * n_total)

    if metric == "std":
        z_sq = z_clean ** 2

        # Sums within window via uniform_filter (constant 0-padding)
        sum_mask = uniform_filter(mask, size=w, mode="constant", cval=0.0) * n_total
        sum_z = uniform_filter(z_clean, size=w, mode="constant", cval=0.0) * n_total
        sum_z2 = uniform_filter(z_sq, size=w, mode="constant", cval=0.0) * n_total

        valid_cells = (sum_mask >= min_count) & (~np.isnan(z))
        roughness = np.full_like(z, np.nan)

        n_v = sum_mask[valid_cells]
        s1 = sum_z[valid_cells]
        s2 = sum_z2[valid_cells]

        var = (s2 - (s1 ** 2) / n_v) / np.maximum(1.0, n_v - 1.0)
        var = np.maximum(0.0, var)
        roughness[valid_cells] = np.sqrt(var)
        return roughness

    elif metric == "relief":
        # Local relief: max - min using local rank filters
        z_for_max = np.where(np.isnan(z), -1e9, z)
        z_for_min = np.where(np.isnan(z), 1e9, z)

        local_max = maximum_filter(z_for_max, size=w, mode="constant", cval=-1e9)
        local_min = minimum_filter(z_for_min, size=w, mode="constant", cval=1e9)
        sum_mask = uniform_filter(mask, size=w, mode="constant", cval=0.0) * n_total

        valid_cells = (sum_mask >= min_count) & (~np.isnan(z))
        relief = np.full_like(z, np.nan)
        relief[valid_cells] = local_max[valid_cells] - local_min[valid_cells]
        return relief

    else:
        raise ValueError(f"Unsupported roughness metric '{metric}'. Supported: 'std', 'relief'")


def calculate_roughness_statistics(
    roughness_array: np.ndarray,
    metric: str = "std_dev",
    window_size_pixels: int = 5,
    pixel_size_x_m: float = 100.0,
    pixel_size_y_m: float = 100.0
) -> Dict[str, Any]:
    """
    Compute summary statistics and physical scale metrics for a roughness raster.
    
    Args:
        roughness_array: 1D or 2D array of roughness values in meters.
        metric: 'std' or 'relief'.
        window_size_pixels: Dimension of square moving kernel in pixels.
        pixel_size_x_m: Horizontal pixel size in Easting (meters).
        pixel_size_y_m: Horizontal pixel size in Northing (meters).
    """
    arr = np.asarray(roughness_array, dtype=np.float64)
    total_cells = int(arr.size)
    valid_mask = ~np.isnan(arr)
    valid_cells = int(np.count_nonzero(valid_mask))

    baseline_x = round(float(window_size_pixels * pixel_size_x_m), 2)
    baseline_y = round(float(window_size_pixels * pixel_size_y_m), 2)
    baseline_m = round(float(math.sqrt(baseline_x * baseline_y)), 2)

    convention_str = (
        f"Anisotropic baselines: X={baseline_x}m ({window_size_pixels}px * {pixel_size_x_m:.1f}m), "
        f"Y={baseline_y}m ({window_size_pixels}px * {pixel_size_y_m:.1f}m)."
        if abs(baseline_x - baseline_y) > 1e-2 else
        f"Isotropic baseline = {baseline_x}m ({window_size_pixels}px * {pixel_size_x_m:.1f}m)."
    )

    if valid_cells == 0:
        return {
            "metric": metric,
            "window_size_pixels": window_size_pixels,
            "pixel_size_x_m": round(pixel_size_x_m, 2),
            "pixel_size_y_m": round(pixel_size_y_m, 2),
            "physical_baseline_x_m": baseline_x,
            "physical_baseline_y_m": baseline_y,
            "physical_baseline_m": baseline_m,
            "baseline_convention": convention_str,
            "min_roughness_m": None,
            "max_roughness_m": None,
            "mean_roughness_m": None,
            "median_roughness_m": None,
            "std_roughness_m": None,
            "valid_data_percent": 0.0
        }

    valid_vals = arr[valid_mask]
    return {
        "metric": metric,
        "window_size_pixels": window_size_pixels,
        "pixel_size_x_m": round(pixel_size_x_m, 2),
        "pixel_size_y_m": round(pixel_size_y_m, 2),
        "physical_baseline_x_m": baseline_x,
        "physical_baseline_y_m": baseline_y,
        "physical_baseline_m": baseline_m,
        "baseline_convention": convention_str,
        "min_roughness_m": round(float(np.min(valid_vals)), 2),
        "max_roughness_m": round(float(np.max(valid_vals)), 2),
        "mean_roughness_m": round(float(np.mean(valid_vals)), 2),
        "median_roughness_m": round(float(np.median(valid_vals)), 2),
        "std_roughness_m": round(float(np.std(valid_vals)), 2),
        "valid_data_percent": round(float(valid_cells / total_cells * 100.0), 2)
    }


# =============================================================================
# 4. Terrain Profile Analysis
# =============================================================================

def calculate_terrain_profile(
    dem_raster: DemRaster,
    start_coord: Tuple[float, float],
    end_coord: Tuple[float, float],
    sample_step_m: float = 100.0,
    max_gap_allowed_m: float = 500.0,
    interpolation_method: str = "bilinear"
) -> Dict[str, Any]:
    """
    Extract continuous elevation transect between two projected coordinates (Easting, Northing).
    Detects coverage boundaries and issues clear warnings for missing data gaps
    without silently interpolating across unmapped terrain.
    
    Args:
        dem_raster: DemRaster container.
        start_coord: (x_start, y_start) in projected meters.
        end_coord: (x_end, y_end) in projected meters.
        sample_step_m: Transect sampling step in meters (must be > 0).
        max_gap_allowed_m: Threshold distance of continuous missing cells that triggers a gap warning.
        interpolation_method: 'bilinear' (continuous smooth surface) or 'nearest' (grid cell center).
        
    Returns:
        Dictionary with profile_length_m, sample_step_m, points, min/max elevations,
        valid_coverage_percent, interpolation_method, and warnings.
    """
    if sample_step_m <= 0.0:
        raise ValueError(f"Sample step must be strictly positive: {sample_step_m}")

    x1, y1 = float(start_coord[0]), float(start_coord[1])
    x2, y2 = float(end_coord[0]), float(end_coord[1])

    profile_len = math.hypot(x2 - x1, y2 - y1)
    if profile_len < 1e-3:
        # Start and end coordinates are identical
        elev = dem_raster.get_elevation_at_projected(x1, y1, method=interpolation_method)
        is_val = not math.isnan(elev)
        return {
            "profile_length_m": 0.0,
            "sample_step_m": sample_step_m,
            "points": [{
                "distance_m": 0.0,
                "projected_x_m": x1,
                "projected_y_m": y1,
                "elevation_m": round(elev, 2) if is_val else None
            }],
            "min_elevation_m": round(elev, 2) if is_val else None,
            "max_elevation_m": round(elev, 2) if is_val else None,
            "valid_coverage_percent": 100.0 if is_val else 0.0,
            "interpolation_method": interpolation_method,
            "warnings": [] if is_val else ["Point falls on nodata or outside DEM bounds."],
            "elevation_datum": dem_raster.metadata.reference_datum
        }

    num_samples = max(2, int(math.ceil(profile_len / sample_step_m)) + 1)
    distances = np.linspace(0.0, profile_len, num_samples)
    t = distances / profile_len
    xs = x1 + t * (x2 - x1)
    ys = y1 + t * (y2 - y1)

    points = []
    valid_elevations = []
    warnings = []

    current_gap_start = None
    current_gap_len = 0.0

    for i in range(num_samples):
        dist = float(distances[i])
        px_x = float(xs[i])
        px_y = float(ys[i])
        elev = dem_raster.get_elevation_at_projected(px_x, px_y, method=interpolation_method)

        if math.isnan(elev):
            points.append({
                "distance_m": round(dist, 1),
                "projected_x_m": round(px_x, 1),
                "projected_y_m": round(px_y, 1),
                "elevation_m": None
            })
            if current_gap_start is None:
                current_gap_start = distances[max(0, i - 1)]
            step_len = distances[i] - distances[max(0, i - 1)] if i > 0 else 0.0
            current_gap_len += step_len
        else:
            elev_rounded = round(elev, 2)
            valid_elevations.append(elev_rounded)
            points.append({
                "distance_m": round(dist, 1),
                "projected_x_m": round(px_x, 1),
                "projected_y_m": round(px_y, 1),
                "elevation_m": elev_rounded
            })
            # Check if an open gap exceeded threshold
            if current_gap_len >= max_gap_allowed_m:
                warnings.append(
                    f"Data gap of {current_gap_len:.1f}m detected between distance "
                    f"{current_gap_start:.1f}m and {dist:.1f}m (no silent interpolation)."
                )
            current_gap_start = None
            current_gap_len = 0.0

    # Check trailing gap at profile end
    if current_gap_len >= max_gap_allowed_m:
        warnings.append(
            f"Trailing data gap of {current_gap_len:.1f}m detected between distance "
            f"{current_gap_start:.1f}m and {profile_len:.1f}m."
        )

    valid_count = len(valid_elevations)
    valid_pct = round(float(valid_count / num_samples * 100.0), 2)

    if valid_pct < 100.0:
        warnings.insert(0, f"Profile crosses {num_samples - valid_count} nodata/out-of-bounds cells ({valid_pct}% valid coverage).")
    if valid_pct < 50.0:
        warnings.insert(0, "WARNING: Insufficient DEM coverage along transect (<50% valid cells).")

    min_el = min(valid_elevations) if valid_count > 0 else None
    max_el = max(valid_elevations) if valid_count > 0 else None

    return {
        "profile_length_m": round(profile_len, 2),
        "sample_step_m": round(sample_step_m, 2),
        "points": points,
        "min_elevation_m": min_el,
        "max_elevation_m": max_el,
        "valid_coverage_percent": valid_pct,
        "interpolation_method": interpolation_method,
        "warnings": warnings,
        "elevation_datum": dem_raster.metadata.reference_datum
    }


# =============================================================================
# 5. Candidate Landing Area Terrain Assessment (Member 4 Integration)
# =============================================================================

def assess_landing_area(
    dem_raster: DemRaster,
    center_projected: Tuple[float, float],
    radius_m: float = 1000.0,
    slope_thresholds: Sequence[float] = (5.0, 10.0, 12.0, 15.0),
    roughness_window_size: int = 5
) -> Dict[str, Any]:
    """
    Perform a complete topographic assessment of a candidate landing site ellipse or circular zone.
    Computes slope distribution, moving-window roughness, slope-threshold safety coverage,
    and valid-data compliance for consumption by Member 4's multi-criteria recommendation engine.
    
    Coverage Normalization:
      Coverage is strictly normalized against the full requested circular footprint area (pi * R^2).
      Any portion of the circle extending outside DEM raster bounds is accounted for as unmapped,
      guaranteeing valid_data_percent reflects actual operational coverage and never exceeds 100%.
      
    Args:
        dem_raster: Full or regional DemRaster dataset.
        center_projected: (center_x, center_y) in projected meters.
        radius_m: Candidate landing site radius in meters (e.g. 1000m).
        slope_thresholds: Sequence of critical lander slope angles in degrees.
        roughness_window_size: Moving kernel size for roughness analysis in pixels.
        
    Returns:
        Structured dictionary conforming to LandingAreaAssessmentResponse.
    """
    if radius_m <= 0.0:
        raise ValueError(f"Assessment radius must be strictly positive: {radius_m}")

    cx, cy = float(center_projected[0]), float(center_projected[1])
    px_x = float(dem_raster.metadata.pixel_size_x_m)
    px_y = float(dem_raster.metadata.pixel_size_y_m)

    # Expected cell count for nominal Euclidean circle of radius R
    nominal_cells = max(1, int(round((math.pi * radius_m ** 2) / (px_x * px_y))))

    # Buffer the window to accommodate kernel boundaries for slope (1 cell) and roughness (w//2 cells)
    kernel_pad_m = (roughness_window_size // 2 + 2) * max(px_x, px_y)
    total_buffer_m = radius_m + kernel_pad_m

    # Bounds of sub-window in projected coordinates
    sub_left = cx - total_buffer_m
    sub_bottom = cy - total_buffer_m
    sub_right = cx + total_buffer_m
    sub_top = cy + total_buffer_m

    # Intersect with raster bounds
    r_left, r_bottom, r_right, r_top = dem_raster.metadata.bounds_projected
    warnings = []

    if sub_right < r_left or sub_left > r_right or sub_top < r_bottom or sub_bottom > r_top:
        warnings.append(f"Landing footprint (radius {radius_m:.1f}m) is completely outside the loaded DEM bounds.")
        empty_elev = calculate_elevation_statistics(np.array([]))
        empty_slope = calculate_slope_statistics(np.array([]), thresholds=slope_thresholds)
        empty_roughness = calculate_roughness_statistics(
            np.array([]),
            window_size_pixels=roughness_window_size,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y
        )
        return {
            "center_projected": (cx, cy),
            "radius_m": radius_m,
            "elevation_stats": empty_elev,
            "slope_stats": empty_slope,
            "roughness_stats": empty_roughness,
            "valid_data_percent": 0.0,
            "warnings": warnings,
            "disclaimer": "Topographic assessment reflects slope and elevation roughness metrics only; does not evaluate sub-pixel boulder hazards (<20m) or regolith bearing strength."
        }

    # Slice sub-array from DemRaster
    c_start = max(0, int((sub_left - r_left) / px_x))
    c_end = min(dem_raster.metadata.width_pixels, int(math.ceil((sub_right - r_left) / px_x)))
    r_start = max(0, int((r_top - sub_top) / px_y))
    r_end = min(dem_raster.metadata.height_pixels, int(math.ceil((r_top - sub_bottom) / px_y)))

    sub_elev = dem_raster.elevation[r_start:r_end, c_start:c_end]
    if sub_elev.size == 0:
        warnings.append("No DEM cells available within landing area window.")
        empty_elev = calculate_elevation_statistics(np.array([]))
        empty_slope = calculate_slope_statistics(np.array([]), thresholds=slope_thresholds)
        empty_roughness = calculate_roughness_statistics(
            np.array([]),
            window_size_pixels=roughness_window_size,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y
        )
        return {
            "center_projected": (cx, cy),
            "radius_m": radius_m,
            "elevation_stats": empty_elev,
            "slope_stats": empty_slope,
            "roughness_stats": empty_roughness,
            "valid_data_percent": 0.0,
            "warnings": warnings,
            "disclaimer": "Topographic assessment reflects slope and elevation roughness metrics only; does not evaluate sub-pixel boulder hazards (<20m) or regolith bearing strength."
        }

    # Calculate slope and roughness on the buffered sub-window
    slope_grid = calculate_slope(
        sub_elev,
        nodata_value=dem_raster.metadata.nodata_value,
        pixel_size_x_m=px_x,
        pixel_size_y_m=px_y
    )
    roughness_grid = calculate_roughness(
        sub_elev,
        window_size=roughness_window_size,
        metric="std",
        nodata_value=dem_raster.metadata.nodata_value
    )

    # Build coordinate grid for pixels in sub-window to construct the circular mask
    sub_h, sub_w = sub_elev.shape
    cols = np.arange(c_start, c_end)
    rows = np.arange(r_start, r_end)
    grid_cols, grid_rows = np.meshgrid(cols, rows)

    # Pixel center projected coordinates
    grid_x = r_left + (grid_cols + 0.5) * px_x
    grid_y = r_top - (grid_rows + 0.5) * px_y

    distances = np.hypot(grid_x - cx, grid_y - cy)
    circle_mask = distances <= radius_m

    n_in_bounds = int(np.count_nonzero(circle_mask))
    if n_in_bounds == 0:
        circle_mask[sub_h // 2, sub_w // 2] = True
        n_in_bounds = 1

    elev_in_circle = sub_elev[circle_mask]
    slope_in_circle = slope_grid[circle_mask]
    roughness_in_circle = roughness_grid[circle_mask]

    # Accounting for cells outside the DEM raster boundary
    n_outside_dem = max(0, nominal_cells - n_in_bounds)
    total_footprint_cells = max(nominal_cells, n_in_bounds)

    nodata_val = dem_raster.metadata.nodata_value
    if nodata_val is not None:
        valid_in_circle_mask = ~np.isnan(elev_in_circle) & (elev_in_circle != nodata_val)
    else:
        valid_in_circle_mask = ~np.isnan(elev_in_circle)

    valid_cells_count = int(np.count_nonzero(valid_in_circle_mask))
    nodata_in_bounds = n_in_bounds - valid_cells_count

    # Calculate rigorous coverage: guaranteed <= 100.0%
    overall_valid_pct = min(100.0, round(float(valid_cells_count / total_footprint_cells * 100.0), 2))

    # Calculate statistics
    elev_stats = calculate_elevation_statistics(elev_in_circle, nodata_value=dem_raster.metadata.nodata_value)
    elev_stats["valid_data_percent"] = overall_valid_pct
    elev_stats["total_cells"] = total_footprint_cells
    elev_stats["valid_cells"] = valid_cells_count

    slope_stats = calculate_slope_statistics(slope_in_circle, thresholds=slope_thresholds)
    slope_stats["valid_data_percent"] = overall_valid_pct

    rough_stats = calculate_roughness_statistics(
        roughness_in_circle,
        metric="std_dev",
        window_size_pixels=roughness_window_size,
        pixel_size_x_m=px_x,
        pixel_size_y_m=px_y
    )
    rough_stats["valid_data_percent"] = overall_valid_pct

    if n_outside_dem > 0:
        outside_pct = round(float(n_outside_dem / total_footprint_cells * 100.0), 1)
        warnings.append(
            f"Landing footprint extends outside DEM boundary: {n_outside_dem} cells ({outside_pct}%) lie outside raster coverage."
        )
    if nodata_in_bounds > 0:
        warnings.append(f"Landing footprint contains {nodata_in_bounds} unmapped nodata cells inside coverage.")
    if overall_valid_pct < 100.0:
        warnings.insert(0, f"Landing footprint contains {100.0 - overall_valid_pct:.1f}% unmapped or invalid cells ({overall_valid_pct}% valid coverage).")
    if overall_valid_pct < 80.0:
        warnings.insert(0, "WARNING: Insufficient DEM coverage (<80% valid footprint). Landing site assessment uncertain.")

    return {
        "center_projected": (round(cx, 1), round(cy, 1)),
        "radius_m": round(radius_m, 1),
        "elevation_stats": elev_stats,
        "slope_stats": slope_stats,
        "roughness_stats": rough_stats,
        "valid_data_percent": overall_valid_pct,
        "warnings": warnings,
        "disclaimer": "Topographic assessment reflects slope and elevation roughness metrics only; does not evaluate sub-pixel boulder hazards (<20m) or regolith bearing strength."
    }


# =============================================================================
# 6. Physical Cliff-Buffer Distance Extractor
# =============================================================================

def calculate_cliff_buffer_distance(
    dem: Union[np.ndarray, DemRaster],
    center_projected: Tuple[float, float],
    cliff_slope_threshold_deg: float = 20.0,
    max_search_radius_m: float = 2000.0,
    pixel_size_x_m: Optional[float] = None,
    pixel_size_y_m: Optional[float] = None,
    nodata_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate Euclidean distance from a candidate landing center to the nearest terrain drop-off
    or cliff edge where local slope >= cliff_slope_threshold_deg (default 20.0°).
    
    Uses raster pixel coordinates and affine dimensions (dx, dy).
    Returns an explicit unavailable/none result if no qualifying cliff exists within max_search_radius_m.
    
    Args:
        dem: DemRaster or 2D elevation numpy array.
        center_projected: (cx, cy) center coordinate in projected meters.
        cliff_slope_threshold_deg: Minimum slope angle defining a critical cliff/drop-off (default 20.0°).
        max_search_radius_m: Maximum Euclidean search radius in meters (default 2000.0m).
        pixel_size_x_m: Pixel dimension in X (Easting) in meters (required if dem is np.ndarray).
        pixel_size_y_m: Pixel dimension in Y (Northing) in meters (required if dem is np.ndarray).
        nodata_value: Nodata sentinel value (optional).
        
    Returns:
        Dictionary containing:
        - has_cliff_in_radius (bool)
        - cliff_distance_m (Optional[float], rounded to 2 decimal places, or None)
        - nearest_cliff_coord (Optional[Tuple[float, float]], (x, y) in meters, or None)
        - nearest_cliff_slope_deg (Optional[float], or None)
        - cliff_slope_threshold_deg (float)
        - search_radius_m (float)
        - status (str: 'CLIFF_DETECTED', 'NO_CLIFF_WITHIN_SEARCH_RADIUS', 'CENTER_OUTSIDE_DEM')
    """
    if max_search_radius_m <= 0.0:
        raise ValueError(f"max_search_radius_m must be strictly positive: {max_search_radius_m}")
    if cliff_slope_threshold_deg <= 0.0:
        raise ValueError(f"cliff_slope_threshold_deg must be strictly positive: {cliff_slope_threshold_deg}")

    cx, cy = float(center_projected[0]), float(center_projected[1])

    if isinstance(dem, DemRaster):
        dem_raster = dem
        px_x = float(dem_raster.metadata.pixel_size_x_m)
        px_y = float(dem_raster.metadata.pixel_size_y_m)
        nodata_val = dem_raster.metadata.nodata_value if nodata_value is None else nodata_value
        r_left, r_bottom, r_right, r_top = dem_raster.metadata.bounds_projected

        # Check if center is outside DEM bounds
        if not (r_left <= cx <= r_right and r_bottom <= cy <= r_top):
            return {
                "has_cliff_in_radius": False,
                "cliff_distance_m": None,
                "nearest_cliff_coord": None,
                "nearest_cliff_slope_deg": None,
                "cliff_slope_threshold_deg": cliff_slope_threshold_deg,
                "search_radius_m": max_search_radius_m,
                "status": "CENTER_OUTSIDE_DEM"
            }

        # Sub-window bounds with padding for slope stencil (1 cell)
        pad_m = max(px_x, px_y) * 2.0
        sub_left = max(r_left, cx - max_search_radius_m - pad_m)
        sub_bottom = max(r_bottom, cy - max_search_radius_m - pad_m)
        sub_right = min(r_right, cx + max_search_radius_m + pad_m)
        sub_top = min(r_top, cy + max_search_radius_m + pad_m)

        c_start = max(0, int((sub_left - r_left) / px_x))
        c_end = min(dem_raster.metadata.width_pixels, int(math.ceil((sub_right - r_left) / px_x)))
        r_start = max(0, int((r_top - sub_top) / px_y))
        r_end = min(dem_raster.metadata.height_pixels, int(math.ceil((r_top - sub_bottom) / px_y)))

        sub_elev = dem_raster.elevation[r_start:r_end, c_start:c_end]
        if sub_elev.size == 0:
            return {
                "has_cliff_in_radius": False,
                "cliff_distance_m": None,
                "nearest_cliff_coord": None,
                "nearest_cliff_slope_deg": None,
                "cliff_slope_threshold_deg": cliff_slope_threshold_deg,
                "search_radius_m": max_search_radius_m,
                "status": "NO_CLIFF_WITHIN_SEARCH_RADIUS"
            }

        slope_grid = calculate_slope(sub_elev, nodata_value=nodata_val, pixel_size_x_m=px_x, pixel_size_y_m=px_y)

        cols = np.arange(c_start, c_end)
        rows = np.arange(r_start, r_end)
        grid_cols, grid_rows = np.meshgrid(cols, rows)
        grid_x = r_left + (grid_cols + 0.5) * px_x
        grid_y = r_top - (grid_rows + 0.5) * px_y

    else:
        elev_arr = np.asarray(dem, dtype=np.float64)
        px_x = float(pixel_size_x_m if pixel_size_x_m is not None else 100.0)
        px_y = float(pixel_size_y_m if pixel_size_y_m is not None else 100.0)
        nodata_val = nodata_value

        h, w = elev_arr.shape
        slope_grid = calculate_slope(elev_arr, nodata_value=nodata_val, pixel_size_x_m=px_x, pixel_size_y_m=px_y)

        cols = np.arange(w)
        rows = np.arange(h)
        grid_cols, grid_rows = np.meshgrid(cols, rows)
        grid_x = (grid_cols + 0.5 - w / 2.0) * px_x
        grid_y = (h / 2.0 - (grid_rows + 0.5)) * px_y

    distances = np.hypot(grid_x - cx, grid_y - cy)
    cliff_mask = (distances <= max_search_radius_m) & (slope_grid >= cliff_slope_threshold_deg) & (~np.isnan(slope_grid))

    if not np.any(cliff_mask):
        return {
            "has_cliff_in_radius": False,
            "cliff_distance_m": None,
            "nearest_cliff_coord": None,
            "nearest_cliff_slope_deg": None,
            "cliff_slope_threshold_deg": cliff_slope_threshold_deg,
            "search_radius_m": max_search_radius_m,
            "status": "NO_CLIFF_WITHIN_SEARCH_RADIUS"
        }

    valid_dists = distances[cliff_mask]
    valid_xs = grid_x[cliff_mask]
    valid_ys = grid_y[cliff_mask]
    valid_slopes = slope_grid[cliff_mask]

    min_idx = np.argmin(valid_dists)
    min_dist = float(valid_dists[min_idx])
    near_x = float(valid_xs[min_idx])
    near_y = float(valid_ys[min_idx])
    near_slope = float(valid_slopes[min_idx])

    return {
        "has_cliff_in_radius": True,
        "cliff_distance_m": round(min_dist, 2),
        "nearest_cliff_coord": (round(near_x, 2), round(near_y, 2)),
        "nearest_cliff_slope_deg": round(near_slope, 2),
        "cliff_slope_threshold_deg": cliff_slope_threshold_deg,
        "search_radius_m": max_search_radius_m,
        "status": "CLIFF_DETECTED"
    }


# =============================================================================
# 7. GIS-to-MCDA Metrics Adapter
# =============================================================================

def evaluate_site_terrain_safety(
    dem: DemRaster,
    center_coord: Optional[Tuple[float, float]] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    center_x_m: Optional[float] = None,
    center_y_m: Optional[float] = None,
    landing_radius_m: float = 200.0,
    max_acceptable_slope_deg: float = 15.0,
    min_dem_coverage_percent: float = 75.0,
    roughness_window_size: int = 5,
    cliff_search_radius_m: float = 2000.0
) -> Dict[str, Any]:
    """
    GIS-to-MCDA Metrics Adapter: Evaluates landing footprint topographic safety and generates
    pure measured GIS metrics and compliance inputs for Member 4's MCDA recommendation engine.
    
    Safety Compliance Gates Evaluated:
      1. DEM Coverage Compliance: C_DEM >= min_dem_coverage_percent (default 75.0%).
      2. Landing Slope Compliance: max_slope <= max_acceptable_slope_deg (default 15.0°)
         AND severe slope hazard footprint p(>15°) <= 1.0%.
         
    Returns:
      Structured dictionary with measured GIS metrics, disjoint slope categories,
      cliff buffer distance, compliance flags, hard constraint violations,
      coordinate metadata, units, and data provenance.
      
    NOTE: Does NOT perform weighted MCDA scoring or multi-site ranking.
    Delivers objective measured GIS metrics directly to Member 4.
    """
    from app.services.geospatial.projection import lola_polar_projection

    # 1. Resolve coordinates
    lat_c, lon_c = None, None
    if center_lat is not None and center_lon is not None:
        lat_c, lon_c = float(center_lat), float(center_lon)
        cx, cy = lola_polar_projection.forward(lat_c, lon_c)
    elif center_x_m is not None and center_y_m is not None:
        cx, cy = float(center_x_m), float(center_y_m)
        try:
            lat_c, lon_c = lola_polar_projection.inverse(cx, cy)
        except Exception:
            pass
    elif center_coord is not None:
        c1, c2 = float(center_coord[0]), float(center_coord[1])
        b_left, b_bottom, b_right, b_top = dem.metadata.bounds_projected
        if -90.0 <= c1 <= 0.0 and 0.0 <= c2 <= 360.0 and (c1 < b_left or c1 > b_right):
            lat_c, lon_c = c1, c2
            cx, cy = lola_polar_projection.forward(lat_c, lon_c)
        else:
            cx, cy = c1, c2
            try:
                lat_c, lon_c = lola_polar_projection.inverse(cx, cy)
            except Exception:
                pass
    else:
        raise ValueError("Candidate landing center coordinate must be provided.")

    # 2. Assess landing area footprint
    assessment = assess_landing_area(
        dem_raster=dem,
        center_projected=(cx, cy),
        radius_m=landing_radius_m,
        roughness_window_size=roughness_window_size
    )

    cov_pct = assessment["valid_data_percent"]
    slope_stats = assessment["slope_stats"]
    disjoint = slope_stats.get("disjoint_categories")
    rough_stats = assessment["roughness_stats"]
    elev_stats = assessment["elevation_stats"]

    # 3. Assess cliff buffer distance
    cliff_res = calculate_cliff_buffer_distance(
        dem=dem,
        center_projected=(cx, cy),
        cliff_slope_threshold_deg=20.0,
        max_search_radius_m=cliff_search_radius_m
    )

    # 4. Check compliance gates
    violations: List[str] = []
    is_cov_compliant = bool(cov_pct >= min_dem_coverage_percent)
    if not is_cov_compliant:
        violations.append(
            f"DEM valid coverage {cov_pct:.1f}% is below minimum required {min_dem_coverage_percent:.1f}%."
        )

    max_slope = slope_stats.get("max_slope_deg")
    mean_slope = slope_stats.get("mean_slope_deg")
    p_gt_15 = disjoint["p_gt_15_deg"] if disjoint else 0.0

    if max_slope is None:
        is_slope_compliant = False
        violations.append("No valid terrain slope data available within landing ellipse.")
    else:
        slope_exceeded = max_slope > max_acceptable_slope_deg
        severe_footprint_exceeded = p_gt_15 > 1.0
        is_slope_compliant = not (slope_exceeded or severe_footprint_exceeded)
        if slope_exceeded:
            violations.append(
                f"Maximum terrain slope {max_slope:.1f}° exceeds allowable screening limit {max_acceptable_slope_deg:.1f}°."
            )
        if severe_footprint_exceeded:
            violations.append(
                f"Severe slope hazard footprint {p_gt_15:.2f}% exceeds 1.0% limit."
            )

    is_safety_gated = bool(is_cov_compliant and is_slope_compliant)

    # 5. Composite Terrain Safety Score (0.70*Slope + 0.30*Roughness)
    score_slope = disjoint["slope_distribution_score"] if disjoint else 0.0
    rough_m = rough_stats.get("mean_roughness_m")
    if rough_m is not None:
        score_roughness = max(0.0, min(100.0, 100.0 - (rough_m / 2.5) * 100.0))
        terrain_safety_score = round(0.70 * score_slope + 0.30 * score_roughness, 1)
    else:
        score_roughness = None
        terrain_safety_score = None

    # Buffer score for Operational Accessibility
    d_cliff = cliff_res.get("cliff_distance_m")
    if d_cliff is not None:
        buffer_score = min(100.0, round(float(d_cliff / 500.0 * 100.0), 1))
    else:
        buffer_score = 100.0 if cliff_res["status"] == "NO_CLIFF_WITHIN_SEARCH_RADIUS" else 0.0

    warnings = list(assessment.get("warnings", []))

    return {
        "center_projected_m": (round(cx, 1), round(cy, 1)),
        "center_geographic_deg": (round(lat_c, 4), round(lon_c, 4)) if (lat_c is not None and lon_c is not None) else None,
        "landing_radius_m": round(landing_radius_m, 1),
        "dem_coverage_percent": cov_pct,
        "is_coverage_compliant": is_cov_compliant,
        "elevation_m": {
            "mean": elev_stats.get("mean_elevation_m"),
            "min": elev_stats.get("min_elevation_m"),
            "max": elev_stats.get("max_elevation_m")
        },
        "disjoint_slope_pcts": {
            "p_0_to_5_deg": disjoint["p_0_to_5_deg"] if disjoint else 0.0,
            "p_5_to_10_deg": disjoint["p_5_to_10_deg"] if disjoint else 0.0,
            "p_10_to_15_deg": disjoint["p_10_to_15_deg"] if disjoint else 0.0,
            "p_gt_15_deg": disjoint["p_gt_15_deg"] if disjoint else 0.0
        },
        "slope_distribution_score": score_slope,
        "max_slope_deg": max_slope,
        "mean_slope_deg": mean_slope,
        "is_slope_compliant": is_slope_compliant,
        "roughness_std_m": rough_m,
        "roughness_score": round(score_roughness, 1) if score_roughness is not None else None,
        "terrain_safety_score": terrain_safety_score,
        "cliff_buffer": {
            "has_cliff_in_radius": cliff_res["has_cliff_in_radius"],
            "cliff_distance_m": cliff_res["cliff_distance_m"],
            "nearest_cliff_slope_deg": cliff_res["nearest_cliff_slope_deg"],
            "buffer_score": buffer_score,
            "status": cliff_res["status"]
        },
        "is_safety_gated": is_safety_gated,
        "violations": violations,
        "warnings": warnings,
        "units": {
            "elevation": "meters above R=1737.4 km lunar sphere",
            "slope": "degrees",
            "roughness": "meters (standard deviation over physical baseline)",
            "distances": "meters",
            "coordinates": "meters (Lunar South Pole Polar Stereographic)"
        },
        "provenance": {
            "terrain_elevation": "MEASURED (NASA LOLA DEM)",
            "slope_and_roughness": "MEASURED (Horn stencil & moving window)",
            "landing_ellipse_radius": "ASSUMPTION (User configurable)",
            "slope_screening_limit": "ASSUMPTION (Configurable flight rule)",
            "elevation_datum": dem.metadata.reference_datum
        }
    }
