"""
LUNA NAVIGATOR — Terrain Horizon & Obstruction Schemas (Member 3 Ownership)
Defines strict Pydantic v2 schemas for terrain horizon profiling, solar/Earth
topographic occultation, and surface-to-surface viewshed line-of-sight analysis.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from app.schemas.common import CoordinatesRequest


class TerrainHorizonRequest(CoordinatesRequest):
    max_distance_km: float = Field(
        default=60.0,
        gt=1.0,
        le=150.0,
        description="Maximum radial ray-trace horizon search distance in km (typically 30–80 km for polar topography)"
    )
    azimuth_step_deg: float = Field(
        default=1.0,
        gt=0.1,
        le=10.0,
        description="Azimuth sampling resolution step in degrees [0.1° to 10°]"
    )
    observer_height_m: float = Field(
        default=2.0,
        ge=0.0,
        le=50.0,
        description="Height of instrument, solar panel array, or communications mast above ground in meters"
    )


class HorizonPoint(BaseModel):
    azimuth_deg: float = Field(..., description="Azimuth angle clockwise from North [0 to 360)")
    horizon_elevation_deg: float = Field(..., description="Maximum terrain obstruction elevation angle in degrees")
    obstruction_distance_km: float = Field(..., description="Distance in km to the blocking terrain feature")
    blocking_feature: Optional[str] = Field(default="Local Topographic Relief", description="Identified geomorphic feature")


class TerrainHorizonResponse(BaseModel):
    latitude: float
    longitude: float
    elevation_m: float
    observer_height_m: float
    elevation_datum: str = "Lunar Sphere R=1737.4 km (NASA LOLA)"
    horizon_points: List[HorizonPoint]
    max_horizon_elevation_deg: float
    min_horizon_elevation_deg: float
    average_horizon_elevation_deg: float
    site_slope_deg: Optional[float] = Field(None, description="Local topographic slope in degrees at the observer location")
    site_aspect_deg: Optional[float] = Field(None, description="Slope aspect / azimuth of maximum downward inclination")
    site_roughness_m: Optional[float] = Field(None, description="Surface elevation roughness (std dev) in meters")
    terrain_classification: Optional[str] = Field(None, description="Lander safety and geomorphic zone classification")


class SolarObstructionRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=0.0, le=360.0)
    elevation_m: float = Field(default=0.0)
    observer_height_m: float = Field(default=2.0, ge=0.0, le=50.0)
    sun_azimuth_deg: float = Field(..., ge=0.0, lt=360.0)
    sun_elevation_deg: float = Field(..., ge=-90.0, le=90.0)
    max_distance_km: float = Field(default=60.0, gt=1.0, le=150.0)


class SolarObstructionResult(BaseModel):
    sun_azimuth_deg: float
    sun_elevation_deg: float
    horizon_elevation_deg: float
    clearance_deg: float
    illumination_state: str = Field(..., description="SUNLIT (full disk above horizon), PENUMBRA (grazing rim), or SHADOW (occulted)")
    illumination_fraction: float = Field(..., ge=0.0, le=1.0, description="Fraction of solar disk unocculted [0.0 to 1.0]")
    effective_flux_w_m2: float = Field(..., ge=0.0, description="Effective solar irradiance incident on horizontal surface in W/m²")
    blocking_feature: Optional[str] = Field(None, description="Name of the blocking crater rim or massif if occulted")


class EarthObstructionRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=0.0, le=360.0)
    elevation_m: float = Field(default=0.0)
    observer_height_m: float = Field(default=2.0, ge=0.0, le=50.0)
    earth_azimuth_deg: float = Field(..., ge=0.0, lt=360.0)
    earth_elevation_deg: float = Field(..., ge=-90.0, le=90.0)
    max_distance_km: float = Field(default=60.0, gt=1.0, le=150.0)


class EarthObstructionResult(BaseModel):
    earth_azimuth_deg: float
    earth_elevation_deg: float
    horizon_elevation_deg: float
    clearance_deg: float
    los_state: str = Field(..., description="CLEAR (unobstructed DTE link), PARTIAL_GRAZING, or BLOCKED")
    link_margin_status: str = Field(..., description="Operational RF communication assessment")
    blocking_feature: Optional[str] = Field(None, description="Name of blocking terrain feature if occulted")


class ViewshedProfilePoint(BaseModel):
    distance_km: float
    terrain_elevation_m: float
    los_elevation_m: float
    clearance_m: float


class SurfaceViewshedRequest(BaseModel):
    site_a_lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude of Station A (e.g. Lander)")
    site_a_lon: float = Field(..., ge=0.0, le=360.0, description="Longitude of Station A")
    mast_height_a_m: float = Field(default=2.0, ge=0.0, le=50.0, description="Antenna mast height of Station A in meters")
    site_b_lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude of Station B (e.g. Rover / Relay)")
    site_b_lon: float = Field(..., ge=0.0, le=360.0, description="Longitude of Station B")
    mast_height_b_m: float = Field(default=1.5, ge=0.0, le=50.0, description="Antenna mast height of Station B in meters")


class SurfaceViewshedResponse(BaseModel):
    is_line_of_sight_clear: bool = Field(..., description="True if direct optical/RF path clears all intermediate lunar terrain")
    total_distance_km: float = Field(..., description="Great-circle ground distance between Station A and B")
    min_clearance_m: float = Field(..., description="Minimum vertical clearance of line-of-sight above terrain in meters")
    blocking_distance_km: Optional[float] = Field(None, description="Distance from Station A where terrain obstruction first occurs")
    blocking_feature: Optional[str] = Field(None, description="Name of the obstructing terrain ridge if blocked")
    profile: List[ViewshedProfilePoint] = Field(..., description="Elevation and LOS transect points along the path")


# =============================================================================
# Phase 2 Schemas: Slope, Roughness, Elevation Stats, Profile & Landing Area
# =============================================================================

class ElevationStatistics(BaseModel):
    min_elevation_m: Optional[float] = Field(None, description="Minimum elevation in meters above R=1737.4km datum")
    max_elevation_m: Optional[float] = Field(None, description="Maximum elevation in meters above R=1737.4km datum")
    mean_elevation_m: Optional[float] = Field(None, description="Arithmetic mean elevation in meters")
    median_elevation_m: Optional[float] = Field(None, description="50th percentile median elevation in meters")
    std_elevation_m: Optional[float] = Field(None, description="Elevation standard deviation in meters")
    valid_data_percent: float = Field(..., description="Percentage of valid data cells [0.0 to 100.0]")
    total_cells: int = Field(..., description="Total cell count examined")
    valid_cells: int = Field(..., description="Count of valid non-nodata cells")


class SlopeStatistics(BaseModel):
    min_slope_deg: Optional[float] = Field(None, description="Minimum slope in degrees [0 to 90]")
    max_slope_deg: Optional[float] = Field(None, description="Maximum slope in degrees [0 to 90]")
    mean_slope_deg: Optional[float] = Field(None, description="Mean slope in degrees")
    median_slope_deg: Optional[float] = Field(None, description="Median slope in degrees")
    std_slope_deg: Optional[float] = Field(None, description="Standard deviation of slope in degrees")
    threshold_coverage_pct: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of valid terrain with slope <= threshold (e.g. {'lte_5deg': 80.0, 'lte_12deg': 95.0})"
    )
    valid_data_percent: float = Field(..., description="Percentage of valid slope cells [0.0 to 100.0]")


class RoughnessStatistics(BaseModel):
    metric: str = Field(default="std_dev", description="Roughness metric applied (e.g. 'std_dev', 'relief')")
    window_size_pixels: int = Field(..., description="Moving window kernel dimension in pixels")
    pixel_size_x_m: float = Field(default=100.0, description="Pixel resolution in Easting (X) in meters")
    pixel_size_y_m: float = Field(default=100.0, description="Pixel resolution in Northing (Y) in meters")
    physical_baseline_x_m: float = Field(default=500.0, description="Physical moving window baseline in X (Easting) in meters: window_size_pixels * pixel_size_x_m")
    physical_baseline_y_m: float = Field(default=500.0, description="Physical moving window baseline in Y (Northing) in meters: window_size_pixels * pixel_size_y_m")
    physical_baseline_m: Optional[float] = Field(default=None, description="Physical baseline in meters (geometric mean for anisotropic pixels)")
    baseline_convention: str = Field(
        default="Anisotropic baselines: baseline_x = window_pixels * dx, baseline_y = window_pixels * dy",
        description="Explanation of non-square physical baseline conventions"
    )
    min_roughness_m: Optional[float] = Field(None, description="Minimum surface roughness in meters")
    max_roughness_m: Optional[float] = Field(None, description="Maximum surface roughness in meters")
    mean_roughness_m: Optional[float] = Field(None, description="Mean surface roughness in meters")
    median_roughness_m: Optional[float] = Field(None, description="Median surface roughness in meters")
    std_roughness_m: Optional[float] = Field(None, description="Roughness standard deviation in meters")
    valid_data_percent: float = Field(..., description="Percentage of valid roughness cells")


class TerrainProfilePoint(BaseModel):
    distance_m: float = Field(..., description="Cumulative distance from start point in meters")
    projected_x_m: float = Field(..., description="Projected Easting coordinate in meters")
    projected_y_m: float = Field(..., description="Projected Northing coordinate in meters")
    elevation_m: Optional[float] = Field(None, description="Interpolated elevation in meters above datum (or None if nodata)")


class TerrainProfileRequest(BaseModel):
    start_lat: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Start latitude in degrees")
    start_lon: Optional[float] = Field(None, ge=0.0, le=360.0, description="Start longitude in degrees")
    start_x_m: Optional[float] = Field(None, description="Start projected Easting in meters")
    start_y_m: Optional[float] = Field(None, description="Start projected Northing in meters")
    end_lat: Optional[float] = Field(None, ge=-90.0, le=90.0, description="End latitude in degrees")
    end_lon: Optional[float] = Field(None, ge=0.0, le=360.0, description="End longitude in degrees")
    end_x_m: Optional[float] = Field(None, description="End projected Easting in meters")
    end_y_m: Optional[float] = Field(None, description="End projected Northing in meters")
    sample_step_m: float = Field(default=100.0, gt=1.0, le=5000.0, description="Sample step interval along transect in meters")
    max_gap_allowed_m: float = Field(default=500.0, gt=0.0, description="Maximum allowable continuous gap in meters before warning")
    interpolation_method: str = Field(default="bilinear", description="Interpolation method: 'bilinear' (continuous) or 'nearest'")


class TerrainProfileResponse(BaseModel):
    profile_length_m: float = Field(..., description="Total transect length in meters")
    sample_step_m: float = Field(..., description="Sampling interval in meters")
    points: List[TerrainProfilePoint] = Field(..., description="Sample points along the path")
    min_elevation_m: Optional[float] = Field(None, description="Minimum elevation along profile")
    max_elevation_m: Optional[float] = Field(None, description="Maximum elevation along profile")
    valid_coverage_percent: float = Field(..., description="Percentage of transect with valid DEM coverage")
    warnings: List[str] = Field(default_factory=list, description="Coverage and data gap warnings")
    interpolation_method: str = Field(default="bilinear", description="Applied interpolation method")
    elevation_datum: str = "Lunar Sphere R=1737.4 km (NASA LOLA)"


class LandingAreaAssessmentRequest(BaseModel):
    center_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    center_lon: Optional[float] = Field(None, ge=0.0, le=360.0)
    center_x_m: Optional[float] = Field(None)
    center_y_m: Optional[float] = Field(None)
    radius_m: float = Field(default=1000.0, gt=50.0, le=25000.0, description="Analysis radius in meters")
    slope_thresholds: List[float] = Field(default=[5.0, 10.0, 12.0, 15.0], description="Slope thresholds in degrees")
    roughness_window_size: int = Field(default=5, ge=3, le=21, description="Kernel window size for roughness in pixels")


class LandingAreaAssessmentResponse(BaseModel):
    center_projected: Tuple[float, float] = Field(..., description="(x_m, y_m) projected coordinates")
    radius_m: float = Field(..., description="Evaluated landing circle radius in meters")
    elevation_stats: ElevationStatistics
    slope_stats: SlopeStatistics
    roughness_stats: RoughnessStatistics
    valid_data_percent: float = Field(..., description="Overall valid data percentage inside candidate region")
    warnings: List[str] = Field(default_factory=list, description="Quality and coverage warnings")
    disclaimer: str = Field(
        default="Topographic assessment reflects slope and elevation roughness metrics only; does not evaluate sub-pixel boulder hazards (<20m) or regolith bearing strength.",
        description="Planetary engineering disclaimer"
    )


# =============================================================================
# Phase 3 Schemas: Terrain Ray Tracing & Line-of-Sight Visibility
# =============================================================================

class RaySamplePoint(BaseModel):
    distance_m: float = Field(..., description="Ground distance from observer in meters")
    projected_x_m: float = Field(..., description="Projected Easting in meters")
    projected_y_m: float = Field(..., description="Projected Northing in meters")
    latitude_deg: Optional[float] = Field(None, description="Selenographic latitude in degrees")
    longitude_deg: Optional[float] = Field(None, description="Selenographic longitude in degrees")
    terrain_elevation_m: Optional[float] = Field(None, description="Raw terrain elevation in meters above datum")
    apparent_elevation_m: Optional[float] = Field(None, description="Apparent elevation relative to local tangent plane with curvature drop subtracted")
    elevation_angle_deg: Optional[float] = Field(None, description="Elevation angle in degrees above observer horizontal tangent plane")
    curvature_drop_m: float = Field(..., description="Elevation drop in meters due to lunar spherical curvature (s^2 / 2R)")
    is_valid: bool = Field(..., description="True if sample is inside DEM coverage and not NoData")


class TerrainRayRequest(BaseModel):
    observer_lat: Optional[float] = Field(None, ge=-90.0, le=0.0, description="Observer latitude in degrees [-90, 0]")
    observer_lon: Optional[float] = Field(None, ge=0.0, le=360.0, description="Observer longitude in degrees [0, 360)")
    observer_x_m: Optional[float] = Field(None, description="Observer projected Easting in meters")
    observer_y_m: Optional[float] = Field(None, description="Observer projected Northing in meters")
    azimuth_deg: float = Field(default=0.0, description="Azimuth angle clockwise from True North [0, 360)")
    max_distance_m: float = Field(default=30000.0, gt=10.0, le=150000.0, description="Max ray tracing distance in meters")
    radial_step_m: float = Field(default=100.0, gt=1.0, le=5000.0, description="Radial step increment in meters")
    observer_height_m: float = Field(default=2.0, ge=0.0, le=50.0, description="Observer instrument height above ground in meters")
    interpolation_method: str = Field(default="bilinear", description="Interpolation method: 'bilinear' or 'nearest'")
    mode: str = Field(default="auto", description="Trajectory mode: 'auto', 'projected', or 'spherical'")


class TerrainRayResponse(BaseModel):
    observer_projected: Tuple[float, float] = Field(..., description="(x_m, y_m) projected coordinates")
    observer_elevation_m: Optional[float] = Field(None, description="Ground elevation at observer location")
    observer_height_m: float = Field(..., description="Observer instrument height above surface in meters")
    azimuth_deg: float = Field(..., description="Normalized azimuth angle [0, 360)")
    max_distance_m: float = Field(..., description="Requested maximum ray distance in meters")
    radial_step_m: float = Field(..., description="Radial sampling resolution in meters")
    max_horizon_angle_deg: Optional[float] = Field(None, description="Maximum terrain elevation angle encountered along ray")
    max_horizon_distance_m: Optional[float] = Field(None, description="Distance to highest terrain feature along ray in meters")
    max_horizon_point: Optional[Dict[str, Any]] = Field(None, description="Coordinates and elevation of peak horizon obstacle")
    samples: List[RaySamplePoint] = Field(default_factory=list, description="Transect points along the ray")
    evaluated_distance_m: float = Field(..., description="Distance traversed before boundary exit or completion in meters")
    has_boundary_exit: bool = Field(..., description="True if ray exited DEM boundary before max_distance_m")
    has_nodata: bool = Field(..., description="True if ray crossed NoData cells")
    warnings: List[str] = Field(default_factory=list, description="Coverage and scientific warnings")
    mode: str = Field(..., description="Applied trajectory mode ('projected' or 'spherical')")
    elevation_datum: str = "Lunar Sphere R=1737.4 km (NASA LOLA)"


class LineOfSightObstructionRequest(BaseModel):
    target_azimuth_deg: float = Field(..., description="Azimuth to target body clockwise from North [0, 360)")
    target_elevation_deg: float = Field(..., ge=-90.0, le=90.0, description="Elevation angle of target body above horizontal tangent plane [-90°, +90°]")
    observer_lat: Optional[float] = Field(None, ge=-90.0, le=0.0, description="Observer selenographic latitude")
    observer_lon: Optional[float] = Field(None, ge=0.0, le=360.0, description="Observer selenographic longitude")
    observer_x_m: Optional[float] = Field(None, description="Observer projected Easting in meters")
    observer_y_m: Optional[float] = Field(None, description="Observer projected Northing in meters")
    max_distance_m: float = Field(default=60000.0, gt=10.0, le=150000.0, description="Maximum search distance in meters")
    radial_step_m: float = Field(default=100.0, gt=1.0, le=5000.0, description="Radial step increment in meters")
    observer_height_m: float = Field(default=2.0, ge=0.0, le=50.0, description="Observer instrument height above surface in meters")
    interpolation_method: str = Field(default="bilinear", description="Interpolation method: 'bilinear' or 'nearest'")
    mode: str = Field(default="auto", description="Trajectory mode: 'auto', 'projected', or 'spherical'")


class LineOfSightObstructionResponse(BaseModel):
    visibility_state: str = Field(..., description="Tri-state visibility: 'obstructed', 'clear', or 'unknown'")
    is_obstructed: Optional[bool] = Field(None, description="True if obstructed, False if clear, None if unknown")
    target_azimuth_deg: float = Field(..., description="Normalized target azimuth angle [0, 360)")
    target_elevation_deg: float = Field(..., description="Target elevation angle above tangent plane in degrees")
    horizon_elevation_deg: Optional[float] = Field(None, description="Highest terrain horizon angle along the ray in degrees")
    elevation_margin_deg: Optional[float] = Field(None, description="Clearance margin: target_elevation - horizon_elevation (>0 clear, <=0 obstructed)")
    obstructing_distance_m: Optional[float] = Field(None, description="Distance in meters to the first obstructing terrain feature")
    obstructing_point: Optional[Dict[str, Any]] = Field(None, description="Coordinates and elevation of first obstructing terrain obstacle")
    highest_point: Optional[Dict[str, Any]] = Field(None, description="Coordinates and elevation of highest terrain obstacle along ray")
    evaluated_distance_m: float = Field(..., description="Distance in meters evaluated before obstruction, boundary, or completion")
    max_distance_m: float = Field(..., description="Requested maximum search radius in meters")
    reason: str = Field(..., description="Scientific explanation of visibility determination")
    warnings: List[str] = Field(default_factory=list, description="Coverage and scientific warnings")
    elevation_datum: str = "Lunar Sphere R=1737.4 km (NASA LOLA)"


class DemHorizonPoint(BaseModel):
    azimuth_deg: float = Field(..., description="Azimuth angle clockwise from True North [0, 360)")
    horizon_elevation_deg: Optional[float] = Field(None, description="Maximum terrain elevation angle in degrees")
    obstruction_distance_m: Optional[float] = Field(None, description="Distance in meters to highest terrain obstacle")
    highest_point: Optional[Dict[str, Any]] = Field(None, description="Coordinates of peak terrain obstacle")
    has_boundary_exit: bool = Field(..., description="True if ray exited DEM before max distance")
    has_nodata: bool = Field(..., description="True if ray crossed NoData cells")


class DemHorizonProfileRequest(BaseModel):
    observer_lat: Optional[float] = Field(None, ge=-90.0, le=0.0)
    observer_lon: Optional[float] = Field(None, ge=0.0, le=360.0)
    observer_x_m: Optional[float] = Field(None)
    observer_y_m: Optional[float] = Field(None)
    max_distance_m: float = Field(default=30000.0, gt=10.0, le=150000.0, description="Search distance in meters")
    azimuth_step_deg: float = Field(default=5.0, gt=0.1, le=45.0, description="Azimuth sampling step in degrees")
    radial_step_m: float = Field(default=100.0, gt=1.0, le=5000.0, description="Radial step in meters")
    observer_height_m: float = Field(default=2.0, ge=0.0, le=50.0, description="Observer height above surface in meters")
    interpolation_method: str = Field(default="bilinear", description="'bilinear' or 'nearest'")
    mode: str = Field(default="auto", description="'auto', 'projected', or 'spherical'")


class DemHorizonProfileResponse(BaseModel):
    observer_projected: Optional[Tuple[float, float]] = None
    observer_elevation_m: Optional[float] = None
    observer_height_m: float
    max_distance_m: float
    azimuth_step_deg: float
    radial_step_m: float
    horizon_points: List[DemHorizonPoint]
    min_horizon_elevation_deg: Optional[float] = None
    max_horizon_elevation_deg: Optional[float] = None
    mean_horizon_elevation_deg: Optional[float] = None
    azimuth_of_max_elevation_deg: Optional[float] = None
    azimuth_of_min_elevation_deg: Optional[float] = None
    valid_coverage_percent: float = Field(..., description="Percentage of azimuth rays with complete valid coverage")
    elevation_datum: str = "Lunar Sphere R=1737.4 km (NASA LOLA)"


# =============================================================================
# Phase 4 Schemas: Disjoint Slopes, Cliff Buffers, Fresnel & GIS-to-MCDA Metrics
# =============================================================================

class DisjointSlopeBreakdown(BaseModel):
    p_0_to_5_deg: float = Field(..., description="Percentage of valid cells with slope in [0°, 5°] (Gentle/Nominal)")
    p_5_to_10_deg: float = Field(..., description="Percentage of valid cells with slope in (5°, 10°] (Moderate/Acceptable)")
    p_10_to_15_deg: float = Field(..., description="Percentage of valid cells with slope in (10°, 15°] (Steep/Marginal)")
    p_gt_15_deg: float = Field(..., description="Percentage of valid cells with slope > 15° (Hazardous/Severe)")
    cell_counts: Dict[str, int] = Field(default_factory=dict, description="Raw pixel counts per category")
    valid_cells: int
    total_cells: int
    slope_distribution_score: float = Field(..., description="Normalized slope score in [0.0, 100.0]")
    is_valid: bool


class CliffBufferResponse(BaseModel):
    has_cliff_in_radius: bool = Field(..., description="True if a cliff (slope >= threshold) was detected within radius")
    cliff_distance_m: Optional[float] = Field(None, description="Euclidean distance in meters to nearest cliff edge")
    nearest_cliff_coord: Optional[Tuple[float, float]] = Field(None, description="(x, y) coordinates of nearest cliff cell")
    nearest_cliff_slope_deg: Optional[float] = Field(None, description="Slope in degrees of nearest cliff cell")
    cliff_slope_threshold_deg: float = 20.0
    search_radius_m: float = 2000.0
    status: str = Field(..., description="'CLIFF_DETECTED', 'NO_CLIFF_WITHIN_SEARCH_RADIUS', 'CENTER_OUTSIDE_DEM'")


class SolarDiskVisibilityResponse(BaseModel):
    visibility_state: str = Field(..., description="'full', 'partial', 'occulted', or 'unknown'")
    visible_fraction: float = Field(..., ge=0.0, le=1.0, description="Visible disk area fraction in [0.0, 1.0]")
    is_illuminated: bool
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: float
    horizon_elevation_deg: Optional[float] = None
    clearance_deg: Optional[float] = None
    solar_irradiance_w_m2: Optional[float] = None
    direct_normal_flux_w_m2: Optional[float] = None
    horizontal_flux_w_m2: Optional[float] = None
    is_distance_scaled: bool = False
    sun_distance_km: Optional[float] = None
    provenance: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class FresnelClearanceResponse(BaseModel):
    obstacle_distance_m: float = Field(..., description="Distance d1 from lunar antenna to obstructing ridge in meters along topocentric azimuth")
    earth_distance_m: float = Field(..., description="Distance d_Earth to Earth receiving station in meters")
    frequency_hz: float = Field(..., description="Carrier frequency in Hertz")
    wavelength_m: float = Field(..., description="Carrier wavelength in meters (c / frequency_hz)")
    fresnel_radius_r1_m: float = Field(..., description="First Fresnel zone radius r_F1 in meters at obstacle plane: sqrt(lambda * d1 * d2 / (d1 + d2))")
    screening_fraction: float = Field(0.6, description="Screening clearance fraction (default 0.6 * r_F1)")
    required_clearance_m: float = Field(..., description="Required physical clearance (screening_fraction * r_F1) in meters")
    vertical_clearance_m: float = Field(..., description="Exact vertical clearance Delta_z above ridge in meters: d1 * (tan(theta_Earth) - tan(theta_obs))")
    perpendicular_clearance_m: float = Field(..., description="Perpendicular distance from obstacle knife-edge to direct line-of-sight ray: Delta_z * cos(theta_Earth)")
    clearance_margin_m: float = Field(..., description="Margin: perpendicular_clearance - required_clearance in meters")
    fresnel_screening_pass: bool = Field(..., description="True if perpendicular clearance >= 0.6 * r_F1")
    earth_elevation_deg: float = Field(..., description="Topocentric Earth elevation angle in degrees above local tangent plane at lunar transmitter")
    obstacle_elevation_angle_deg: float = Field(..., description="Topocentric obstacle elevation angle in degrees above local tangent plane at lunar transmitter")
    units: Dict[str, str] = Field(default_factory=dict)
    provenance: Dict[str, str] = Field(default_factory=dict)


class SiteTerrainSafetyAssessment(BaseModel):
    center_projected_m: Tuple[float, float]
    center_geographic_deg: Optional[Tuple[float, float]] = None
    landing_radius_m: float
    dem_coverage_percent: float
    is_coverage_compliant: bool
    elevation_m: Dict[str, Optional[float]]
    disjoint_slope_pcts: Dict[str, float]
    slope_distribution_score: float
    max_slope_deg: Optional[float] = None
    mean_slope_deg: Optional[float] = None
    is_slope_compliant: bool
    roughness_std_m: Optional[float] = None
    roughness_score: Optional[float] = None
    terrain_safety_score: Optional[float] = None
    cliff_buffer: Dict[str, Any]
    is_safety_gated: bool
    violations: List[str]
    warnings: List[str]
    units: Dict[str, str]
    provenance: Dict[str, str]
