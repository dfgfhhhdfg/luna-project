"""
LUNA NAVIGATOR — Geospatial & Lunar DEM Services (Member 3 Ownership)
"""

from app.services.geospatial.projection import (
    LunarPolarStereographic,
    lola_polar_projection
)
from app.services.geospatial.dem_loader import (
    DemLoader,
    DemRaster,
    DemMetadata,
    DemError,
    DemFileNotFoundError,
    DemFormatError,
    DemWindowError,
    generate_dem_report
)
from app.services.geospatial.terrain_analysis import (
    TerrainAnalysisError,
    DemCoordinateError,
    calculate_slope,
    calculate_slope_statistics,
    calculate_disjoint_slope_categories,
    calculate_roughness,
    calculate_roughness_statistics,
    calculate_elevation_statistics,
    calculate_terrain_profile,
    assess_landing_area,
    calculate_cliff_buffer_distance,
    evaluate_site_terrain_safety
)
from app.services.geospatial.horizon_service import (
    HorizonServiceError,
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

__all__ = [
    "LunarPolarStereographic",
    "lola_polar_projection",
    "DemLoader",
    "DemRaster",
    "DemMetadata",
    "DemError",
    "DemFileNotFoundError",
    "DemFormatError",
    "DemWindowError",
    "DemCoordinateError",
    "TerrainAnalysisError",
    "generate_dem_report",
    "calculate_slope",
    "calculate_slope_statistics",
    "calculate_disjoint_slope_categories",
    "calculate_roughness",
    "calculate_roughness_statistics",
    "calculate_elevation_statistics",
    "calculate_terrain_profile",
    "assess_landing_area",
    "calculate_cliff_buffer_distance",
    "evaluate_site_terrain_safety",
    "HorizonServiceError",
    "calculate_curvature_drop",
    "planar_ray_waypoints",
    "geodesic_ray_waypoints",
    "sample_elevations_vectorized",
    "sample_terrain_ray",
    "check_line_of_sight",
    "calculate_horizon_profile",
    "get_horizon_elevation_at_azimuth",
    "calculate_solar_irradiance",
    "calculate_solar_disk_visibility",
    "evaluate_solar_occultation_and_flux",
    "calculate_fresnel_clearance"
]
