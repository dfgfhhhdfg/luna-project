"""
LUNA NAVIGATOR — Terrain Horizon & Obstruction Schemas (Member 3 Ownership)
Defines strict Pydantic v2 schemas for terrain horizon profiling, solar/Earth
topographic occultation, and surface-to-surface viewshed line-of-sight analysis.
"""

from typing import List, Optional
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
