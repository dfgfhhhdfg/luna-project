"""
LUNA NAVIGATOR — Terrain Horizon & Obstruction Endpoints (Member 3 Ownership)
Exposes REST endpoints for 360° terrain horizon profiling, solar/Earth topographic
occultation, and surface-to-surface line-of-sight viewshed analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from app.engines.terrain.horizon import terrain_engine
from app.services.geospatial.dem_sampler import dem_model
from app.schemas.terrain import (
    TerrainHorizonRequest,
    TerrainHorizonResponse,
    SolarObstructionRequest,
    SolarObstructionResult,
    EarthObstructionRequest,
    EarthObstructionResult,
    SurfaceViewshedRequest,
    SurfaceViewshedResponse
)

router = APIRouter()


@router.post("/terrain/horizon", response_model=TerrainHorizonResponse, tags=["Geospatial & Terrain"])
def compute_terrain_horizon(request: TerrainHorizonRequest):
    """
    Calculate 360-degree local terrain horizon elevation profile and ray-tracing obstruction masks
    from NASA LOLA elevation models with lunar curvature correction.
    """
    try:
        return terrain_engine.generate_horizon_profile(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terrain horizon processing failed: {str(e)}")


@router.post("/terrain/solar-obstruction", response_model=SolarObstructionResult, tags=["Geospatial & Terrain"])
def compute_solar_obstruction(request: SolarObstructionRequest):
    """
    Evaluate whether the Sun is visible, occulted by crater rims, or in penumbra
    at a specific lunar coordinate and epoch.
    """
    try:
        # Generate horizon profile
        horizon_req = TerrainHorizonRequest(
            latitude=request.latitude,
            longitude=request.longitude,
            elevation_m=request.elevation_m,
            observer_height_m=request.observer_height_m,
            max_distance_km=request.max_distance_km,
            azimuth_step_deg=2.0
        )
        horizon = terrain_engine.generate_horizon_profile(horizon_req)
        return terrain_engine.calculate_solar_obstruction(
            horizon, request.sun_azimuth_deg, request.sun_elevation_deg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solar obstruction evaluation failed: {str(e)}")


@router.post("/terrain/earth-obstruction", response_model=EarthObstructionResult, tags=["Geospatial & Terrain"])
def compute_earth_obstruction(request: EarthObstructionRequest):
    """
    Evaluate Direct-to-Earth (DTE) communication visibility and RF link margin status
    against local topographic masking and lunar libration.
    """
    try:
        horizon_req = TerrainHorizonRequest(
            latitude=request.latitude,
            longitude=request.longitude,
            elevation_m=request.elevation_m,
            observer_height_m=request.observer_height_m,
            max_distance_km=request.max_distance_km,
            azimuth_step_deg=2.0
        )
        horizon = terrain_engine.generate_horizon_profile(horizon_req)
        return terrain_engine.calculate_earth_obstruction(
            horizon, request.earth_azimuth_deg, request.earth_elevation_deg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Earth obstruction evaluation failed: {str(e)}")


@router.post("/terrain/viewshed", response_model=SurfaceViewshedResponse, tags=["Geospatial & Terrain"])
def compute_surface_viewshed(request: SurfaceViewshedRequest):
    """
    Calculate surface-to-surface line-of-sight clearance between two surface assets
    (e.g., lander to rover, or lander to optical communication mast).
    """
    try:
        return terrain_engine.calculate_viewshed(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Surface viewshed ray-tracing failed: {str(e)}")


@router.get("/terrain/sample", tags=["Geospatial & Terrain"])
def sample_terrain_point(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=0.0, le=360.0),
):
    """
    Sample point elevation, slope, aspect, and roughness metrics at an arbitrary lunar coordinate.
    """
    try:
        metrics = dem_model.sample_terrain_metrics(latitude, longitude)
        return {
            "latitude": latitude,
            "longitude": longitude,
            "metrics": metrics.to_dict(),
            "closest_feature": dem_model.identify_feature(latitude, longitude),
            "elevation_datum": "Lunar Sphere R=1737.4 km (NASA LOLA)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terrain point sampling failed: {str(e)}")
