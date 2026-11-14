"""
LUNA NAVIGATOR — Terrain Horizon & Obstruction Endpoints (Member 3 Ownership)
Exposes REST endpoints for 360° terrain horizon profiling, solar/Earth topographic
occultation, and surface-to-surface line-of-sight viewshed analysis.
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.engines.terrain.horizon import terrain_engine
from app.services.geospatial.dem_sampler import dem_model
from app.services.geospatial.projection import lola_polar_projection
from app.services.geospatial.dem_loader import DemLoader, DemRaster
from app.services.geospatial.terrain_analysis import (
    calculate_terrain_profile,
    assess_landing_area
)
from app.services.geospatial.horizon_service import (
    sample_terrain_ray,
    check_line_of_sight,
    calculate_horizon_profile
)
from app.schemas.terrain import (
    TerrainHorizonRequest,
    TerrainHorizonResponse,
    SolarObstructionRequest,
    SolarObstructionResult,
    EarthObstructionRequest,
    EarthObstructionResult,
    SurfaceViewshedRequest,
    SurfaceViewshedResponse,
    TerrainProfileRequest,
    TerrainProfileResponse,
    LandingAreaAssessmentRequest,
    LandingAreaAssessmentResponse,
    TerrainRayRequest,
    TerrainRayResponse,
    LineOfSightObstructionRequest,
    LineOfSightObstructionResponse,
    DemHorizonProfileRequest,
    DemHorizonProfileResponse
)

router = APIRouter()

_CACHED_DEM: Optional[DemRaster] = None


def get_active_dem() -> DemRaster:
    """Retrieve or load cached active DEM raster."""
    global _CACHED_DEM
    if _CACHED_DEM is not None:
        return _CACHED_DEM

    possible_paths = [
        Path("datasets/dem/synthetic_test_dem.tif"),
        Path("../datasets/dem/synthetic_test_dem.tif"),
        Path(__file__).resolve().parents[4] / "datasets" / "dem" / "synthetic_test_dem.tif",
    ]
    for p in possible_paths:
        if p.exists():
            _CACHED_DEM = DemLoader.load_dem(p)
            return _CACHED_DEM

    raise HTTPException(
        status_code=503,
        detail="No DEM raster loaded. Please place a valid LOLA GeoTIFF in datasets/dem/."
    )


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


@router.post("/terrain/profile", response_model=TerrainProfileResponse, tags=["Geospatial & Terrain"])
def compute_terrain_profile(request: TerrainProfileRequest):
    """
    Extract continuous elevation profile transect along a path between two coordinates.
    Accepts geographic (lat/lon) or projected (x/y) endpoints.
    """
    try:
        dem = get_active_dem()

        # Resolve start coordinate
        if request.start_x_m is not None and request.start_y_m is not None:
            start_coord = (request.start_x_m, request.start_y_m)
        elif request.start_lat is not None and request.start_lon is not None:
            if request.start_lat > -60.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid start coordinate: latitude {request.start_lat}° is outside south-polar mission range (-90° to -60°)."
                )
            try:
                start_coord = lola_polar_projection.forward(request.start_lat, request.start_lon)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid start coordinate: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Must provide start_lat/start_lon or start_x_m/start_y_m.")

        # Resolve end coordinate
        if request.end_x_m is not None and request.end_y_m is not None:
            end_coord = (request.end_x_m, request.end_y_m)
        elif request.end_lat is not None and request.end_lon is not None:
            if request.end_lat > -60.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid end coordinate: latitude {request.end_lat}° is outside south-polar mission range (-90° to -60°)."
                )
            try:
                end_coord = lola_polar_projection.forward(request.end_lat, request.end_lon)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid end coordinate: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Must provide end_lat/end_lon or end_x_m/end_y_m.")

        result = calculate_terrain_profile(
            dem,
            start_coord=start_coord,
            end_coord=end_coord,
            sample_step_m=request.sample_step_m,
            max_gap_allowed_m=request.max_gap_allowed_m,
            interpolation_method=request.interpolation_method
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input parameter: {str(e)}")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error during terrain profile calculation.")


@router.post("/terrain/assess-landing-area", response_model=LandingAreaAssessmentResponse, tags=["Geospatial & Terrain"])
def compute_landing_area_assessment(request: LandingAreaAssessmentRequest):
    """
    Comprehensive topographic assessment of a candidate landing site ellipse or circular zone.
    Computes slope distribution, moving-window roughness, slope-threshold safety coverage,
    and valid-data compliance for consumption by Member 4's multi-criteria recommendation engine.
    """
    try:
        dem = get_active_dem()

        if request.center_x_m is not None and request.center_y_m is not None:
            center_coord = (request.center_x_m, request.center_y_m)
        elif request.center_lat is not None and request.center_lon is not None:
            if request.center_lat > -60.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid center coordinate: latitude {request.center_lat}° is outside south-polar mission range (-90° to -60°)."
                )
            try:
                center_coord = lola_polar_projection.forward(request.center_lat, request.center_lon)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid center coordinate: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Must provide center_lat/center_lon or center_x_m/center_y_m.")

        result = assess_landing_area(
            dem,
            center_projected=center_coord,
            radius_m=request.radius_m,
            slope_thresholds=request.slope_thresholds,
            roughness_window_size=request.roughness_window_size
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input parameter: {str(e)}")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error during landing area assessment.")


# =============================================================================
# Phase 3 Endpoints: Terrain Ray Tracing & Line-of-Sight Visibility
# =============================================================================

@router.post("/terrain/ray", response_model=TerrainRayResponse, tags=["Geospatial & Terrain"])
def compute_terrain_ray(request: TerrainRayRequest):
    """
    Sample a radial transect across lunar terrain from an observer along an azimuth direction.
    Applies lunar spherical curvature correction (Delta_h = s^2 / 2R) and returns
    apparent elevation angles and peak horizon obstacle metrics.
    """
    try:
        dem = get_active_dem()

        if request.observer_lat is not None and request.observer_lat > -60.0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid observer coordinate: latitude {request.observer_lat}° is outside south-polar mission range (-90° to -60°)."
            )

        result = sample_terrain_ray(
            dem=dem,
            azimuth_deg=request.azimuth_deg,
            max_distance_m=request.max_distance_m,
            radial_step_m=request.radial_step_m,
            observer_height_m=request.observer_height_m,
            observer_lat=request.observer_lat,
            observer_lon=request.observer_lon,
            observer_x_m=request.observer_x_m,
            observer_y_m=request.observer_y_m,
            interpolation_method=request.interpolation_method,
            mode=request.mode
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid ray parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terrain ray processing failed: {str(e)}")


@router.post("/terrain/check-los", response_model=LineOfSightObstructionResponse, tags=["Geospatial & Terrain"])
def compute_line_of_sight(request: LineOfSightObstructionRequest):
    """
    Determine whether lunar terrain obstructs a target line-of-sight in a specified direction.
    Returns explicit tri-state visibility: 'obstructed', 'clear', or 'unknown' (when truncated
    by DEM boundary or NoData without confirming obstruction).
    """
    try:
        dem = get_active_dem()

        if request.observer_lat is not None and request.observer_lat > -60.0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid observer coordinate: latitude {request.observer_lat}° is outside south-polar mission range (-90° to -60°)."
            )

        result = check_line_of_sight(
            dem=dem,
            target_azimuth_deg=request.target_azimuth_deg,
            target_elevation_deg=request.target_elevation_deg,
            max_distance_m=request.max_distance_m,
            radial_step_m=request.radial_step_m,
            observer_height_m=request.observer_height_m,
            observer_lat=request.observer_lat,
            observer_lon=request.observer_lon,
            observer_x_m=request.observer_x_m,
            observer_y_m=request.observer_y_m,
            interpolation_method=request.interpolation_method,
            mode=request.mode
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid line-of-sight parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Line-of-sight evaluation failed: {str(e)}")


@router.post("/terrain/dem-horizon", response_model=DemHorizonProfileResponse, tags=["Geospatial & Terrain"])
def compute_dem_horizon_profile(request: DemHorizonProfileRequest):
    """
    Compute 360-degree local terrain horizon elevation profile directly from loaded DEM raster.
    """
    try:
        dem = get_active_dem()

        if request.observer_lat is not None and request.observer_lat > -60.0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid observer coordinate: latitude {request.observer_lat}° is outside south-polar mission range (-90° to -60°)."
            )

        result = calculate_horizon_profile(
            dem=dem,
            max_distance_m=request.max_distance_m,
            azimuth_step_deg=request.azimuth_step_deg,
            radial_step_m=request.radial_step_m,
            observer_height_m=request.observer_height_m,
            observer_lat=request.observer_lat,
            observer_lon=request.observer_lon,
            observer_x_m=request.observer_x_m,
            observer_y_m=request.observer_y_m,
            interpolation_method=request.interpolation_method,
            mode=request.mode
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid horizon profile parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DEM horizon profiling failed: {str(e)}")
