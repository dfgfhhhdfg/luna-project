"""
LUNA NAVIGATOR — Lunar DEM Loader & Metadata Reporting Engine (Member 3 Ownership)

Provides robust, memory-efficient loading, validation, windowed extraction, and
metadata reporting for Lunar Digital Elevation Models (DEMs) conforming to NASA LOLA standards.

Supported Formats:
- GeoTIFF (.tif, .tiff) with embedded planetary polar stereographic tags (via Rasterio).
- NumPy compressed/uncompressed gridded rasters (.npz, .npy).
- Fallback raster parsing with graceful degradation if native GDAL drivers are unavailable.

Elevation Datum:
- Reference Sphere: R_moon = 1,737,400.0 meters.
- Elevation Units: Meters relative to the standard lunar datum.
"""

import os
import math
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, Sequence
from dataclasses import dataclass, asdict
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.errors import RasterioIOError
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class DemError(Exception):
    """Base exception for lunar DEM operations."""
    pass


class DemFileNotFoundError(DemError):
    """Raised when the requested DEM file does not exist on disk."""
    pass


class DemFormatError(DemError):
    """Raised when the file format is unsupported or the header is corrupted."""
    pass


class DemWindowError(DemError):
    """Raised when the requested spatial extraction window lies outside the raster bounds."""
    pass


@dataclass
class DemMetadata:
    """Standardized metadata representation for a Lunar DEM dataset."""
    dataset_name: str
    data_source: str
    file_path: str
    width_pixels: int
    height_pixels: int
    pixel_size_x_m: float
    pixel_size_y_m: float
    crs_repr: Optional[str]
    bounds_projected: Tuple[float, float, float, float]  # (left_m, bottom_m, right_m, top_m)
    elevation_units: str
    reference_datum: str
    nodata_value: Optional[float]
    valid_data_percent: float
    min_elevation_m: Optional[float]
    max_elevation_m: Optional[float]
    mean_elevation_m: Optional[float]
    is_synthetic: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for reporting and API output."""
        d = asdict(self)
        # Round numerical metrics for clean serialization
        if self.min_elevation_m is not None:
            d["min_elevation_m"] = round(self.min_elevation_m, 2)
        if self.max_elevation_m is not None:
            d["max_elevation_m"] = round(self.max_elevation_m, 2)
        if self.mean_elevation_m is not None:
            d["mean_elevation_m"] = round(self.mean_elevation_m, 2)
        d["valid_data_percent"] = round(self.valid_data_percent, 2)
        d["pixel_size_x_m"] = round(self.pixel_size_x_m, 2)
        d["pixel_size_y_m"] = round(self.pixel_size_y_m, 2)
        return d


class DemRaster:
    """
    Container holding loaded DEM elevation array and spatial referencing transform.
    Supports in-memory querying and sub-window spatial slicing.
    """

    def __init__(
        self,
        elevation_array: np.ndarray,
        metadata: DemMetadata,
        transform: Optional[Any] = None
    ):
        self.elevation = elevation_array
        self.metadata = metadata
        self.transform = transform

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    def get_elevation_at_pixel(self, col: int, row: int) -> float:
        """Query elevation by raster column and row indices."""
        h, w = self.elevation.shape
        if not (0 <= col < w and 0 <= row < h):
            raise IndexError(f"Pixel index ({col}, {row}) out of raster bounds ({w}, {h}).")

        val = float(self.elevation[row, col])
        if self.metadata.nodata_value is not None and val == self.metadata.nodata_value:
            return float("nan")
        return val

    def get_elevation_at_projected(
        self,
        x_m: float,
        y_m: float,
        method: str = "bilinear"
    ) -> float:
        """
        Query elevation by planar projected coordinates (X Easting, Y Northing in meters).
        
        Args:
            x_m: Projected Easting in meters.
            y_m: Projected Northing in meters.
            method: 'bilinear' (default, continuous 4-point surface interpolation) or
                    'nearest' (discrete pixel center lookup).
                    
        Returns:
            Elevation in meters, or float('nan') if out of bounds or if any required neighbor is nodata.
        """
        left, bottom, right, top = self.metadata.bounds_projected
        if not (left <= x_m <= right and bottom <= y_m <= top):
            return float("nan")

        w = self.metadata.width_pixels
        h = self.metadata.height_pixels

        # Convert projected coordinate to continuous pixel coordinates (u column, v row)
        if self.transform is not None and HAS_RASTERIO:
            try:
                inv_t = ~self.transform
                u = inv_t.a * x_m + inv_t.b * y_m + inv_t.c
                v = inv_t.d * x_m + inv_t.e * y_m + inv_t.f
            except Exception:
                u = (x_m - left) / self.metadata.pixel_size_x_m
                v = (top - y_m) / self.metadata.pixel_size_y_m
        else:
            u = (x_m - left) / self.metadata.pixel_size_x_m
            v = (top - y_m) / self.metadata.pixel_size_y_m

        if method == "nearest":
            col = int(math.floor(u))
            row = int(math.floor(v))
            col = min(max(0, col), w - 1)
            row = min(max(0, row), h - 1)
            return self.get_elevation_at_pixel(col, row)

        elif method == "bilinear":
            # Pixel center is at (c + 0.5, r + 0.5)
            u_c = u - 0.5
            v_c = v - 0.5

            c0 = int(math.floor(u_c))
            r0 = int(math.floor(v_c))
            c1 = c0 + 1
            r1 = r0 + 1

            # Strict boundary policy: all 4 neighbors must be inside raster bounds
            if c0 < 0 or c1 >= w or r0 < 0 or r1 >= h:
                return float("nan")

            z00 = self.get_elevation_at_pixel(c0, r0)
            z10 = self.get_elevation_at_pixel(c1, r0)
            z01 = self.get_elevation_at_pixel(c0, r1)
            z11 = self.get_elevation_at_pixel(c1, r1)

            # Strict nodata policy: if ANY corner is nodata/NaN, return NaN (no silent substitution)
            if math.isnan(z00) or math.isnan(z10) or math.isnan(z01) or math.isnan(z11):
                return float("nan")

            alpha = u_c - c0
            beta = v_c - r0

            z = (
                (1.0 - alpha) * (1.0 - beta) * z00 +
                alpha * (1.0 - beta) * z10 +
                (1.0 - alpha) * beta * z01 +
                alpha * beta * z11
            )
            return float(z)

        else:
            raise ValueError(f"Unsupported interpolation method '{method}'. Supported: 'bilinear', 'nearest'")

    def calculate_slope(self) -> np.ndarray:
        """Calculate slope array in degrees for the loaded DEM."""
        from app.services.geospatial.terrain_analysis import calculate_slope
        return calculate_slope(self)

    def calculate_roughness(self, window_size: int = 5, metric: str = "std") -> np.ndarray:
        """Calculate terrain roughness array in meters (std dev or relief)."""
        from app.services.geospatial.terrain_analysis import calculate_roughness
        return calculate_roughness(self, window_size=window_size, metric=metric)

    def calculate_elevation_statistics(self) -> Dict[str, Any]:
        """Calculate elevation summary statistics across the raster."""
        from app.services.geospatial.terrain_analysis import calculate_elevation_statistics
        return calculate_elevation_statistics(self.elevation, nodata_value=self.metadata.nodata_value)

    def calculate_terrain_profile(
        self,
        start_coord: Tuple[float, float],
        end_coord: Tuple[float, float],
        sample_step_m: float = 100.0,
        max_gap_allowed_m: float = 500.0,
        interpolation_method: str = "bilinear"
    ) -> Dict[str, Any]:
        """Extract continuous elevation profile transect between two projected coordinates."""
        from app.services.geospatial.terrain_analysis import calculate_terrain_profile
        return calculate_terrain_profile(
            self,
            start_coord,
            end_coord,
            sample_step_m,
            max_gap_allowed_m,
            interpolation_method=interpolation_method
        )

    def assess_landing_area(
        self,
        center_projected: Tuple[float, float],
        radius_m: float = 1000.0,
        slope_thresholds: Sequence[float] = (5.0, 10.0, 12.0, 15.0),
        roughness_window_size: int = 5
    ) -> Dict[str, Any]:
        """Perform comprehensive landing site topographic assessment."""
        from app.services.geospatial.terrain_analysis import assess_landing_area
        return assess_landing_area(self, center_projected, radius_m, slope_thresholds, roughness_window_size)

    def sample_terrain_ray(
        self,
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
        """Sample a radial terrain ray transect with lunar curvature correction."""
        from app.services.geospatial.horizon_service import sample_terrain_ray
        return sample_terrain_ray(
            self,
            observer_coord=observer_coord,
            azimuth_deg=azimuth_deg,
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

    def check_line_of_sight(
        self,
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
        """Evaluate line-of-sight obstruction for a target body with tri-state status."""
        from app.services.geospatial.horizon_service import check_line_of_sight
        return check_line_of_sight(
            self,
            target_azimuth_deg=target_azimuth_deg,
            target_elevation_deg=target_elevation_deg,
            observer_coord=observer_coord,
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

    def calculate_horizon_profile(
        self,
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
        """Calculate 360-degree local terrain horizon elevation profile."""
        from app.services.geospatial.horizon_service import calculate_horizon_profile
        return calculate_horizon_profile(
            self,
            observer_coord=observer_coord,
            max_distance_m=max_distance_m,
            azimuth_step_deg=azimuth_step_deg,
            radial_step_m=radial_step_m,
            observer_height_m=observer_height_m,
            observer_lat=observer_lat,
            observer_lon=observer_lon,
            observer_x_m=observer_x_m,
            observer_y_m=observer_y_m,
            interpolation_method=interpolation_method,
            mode=mode
        )

    def calculate_cliff_buffer_distance(
        self,
        center_projected: Tuple[float, float],
        cliff_slope_threshold_deg: float = 20.0,
        max_search_radius_m: float = 2000.0
    ) -> Dict[str, Any]:
        """Extract distance from candidate landing center to nearest terrain slope >= 20°."""
        from app.services.geospatial.terrain_analysis import calculate_cliff_buffer_distance
        return calculate_cliff_buffer_distance(
            self,
            center_projected=center_projected,
            cliff_slope_threshold_deg=cliff_slope_threshold_deg,
            max_search_radius_m=max_search_radius_m
        )

    def evaluate_site_terrain_safety(
        self,
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
        """GIS-to-MCDA Metrics Adapter: Evaluates landing footprint topographic safety."""
        from app.services.geospatial.terrain_analysis import evaluate_site_terrain_safety
        return evaluate_site_terrain_safety(
            self,
            center_coord=center_coord,
            center_lat=center_lat,
            center_lon=center_lon,
            center_x_m=center_x_m,
            center_y_m=center_y_m,
            landing_radius_m=landing_radius_m,
            max_acceptable_slope_deg=max_acceptable_slope_deg,
            min_dem_coverage_percent=min_dem_coverage_percent,
            roughness_window_size=roughness_window_size,
            cliff_search_radius_m=cliff_search_radius_m
        )


class DemLoader:
    """
    Production DEM loader supporting GeoTIFF and NumPy arrays with memory-conscious
    sub-window reading and complete metadata validation.
    """

    DEFAULT_LUNAR_DATUM = "Lunar Sphere R=1737.4 km (NASA LOLA)"
    DEFAULT_UNITS = "meters"

    @classmethod
    def inspect_metadata(cls, file_path: Union[str, Path]) -> DemMetadata:
        """
        Quickly inspect raster header and metadata without reading entire raster into memory.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise DemFileNotFoundError(f"DEM file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in [".tif", ".tiff"]:
            return cls._inspect_geotiff(path)
        elif suffix in [".npz", ".npy"]:
            return cls._inspect_numpy(path)
        else:
            raise DemFormatError(f"Unsupported DEM file extension '{suffix}'. Supported: .tif, .tiff, .npz, .npy")

    @classmethod
    def load_dem(
        cls,
        file_path: Union[str, Path],
        window_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> DemRaster:
        """
        Load a DEM raster into memory.
        
        Args:
            file_path: Path to DEM file (.tif, .tiff, .npz).
            window_bounds: Optional projected bounding box (left, bottom, right, top) in meters.
                           When specified, only the subset region is loaded into RAM.
                           
        Returns:
            DemRaster container.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise DemFileNotFoundError(f"DEM file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".tif", ".tiff"]:
            return cls._load_geotiff(path, window_bounds)
        elif suffix in [".npz", ".npy"]:
            return cls._load_numpy(path, window_bounds)
        else:
            raise DemFormatError(f"Unsupported DEM format '{suffix}'.")

    # -------------------------------------------------------------------------
    # GeoTIFF Driver
    # -------------------------------------------------------------------------

    @classmethod
    def _inspect_geotiff(cls, path: Path) -> DemMetadata:
        if not HAS_RASTERIO:
            raise DemError("Rasterio is required to inspect GeoTIFF files.")

        try:
            with rasterio.open(path) as src:
                width = src.width
                height = src.height
                bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                res_x, res_y = abs(src.res[0]), abs(src.res[1])
                crs_str = str(src.crs) if src.crs else "UNKNOWN"
                nodata = src.nodata

                # Sample quick decimation statistics for valid data percentage without loading full raster
                target_h = max(1, min(height, 256))
                target_w = max(1, min(width, 256))
                sub_sample = src.read(1, out_shape=(target_h, target_w))

                if nodata is not None:
                    valid_mask = (sub_sample != nodata) & ~np.isnan(sub_sample)
                else:
                    valid_mask = ~np.isnan(sub_sample)

                valid_pct = float(np.count_nonzero(valid_mask) / sub_sample.size * 100.0)

                valid_vals = sub_sample[valid_mask]
                min_el = float(np.min(valid_vals)) if len(valid_vals) > 0 else None
                max_el = float(np.max(valid_vals)) if len(valid_vals) > 0 else None
                mean_el = float(np.mean(valid_vals)) if len(valid_vals) > 0 else None

                is_synthetic = "synthetic" in path.name.lower() or "test" in path.name.lower()

                return DemMetadata(
                    dataset_name=path.stem,
                    data_source="NASA LOLA PDS / LDEM Product" if not is_synthetic else "Synthetic Test Fixture",
                    file_path=str(path),
                    width_pixels=width,
                    height_pixels=height,
                    pixel_size_x_m=res_x,
                    pixel_size_y_m=res_y,
                    crs_repr=crs_str,
                    bounds_projected=bounds,
                    elevation_units=cls.DEFAULT_UNITS,
                    reference_datum=cls.DEFAULT_LUNAR_DATUM,
                    nodata_value=nodata,
                    valid_data_percent=valid_pct,
                    min_elevation_m=min_el,
                    max_elevation_m=max_el,
                    mean_elevation_m=mean_el,
                    is_synthetic=is_synthetic
                )
        except RasterioIOError as e:
            raise DemFormatError(f"Failed to read GeoTIFF header for {path}: {str(e)}")

    @classmethod
    def _load_geotiff(
        cls,
        path: Path,
        window_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> DemRaster:
        if not HAS_RASTERIO:
            raise DemError("Rasterio is required to load GeoTIFF files.")

        try:
            with rasterio.open(path) as src:
                transform = src.transform

                if window_bounds is not None:
                    left, bottom, right, top = window_bounds
                    # Intersect requested window with raster bounds
                    r_left, r_bottom, r_right, r_top = src.bounds
                    if right < r_left or left > r_right or top < r_bottom or bottom > r_top:
                        raise DemWindowError(
                            f"Requested window {window_bounds} is completely outside raster bounds {src.bounds}."
                        )

                    window = rasterio.windows.from_bounds(left, bottom, right, top, transform=transform)
                    elevation_data = src.read(1, window=window)
                    window_transform = rasterio.windows.transform(window, transform)
                    raw_b = rasterio.windows.bounds(window, transform)
                    actual_bounds = (float(raw_b[0]), float(raw_b[1]), float(raw_b[2]), float(raw_b[3]))
                    actual_width = elevation_data.shape[1]
                    actual_height = elevation_data.shape[0]
                else:
                    elevation_data = src.read(1)
                    window_transform = transform
                    actual_bounds = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
                    actual_width = src.width
                    actual_height = src.height

                nodata = src.nodata
                if nodata is not None:
                    valid_mask = (elevation_data != nodata) & ~np.isnan(elevation_data)
                else:
                    valid_mask = ~np.isnan(elevation_data)

                valid_pct = float(np.count_nonzero(valid_mask) / elevation_data.size * 100.0)
                valid_vals = elevation_data[valid_mask]
                min_el = float(np.min(valid_vals)) if len(valid_vals) > 0 else None
                max_el = float(np.max(valid_vals)) if len(valid_vals) > 0 else None
                mean_el = float(np.mean(valid_vals)) if len(valid_vals) > 0 else None

                is_synthetic = "synthetic" in path.name.lower() or "test" in path.name.lower()

                meta = DemMetadata(
                    dataset_name=path.stem,
                    data_source="NASA LOLA PDS / LDEM Product" if not is_synthetic else "Synthetic Test Fixture",
                    file_path=str(path),
                    width_pixels=actual_width,
                    height_pixels=actual_height,
                    pixel_size_x_m=abs(src.res[0]),
                    pixel_size_y_m=abs(src.res[1]),
                    crs_repr=str(src.crs) if src.crs else "UNKNOWN",
                    bounds_projected=actual_bounds,
                    elevation_units=cls.DEFAULT_UNITS,
                    reference_datum=cls.DEFAULT_LUNAR_DATUM,
                    nodata_value=nodata,
                    valid_data_percent=valid_pct,
                    min_elevation_m=min_el,
                    max_elevation_m=max_el,
                    mean_elevation_m=mean_el,
                    is_synthetic=is_synthetic
                )

                return DemRaster(elevation_array=elevation_data, metadata=meta, transform=window_transform)
        except RasterioIOError as e:
            raise DemFormatError(f"Corrupt or unreadable GeoTIFF file {path}: {str(e)}")

    # -------------------------------------------------------------------------
    # NumPy Driver
    # -------------------------------------------------------------------------

    @classmethod
    def _inspect_numpy(cls, path: Path) -> DemMetadata:
        try:
            if path.suffix.lower() == ".npz":
                with np.load(path) as data:
                    if "elevation" not in data:
                        raise DemFormatError("NumPy archive must contain an 'elevation' array key.")
                    arr = data["elevation"]
                    bounds = tuple(data["bounds"]) if "bounds" in data else (-50000.0, -50000.0, 50000.0, 50000.0)
                    res_x = float(data["pixel_size_x_m"]) if "pixel_size_x_m" in data else 100.0
                    res_y = float(data["pixel_size_y_m"]) if "pixel_size_y_m" in data else 100.0
                    crs = str(data["crs"]) if "crs" in data else "Polar Stereographic (Moon R=1737.4km)"
                    nodata = float(data["nodata"]) if "nodata" in data else -9999.0
            else:
                arr = np.load(path)
                bounds = (-50000.0, -50000.0, 50000.0, 50000.0)
                res_x, res_y = 100.0, 100.0
                crs = "Polar Stereographic (Moon R=1737.4km)"
                nodata = -9999.0

            valid_mask = (arr != nodata) & ~np.isnan(arr)
            valid_pct = float(np.count_nonzero(valid_mask) / arr.size * 100.0)
            valid_vals = arr[valid_mask]

            return DemMetadata(
                dataset_name=path.stem,
                data_source="NumPy Gridded LOLA Array",
                file_path=str(path),
                width_pixels=arr.shape[1],
                height_pixels=arr.shape[0],
                pixel_size_x_m=res_x,
                pixel_size_y_m=res_y,
                crs_repr=crs,
                bounds_projected=bounds,
                elevation_units=cls.DEFAULT_UNITS,
                reference_datum=cls.DEFAULT_LUNAR_DATUM,
                nodata_value=nodata,
                valid_data_percent=valid_pct,
                min_elevation_m=float(np.min(valid_vals)) if len(valid_vals) > 0 else None,
                max_elevation_m=float(np.max(valid_vals)) if len(valid_vals) > 0 else None,
                mean_elevation_m=float(np.mean(valid_vals)) if len(valid_vals) > 0 else None,
                is_synthetic="synthetic" in path.name.lower() or "test" in path.name.lower()
            )
        except Exception as e:
            raise DemFormatError(f"Unreadable NumPy DEM {path}: {str(e)}")

    @classmethod
    def _load_numpy(
        cls,
        path: Path,
        window_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> DemRaster:
        meta = cls._inspect_numpy(path)
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                arr = data["elevation"]
        else:
            arr = np.load(path)

        if window_bounds is not None:
            left, bottom, right, top = window_bounds
            r_left, r_bottom, r_right, r_top = meta.bounds_projected
            if right < r_left or left > r_right or top < r_bottom or bottom > r_top:
                raise DemWindowError("Requested bounding box lies outside raster extents.")

            px_x, px_y = meta.pixel_size_x_m, meta.pixel_size_y_m
            c_start = max(0, int((left - r_left) / px_x))
            c_end = min(meta.width_pixels, int((right - r_left) / px_x))
            r_start = max(0, int((r_top - top) / px_y))
            r_end = min(meta.height_pixels, int((r_top - bottom) / px_y))

            arr = arr[r_start:r_end, c_start:c_end]
            meta.width_pixels = arr.shape[1]
            meta.height_pixels = arr.shape[0]
            meta.bounds_projected = (
                r_left + c_start * px_x,
                r_top - r_end * px_y,
                r_left + c_end * px_x,
                r_top - r_start * px_y
            )

        return DemRaster(elevation_array=arr, metadata=meta)


def generate_dem_report(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Public utility to generate a formatted dictionary report of a DEM dataset.
    """
    meta = DemLoader.inspect_metadata(file_path)
    return meta.to_dict()
