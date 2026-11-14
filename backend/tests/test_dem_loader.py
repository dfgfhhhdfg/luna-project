"""
Unit and integration tests for Lunar DEM Loader (dem_loader.py).
Member 3: GIS & Scientific Computing.
"""

import os
import math
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.services.geospatial.dem_loader import (
    DemLoader,
    DemRaster,
    DemMetadata,
    DemFileNotFoundError,
    DemFormatError,
    DemWindowError,
    generate_dem_report,
)

# Canonical synthetic test DEM path
SYNTHETIC_DEM_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "dem", "synthetic_test_dem.tif")
)


def _generate_synthetic_dem_if_missing():
    """Generates synthetic test DEM on the fly if not present on disk."""
    if os.path.exists(SYNTHETIC_DEM_PATH):
        return

    os.makedirs(os.path.dirname(SYNTHETIC_DEM_PATH), exist_ok=True)
    width, height, res = 100, 100, 100.0
    left, top = -5000.0, 5000.0
    transform = from_origin(left, top, res, res)

    x = np.linspace(-4950, 4950, width)
    y = np.linspace(4950, -4950, height)
    xx, yy = np.meshgrid(x, y)
    r = np.hypot(xx, yy)

    elev = -500.0 + 1500.0 * np.exp(-((r - 3000.0) / 800.0)**2) - 2000.0 * (r < 2000.0)
    elev = elev.astype(np.float32)
    elev[:5, :5] = -9999.0  # nodata block

    crs_proj4 = '+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs'
    with rasterio.open(
        SYNTHETIC_DEM_PATH,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=elev.dtype,
        crs=crs_proj4,
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(elev, 1)
        dst.update_tags(
            DATA_ORIGIN='SYNTHETIC_TEST_FIXTURE_NOT_REAL_NASA_DATA',
            DESCRIPTION='Synthetic Lunar South Pole Test DEM for automated unit testing (Member 3)',
            IS_SYNTHETIC='TRUE'
        )


@pytest.fixture(scope="session", autouse=True)
def setup_test_fixtures():
    _generate_synthetic_dem_if_missing()


class TestDemLoaderGeoTIFF:
    """Test GeoTIFF reading, validation, and metadata extraction."""

    def test_inspect_metadata_synthetic_dem(self):
        """Verify inspect_metadata correctly reads GeoTIFF header and metrics."""
        assert os.path.exists(SYNTHETIC_DEM_PATH), f"Synthetic test DEM missing at {SYNTHETIC_DEM_PATH}"

        meta = DemLoader.inspect_metadata(SYNTHETIC_DEM_PATH)
        assert isinstance(meta, DemMetadata)
        assert meta.dataset_name == "synthetic_test_dem"
        assert meta.is_synthetic is True
        assert meta.width_pixels == 100
        assert meta.height_pixels == 100
        assert meta.pixel_size_x_m == pytest.approx(100.0, abs=1e-2)
        assert meta.pixel_size_y_m == pytest.approx(100.0, abs=1e-2)
        assert meta.nodata_value == -9999.0
        assert meta.elevation_units == "meters"
        assert "Lunar Sphere" in meta.reference_datum
        # 25 nodata cells out of 10,000 = 99.75% valid
        assert meta.valid_data_percent == pytest.approx(99.75, abs=0.1)
        assert meta.min_elevation_m == pytest.approx(-2500.0, abs=1.0)
        assert meta.max_elevation_m == pytest.approx(1000.0, abs=10.0)

    def test_load_dem_full_raster(self):
        """Verify full raster loading, spatial querying, and coordinate lookups."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)
        assert isinstance(raster, DemRaster)
        assert raster.shape == (100, 100)

        # Center of crater floor
        el_center = raster.get_elevation_at_projected(0.0, 0.0)
        assert el_center == pytest.approx(-2500.0, abs=1.0)

        # Crater rim (approx 3000m radius)
        el_rim = raster.get_elevation_at_projected(3000.0, 0.0)
        assert el_rim > 900.0

        # Pixel-based access
        el_px = raster.get_elevation_at_pixel(50, 50)
        assert el_px == pytest.approx(-2500.0, abs=1.0)

    def test_nodata_and_missing_values(self):
        """Verify that nodata cells and out-of-bounds queries return NaN."""
        raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH)

        # Top-left corner has nodata (-9999.0)
        nodata_val = raster.get_elevation_at_pixel(0, 0)
        assert math.isnan(nodata_val)

        # Query projected coordinate in nodata area
        val_proj_nodata = raster.get_elevation_at_projected(-4950.0, 4950.0)
        assert math.isnan(val_proj_nodata)

        # Query outside raster extent returns NaN
        val_outside = raster.get_elevation_at_projected(15000.0, 15000.0)
        assert math.isnan(val_outside)

    def test_windowed_raster_loading(self):
        """Verify windowed loading loads only sub-region and maintains coordinate accuracy."""
        # Window of 4000m x 4000m = 40x40 pixels
        window_bounds = (-2000.0, -2000.0, 2000.0, 2000.0)
        sub_raster = DemLoader.load_dem(SYNTHETIC_DEM_PATH, window_bounds=window_bounds)

        assert sub_raster.shape == (40, 40)
        assert sub_raster.metadata.width_pixels == 40
        assert sub_raster.metadata.height_pixels == 40
        assert sub_raster.metadata.bounds_projected == pytest.approx(window_bounds, abs=1.0)

        # Elevation at center should match full raster
        el_center_sub = sub_raster.get_elevation_at_projected(0.0, 0.0)
        assert el_center_sub == pytest.approx(-2500.0, abs=1.0)

    def test_window_out_of_bounds_raises_error(self):
        """Verify requesting a window completely outside raster raises DemWindowError."""
        outside_window = (20000.0, 20000.0, 30000.0, 30000.0)
        with pytest.raises(DemWindowError):
            DemLoader.load_dem(SYNTHETIC_DEM_PATH, window_bounds=outside_window)

    def test_file_not_found_handling(self):
        """Verify non-existent file raises DemFileNotFoundError."""
        with pytest.raises(DemFileNotFoundError):
            DemLoader.inspect_metadata("non_existent_dem_path_12345.tif")

    def test_unsupported_format_handling(self):
        """Verify unsupported file extension raises DemFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            tmp.write(b"invalid data")
            tmp.flush()
            with pytest.raises(DemFormatError):
                DemLoader.inspect_metadata(tmp.name)

    def test_corrupted_file_handling(self):
        """Verify corrupt file with .tif extension raises DemFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(b"this is corrupt binary junk not a tiff")
            tmp_path = tmp.name

        try:
            with pytest.raises(DemFormatError):
                DemLoader.load_dem(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestDemLoaderSpecialCases:
    """Test handling of missing CRS and NumPy archives."""

    def test_missing_crs_geotiff(self):
        """Verify loader handles GeoTIFF missing CRS gracefully without crashing."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            data = np.full((20, 20), 100.0, dtype=np.float32)
            transform = from_origin(0.0, 0.0, 50.0, 50.0)
            with rasterio.open(
                tmp_path,
                "w",
                driver="GTiff",
                height=20,
                width=20,
                count=1,
                dtype=data.dtype,
                crs=None,  # Intentionally omit CRS
                transform=transform,
                nodata=-9999.0
            ) as dst:
                dst.write(data, 1)

            meta = DemLoader.inspect_metadata(tmp_path)
            assert meta.crs_repr == "UNKNOWN" or meta.crs_repr is None
            assert meta.width_pixels == 20
            assert meta.height_pixels == 20

            raster = DemLoader.load_dem(tmp_path)
            assert raster.get_elevation_at_pixel(10, 10) == 100.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_numpy_npz_archive_loading(self):
        """Verify loader supports NumPy .npz gridded rasters."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            elevation = np.linspace(-1000.0, 1000.0, 1600, dtype=np.float32).reshape((40, 40))
            elevation[0, 0] = -9999.0  # nodata
            np.savez(
                tmp_path,
                elevation=elevation,
                bounds=np.array([-2000.0, -2000.0, 2000.0, 2000.0]),
                pixel_size_x_m=100.0,
                pixel_size_y_m=100.0,
                crs="Polar Stereographic",
                nodata=-9999.0
            )

            meta = DemLoader.inspect_metadata(tmp_path)
            assert meta.width_pixels == 40
            assert meta.height_pixels == 40
            assert meta.nodata_value == -9999.0

            raster = DemLoader.load_dem(tmp_path)
            assert raster.shape == (40, 40)
            assert math.isnan(raster.get_elevation_at_pixel(0, 0))
            assert raster.get_elevation_at_pixel(20, 20) == pytest.approx(0.0, abs=50.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestDemReportUtility:
    """Test generate_dem_report function output dictionary."""

    def test_generate_dem_report_output_schema(self):
        """Verify report returns complete, standardized metadata dictionary."""
        report = generate_dem_report(SYNTHETIC_DEM_PATH)
        assert isinstance(report, dict)
        required_keys = [
            "dataset_name",
            "data_source",
            "file_path",
            "width_pixels",
            "height_pixels",
            "pixel_size_x_m",
            "pixel_size_y_m",
            "crs_repr",
            "bounds_projected",
            "elevation_units",
            "reference_datum",
            "nodata_value",
            "valid_data_percent",
            "min_elevation_m",
            "max_elevation_m",
            "mean_elevation_m",
            "is_synthetic",
        ]
        for key in required_keys:
            assert key in report, f"Missing key '{key}' in DEM report."

        assert report["dataset_name"] == "synthetic_test_dem"
        assert report["is_synthetic"] is True
        assert report["width_pixels"] == 100
        assert report["height_pixels"] == 100
        assert report["elevation_units"] == "meters"
