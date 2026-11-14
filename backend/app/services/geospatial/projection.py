"""
LUNA NAVIGATOR — Lunar Polar Stereographic Projection Engine (Member 3 Ownership)
Implements conformal polar stereographic projection conforming to NASA LOLA PDS standards
(LRO LOLA GDR South Pole products, e.g. LDEM_80S_100M / LDEM_875S_20M).

Reference:
- Snyder, J. P. (1987). Map Projections: A Working Manual (USGS Professional Paper 1395).
- NASA PDS LOLA Experiment Software Interface Specification (SIS).
- Standard lunar sphere datum: R_moon = 1,737,400.0 meters.
"""

import math
from typing import Tuple
from app.core.constants import MOON_MEAN_RADIUS_M


class LunarPolarStereographic:
    """
    Polar Stereographic projection for Lunar South Pole.
    
    Parameters:
        central_meridian_deg: Longitude of central meridian (default: 0.0°).
        standard_parallel_deg: Latitude of true scale (default: -90.0° South Pole).
        radius_m: Lunar reference sphere radius in meters (default: 1,737,400 m).
    """

    def __init__(
        self,
        central_meridian_deg: float = 0.0,
        standard_parallel_deg: float = -90.0,
        radius_m: float = MOON_MEAN_RADIUS_M
    ):
        self.central_meridian_deg = central_meridian_deg
        self.central_meridian_rad = math.radians(central_meridian_deg)
        self.standard_parallel_deg = standard_parallel_deg
        self.radius_m = radius_m

    def forward(self, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
        """
        Convert Selenographic (Lat, Lon in degrees) to Polar Stereographic planar coordinates (X, Y in meters).
        
        Args:
            lat_deg: Selenographic latitude in degrees (negative in southern hemisphere, e.g. -89.78°).
            lon_deg: Selenographic longitude in degrees [0, 360) or [-180, +180].
            
        Returns:
            (x_m, y_m): Projected easting and northing in meters.
        """
        if lat_deg > 0.0:
            # Polar stereographic south is invalid for northern hemisphere
            raise ValueError(f"Latitude {lat_deg}° is not in the lunar southern polar domain.")

        # Colatitude in southern hemisphere: c = 90° + lat_deg (in degrees)
        colat_rad = math.radians(90.0 + lat_deg)
        lam_rad = math.radians(lon_deg) - self.central_meridian_rad

        # Radial distance from pole on projection plane:
        # rho = 2 * R * tan(colat / 2)
        rho = 2.0 * self.radius_m * math.tan(colat_rad / 2.0)

        # Standard USGS/NASA planetary orientation:
        # X is along 90° East meridian, Y is along 0° meridian
        x_m = rho * math.sin(lam_rad)
        y_m = -rho * math.cos(lam_rad)

        return (x_m, y_m)

    def inverse(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Convert Polar Stereographic planar coordinates (X, Y in meters) to Selenographic (Lat, Lon in degrees).
        
        Args:
            x_m: Projected easting in meters.
            y_m: Projected northing in meters.
            
        Returns:
            (lat_deg, lon_deg): Latitude and longitude in degrees.
        """
        rho = math.sqrt(x_m**2 + y_m**2)

        if rho == 0.0:
            return (-90.0, 0.0)

        # Colatitude = 2 * atan(rho / (2 * R))
        colat_rad = 2.0 * math.atan(rho / (2.0 * self.radius_m))
        lat_rad = colat_rad - (math.pi / 2.0)
        lat_deg = math.degrees(lat_rad)

        # Longitude from central meridian
        # x = rho * sin(lam), y = -rho * cos(lam) ==> lam = atan2(x, -y)
        lam_rad = math.atan2(x_m, -y_m) + self.central_meridian_rad
        lon_deg = math.degrees(lam_rad)

        # Normalize longitude to [0, 360)
        lon_deg = lon_deg % 360.0

        return (lat_deg, lon_deg)


# Default South Pole stereographic instance conforming to NASA LOLA standards
lola_polar_projection = LunarPolarStereographic()
