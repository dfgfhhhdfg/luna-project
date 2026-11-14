# Lunar Topography, Slope & Roughness Analysis Documentation
**Subsystem**: Member 3 — GIS & Scientific Computing  
**Project**: LUNA — Lunar Landing Site Navigator  
**Datum**: Mean Lunar Sphere $R_{moon} = 1,737,400.0\text{ m}$ ($1737.4\text{ km}$), flattening $f = 0$  
**Projection**: Lunar South Pole Polar Stereographic ($\phi_0 = -90^\circ, \lambda_0 = 0^\circ$)

---

## 1. Topographic Slope Calculation Method

Topographic slope is computed from digital elevation models using an adapted **Horn's 8-Neighbor Weighted Central Difference Algorithm** that explicitly supports unequal pixel resolutions ($\Delta x \ne \Delta y$):

Given a $3 \times 3$ grid cell neighborhood centered on $e_{r, c}$:
```
e_nw   e_n   e_ne
e_w    e_c   e_e
e_sw   e_s   e_se
```

The directional elevation gradients in meters per meter:
$$\frac{\partial z}{\partial x} = \frac{(e_{ne} + 2 e_e + e_{se}) - (e_{nw} + 2 e_w + e_{sw})}{8 \cdot \Delta x}$$
$$\frac{\partial z}{\partial y} = \frac{(e_{nw} + 2 e_n + e_{ne}) - (e_{sw} + 2 e_s + e_{se})}{8 \cdot \Delta y}$$

*Note on Northing convention*: In raster matrices, row index $r$ increases downward (South). Therefore, northern row $r-1$ has greater Northing coordinate $y$ than southern row $r+1$.

The overall slope angle $\theta$ in degrees:
$$\text{gradient} = \sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}$$
$$\theta_{\text{deg}} = \arctan(\text{gradient}) \times \frac{180^\circ}{\pi} \in [0.0^\circ, 90.0^\circ]$$

### Edge and Nodata Margin Handling
- **Full Horn Stencil**: Applied when all 8 neighbors and center cell are valid non-nodata cells.
- **Partial / Boundary Fallback**: When boundary pixels or cells adjacent to nodata lack a complete $3 \times 3$ neighborhood, symmetric 2-point central differences ($(e_e - e_w) / 2\Delta x$) or one-sided forward/backward differences ($(e_e - e_c) / \Delta x$) are applied.
- **Nodata cells**: Strictly output as `np.nan`.
- **False Slope Spikes Prevented**: By masking nodata before gradient evaluation, the algorithm prevents catastrophic $\sim 89^\circ$ slope artifacts along nodata margins.

### Coordinate Reference System (CRS) & Affine Transform Validation
To ensure mathematical and physical validity:
1. **Geographic CRS Rejection**: Datasets in angular degrees (e.g. Equirectangular or `GEOGCS["GCS_Moon_2000"]`) are rejected with `DemCoordinateError`. Slope calculation requires horizontal distances in meters.
2. **Linear Units Verification**: Projected coordinate systems must use meters (`+units=m`). Datasets in feet or survey feet are rejected.
3. **Rotation and Shear Verification**: The Horn 8-neighbor stencil assumes grid columns and rows align with Easting and Northing axes. Datasets with rotated or sheared transforms ($b \ne 0$ or $d \ne 0$) are rejected with `DemCoordinateError`.

---

## 2. Terrain Roughness Metrics & Physical Baseline Scale

Two complementary roughness metrics are supported over a configurable $W \times W$ moving window kernel:

### Metric A: Local Elevation Standard Deviation ($\sigma_z$) — Default
$$\sigma_z = \sqrt{\frac{1}{N_{\text{valid}} - 1} \sum_{i=1}^{N_{\text{valid}}} (z_i - \bar{z})^2}$$
- **Units**: Meters.
- **Physical Interpretation**: Measures high-frequency surface topographic variation and micro-relief rugosity. Completely flat terrain produces $\sigma_z = 0.0\text{ m}$. Smooth mare and highland plateaus typically exhibit $\sigma_z < 5\text{ m}$, while rugged crater rim walls exhibit $\sigma_z > 20\text{ m}$.
- **Minimum Valid Count**: If the number of valid cells in a window is $< 50\%$ of $W^2$, the output cell is assigned `np.nan` to avoid unrepresentative sampling.

### Metric B: Local Relief ($\Delta z$)
$$\Delta z = \max_{W}(z) - \min_{W}(z)$$
- **Units**: Meters.
- **Physical Interpretation**: Peak-to-valley height disparity within the moving footprint.

### Physical Baseline Scale Reporting
Because lunar topography exhibits fractal self-affinity (Hurst exponent $H \approx 0.7\text{–}0.9$), surface roughness is scale-dependent. Reporting a roughness value without its physical baseline scale is scientifically incomplete.
Outputs explicitly report:
- `window_size_pixels`: Kernel dimension in pixels ($W$).
- `pixel_size_x_m`, `pixel_size_y_m`: Horizontal resolution in Easting and Northing.
- `physical_baseline_x_m`: $W \cdot \Delta x$ (meters).
- `physical_baseline_y_m`: $W \cdot \Delta y$ (meters).
- `physical_baseline_m`: Geometric mean $\sqrt{\text{baseline}_x \cdot \text{baseline}_y}$.
- `baseline_convention`: Clear explanation of non-square pixel baselines.

---

## 3. Elevation Interpolation & Terrain Profile Analysis

### Bilinear vs Nearest-Neighbor Interpolation
The `dem.get_elevation_at_projected(x_m, y_m, method="bilinear")` method supports continuous surface sampling:
- **Bilinear (Default)**: In pixel-is-area convention, pixel centers are located at $(c + 0.5, r + 0.5)$. Continuous pixel coordinates $(u, v)$ are derived via affine inversion $\sim \text{transform} \cdot (x, y)$. The four nearest cell centers are identified, and heights are interpolated:
  $$z = (1 - \alpha)(1 - \beta) z_{00} + \alpha(1 - \beta) z_{10} + (1 - \alpha)\beta z_{01} + \alpha\beta z_{11}$$
- **Strict Nodata Propagation**: If ANY of the 4 surrounding interpolation corners is nodata or `NaN`, the function strictly returns `NaN`. No nearest-neighbor values are silently substituted, preventing nodata bleeding into valid terrain profiles.
- **Nearest-Neighbor**: Available via `method="nearest"` for discrete cell-center inspection.

### Terrain Profile Transects
The `calculate_terrain_profile()` function extracts elevation transects between any two projected planar coordinates $(x_1, y_1)$ and $(x_2, y_2)$:
- **Path Distance**: Sampled at configurable intervals $\Delta s$ (e.g. $100\text{ m}$) from $0.0\text{ m}$ to the Euclidean path length $L = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$.
- **Zero Silent Interpolation**: Missing data or nodata cells along the path are represented as `None` / `NaN`.
- **Gap Detection**: If consecutive unmapped cells exceed `max_gap_allowed_m` (default $500\text{ m}$), an explicit descriptive warning is raised in the output.

---

## 4. Candidate Landing Area Assessment (Member 4 Integration)

The `assess_landing_area()` function evaluates a candidate circular landing zone of radius $R$ meters (e.g. $1000\text{ m}$ for Artemis-class landers):

1. Slices a buffered sub-window from the DEM raster to avoid loading unnecessary regions into RAM.
2. Computes slope and roughness grids across the sub-window.
3. Discretizes the circular footprint: cell centers satisfying $\sqrt{(x - x_c)^2 + (y - y_c)^2} \le R$.
4. **Coverage Normalization**:
   The nominal footprint area is $A_{\text{nominal}} = \pi R^2$, representing $N_{\text{nominal}} = \text{round}\left(\frac{\pi R^2}{\Delta x \Delta y}\right)$ grid cells.
   If the circle overlaps the boundary of the DEM:
   - $N_{\text{in\_bounds}}$ is the count of cells inside the DEM.
   - $N_{\text{outside}} = \max(0, N_{\text{nominal}} - N_{\text{in\_bounds}})$ cells lie outside the mapped DEM.
   - $N_{\text{footprint}} = \max(N_{\text{nominal}}, N_{\text{in\_bounds}})$.
   - Valid data coverage is strictly computed as:
     $$\text{valid\_data\_percent} = \min\left(100.0, \frac{N_{\text{valid}}}{N_{\text{footprint}}} \times 100.0\right)$$
   This ensures that a landing site positioned near the edge of available DEM coverage correctly flags that unavailable cells lie outside the boundary, guaranteeing coverage never falsely reports 100% when part of the footprint is unmapped.

### How Member 4 Consumes Output
Member 4's Multi-Criteria Decision Analysis (MCDA) recommendation engine consumes:
```python
assessment = assess_landing_area(dem, center_projected=(cx, cy), radius_m=1000.0)

# 1. Slope Constraint Checking:
is_compliant = (
    assessment["slope_stats"]["max_slope_deg"] <= constraints.max_acceptable_slope_degrees
    and assessment["valid_data_percent"] >= 90.0
)

# 2. Raw Metrics for MCDA Scoring:
raw_metrics = {
    "slope_mean_deg": assessment["slope_stats"]["mean_slope_deg"],
    "slope_safe_pct": assessment["slope_stats"]["threshold_coverage_pct"]["lte_12deg"],
    "roughness_mean_m": assessment["roughness_stats"]["mean_roughness_m"],
    "elevation_m": assessment["elevation_stats"]["mean_elevation_m"]
}
```

---

## 5. Scientific Assumptions & Known Limitations

1. **Polar Stereographic Scale Factor**: The projection uses conformal Polar Stereographic with scale factor $k = \frac{2}{1 + \sin(-\phi)}$.
   - At $85^\circ\text{S}$, distortion is $< 0.2\%$.
   - At $80^\circ\text{S}$, distortion is $< 0.8\%$.
   - For landing site analysis ($< 10\text{ km}$), distortion error is negligible compared to DEM vertical accuracy ($\pm 1\text{–}2\text{ m}$).
2. **Sub-Pixel Hazards**: LOLA 20m/100m gridded DEMs capture regional slopes and crater morphology, but do not resolve sub-pixel meter-scale boulders ($< 20\text{ m}$).
3. **Engineering Disclaimer**: Topographic slope and roughness analysis does not constitute a complete landing safety certification. Geotechnical regolith bearing capacity and autonomous optical hazard detection systems are required for real flight operations.

---

## 6. Example Python Function Usage

```python
from app.services.geospatial import DemLoader, calculate_slope, assess_landing_area

# 1. Load DEM (GeoTIFF or NumPy)
dem = DemLoader.load_dem("datasets/dem/synthetic_test_dem.tif")

# 2. Compute 2D slope grid (with automatic CRS validation)
slope_grid = dem.calculate_slope()
print(f"Max slope: {np.nanmax(slope_grid):.2f}°")

# 3. Assess candidate landing zone (1 km radius around South Pole)
assessment = dem.assess_landing_area(
    center_projected=(0.0, 0.0),
    radius_m=1000.0,
    slope_thresholds=[5.0, 10.0, 12.0, 15.0]
)

print(f"Valid Coverage: {assessment['valid_data_percent']}%")
print(f"Mean Slope: {assessment['slope_stats']['mean_slope_deg']}°")
print(f"Area <= 12° Slope: {assessment['slope_stats']['threshold_coverage_pct']['lte_12deg']}%")
print(f"Mean Roughness: {assessment['roughness_stats']['mean_roughness_m']} m")
print(f"Physical Baseline: {assessment['roughness_stats']['physical_baseline_m']} m")
```

---

## 7. Lunar Horizon Ray Tracing & Line-of-Sight Analysis (Phase 3)

The `horizon_service.py` module provides scientific computing for line-of-sight obstruction and 360° terrain horizon profiling from NASA LOLA digital elevation models.

### 1. Lunar Spherical Curvature Drop (Parabolic Approximation)
Over lunar surface ground distances $s$, the local horizontal tangent plane at the observer drops away from the curved spherical lunar surface. The elevation correction $\Delta h(s)$ is computed using the **second-order parabolic geodesy approximation**:
$$\Delta h(s) = \frac{s^2}{2 R_{\text{moon}}}$$
where $R_{\text{moon}} = 1,737,400.0\text{ m}$ ($1737.4\text{ km}$, NASA LOLA standard datum).

#### Comparison against Exact Spherical Radial Secant Drop
The exact radial drop along the sphere normal is $\Delta h_{\text{radial}}(s) = R_{\text{moon}} \left(\sec\left(\frac{s}{R_{\text{moon}}}\right) - 1\right)$. The second-order Taylor expansion yields $\Delta h \approx \frac{s^2}{2 R} + \frac{5 s^4}{24 R^3}$.

Analytical verification of parabolic approximation vs. exact spherical geometry:
- **$s = 10\text{ km}$**: Parabolic = $28.7786\text{ m}$, Exact $\sec$-drop = $28.7790\text{ m}$ (Discrepancy: $0.4\text{ mm}$, negligible).
- **$s = 30\text{ km}$**: Parabolic = $259.0077\text{ m}$, Exact $\sec$-drop = $259.0399\text{ m}$ (Discrepancy: $32.2\text{ mm} \approx 3.2\text{ cm}$).
- **$s = 60\text{ km}$**: Parabolic = $1036.0309\text{ m}$, Exact $\sec$-drop = $1036.5459\text{ m}$ (Discrepancy: $51.5\text{ cm} \approx 0.52\text{ m}$).
- **$s = 100\text{ km}$**: Parabolic = $2877.8635\text{ m}$, Exact $\sec$-drop = $2881.8413\text{ m}$ (Discrepancy: $3.978\text{ m} \approx 3.98\text{ m}$).

*Scientific note*: For landing site terrain analyses ($s \le 30\text{ km}$ to $60\text{ km}$), the parabolic approximation error ($< 0.52\text{ m}$) is well within LOLA DEM vertical uncertainty ($\pm 1\text{–}2\text{ m}$). At long ranges ($s > 100\text{ km}$), exact spherical trigonometry should be considered.

### 2. Ray Trajectory Models & Geodetic Alignment
Two complementary trajectory models are supported:
- **Spherical Geodetic (Default in `mode="auto"` and `mode="spherical"`)**:
  Computes true great-circle waypoints on the lunar sphere using spherical forward geodesy and projects each waypoint to Lunar South Polar Stereographic $(x, y)$.
  - **True North Alignment**: Azimuth $0^\circ$ is defined as Lunar True North (pointing towards the equator along the observer's local meridian), agreeing 100% with Member 2's ephemeris convention.
  - **Polar Singularity**: When observer is at the South Pole ($\phi = -90^\circ$), moving in azimuth $\theta$ corresponds to advancing North along meridian $\lambda = \theta$.
- **Planar Grid (`mode="projected"`)**:
  Marches directly in projected Cartesian grid coordinates:
  $$x(s) = x_0 + s \cdot \sin(\theta_{\text{grid}})$$
  $$y(s) = y_0 + s \cdot \cos(\theta_{\text{grid}})$$
  *Note on Grid Azimuth vs. True North*: In Lunar South Pole Polar Stereographic, the prime meridian $\lambda = 0^\circ$ lies along the $-Y$ axis, and $\lambda = 180^\circ$ lies along $+Y$. Therefore, Cartesian Grid North ($+Y$) points towards $\lambda = 180^\circ$, which differs from geodetic True North by the meridian convergence angle. `mode="projected"` is provided for synthetic Cartesian grids and controlled benchmarks.

### 3. Topographic Elevation Angle Formulation
For an observer at ground elevation $z_0$ with instrument mast height $h_{\text{obs}}$, total observer elevation is:
$$z_{\text{obs}} = z_0 + h_{\text{obs}}$$

At ground distance $s > 0$ along azimuth $\theta$, the terrain elevation is $z(s)$.
The apparent elevation relative to the observer's horizontal tangent plane is:
$$z_{\text{apparent}}(s) = z(s) - \Delta h(s) = z(s) - \frac{s^2}{2 R_{\text{moon}}}$$

The topographic elevation angle $\alpha(s)$ in degrees above the horizontal tangent plane is:
$$\alpha(s) = \arctan\left(\frac{z_{\text{apparent}}(s) - z_{\text{obs}}}{s}\right) \times \frac{180^\circ}{\pi}$$

### 4. Tri-State Visibility Evaluation
When evaluating line-of-sight to an extraterrestrial or surface target at azimuth $\theta_{\text{target}}$ and elevation $\gamma_{\text{target}}$:
- **`obstructed`**: Topography along the ray reaches or exceeds the target elevation angle ($\alpha(s) \ge \gamma_{\text{target}}$). Topographic occultation is physically confirmed. Reports `obstructing_distance_m`, `obstructing_point`, and negative elevation margin.
- **`clear`**: The ray traversed the complete requested search distance $s_{\text{max}}$ entirely within valid DEM coverage, with all evaluated terrain strictly below the target elevation ($\alpha(s) < \gamma_{\text{target}}$ for all $s \le s_{\text{max}}$). Reports positive elevation clearance margin.
- **`unknown`**: No obstruction was found in the evaluated terrain, but the ray was truncated because it exited the DEM boundary, encountered NoData cells, or the observer itself was outside coverage. Because unmapped topography beyond the boundary could obstruct the target, clearance cannot be assumed.

### 5. 360-Degree Periodic Horizon Profiling
The `calculate_horizon_profile()` function casts radial rays across all azimuths $[0^\circ, 360^\circ)$ with configurable step resolution $\Delta\theta$ (e.g. $1^\circ$ or $5^\circ$).
- Maximum horizon elevation angle for each azimuth: $H(\theta) = \max_s \alpha(s, \theta)$.
- Summary statistics: min, max, and mean horizon angles, and azimuths of peak terrain obstacles.
- Continuous periodic lookup: `get_horizon_elevation_at_azimuth()` provides smooth periodic linear interpolation across $360^\circ \equiv 0^\circ$ (Lunar North).

---

## 8. Phase 3 Python API & REST Endpoint Examples

### Python Service Usage
```python
from app.services.geospatial import DemLoader

# 1. Load active DEM
dem = DemLoader.load_dem("datasets/dem/synthetic_test_dem.tif")

# 2. Check Line-of-Sight to Sun / Earth (tri-state outcome)
los = dem.check_line_of_sight(
    target_azimuth_deg=45.0,
    target_elevation_deg=1.5,
    observer_coord=(0.0, 0.0),
    max_distance_m=30000.0,
    radial_step_m=100.0,
    observer_height_m=2.0
)
print("Visibility State:", los["visibility_state"])  # 'obstructed', 'clear', or 'unknown'
print("Elevation Margin:", los["elevation_margin_deg"], "degrees")
if los["is_obstructed"]:
    print("Obstructing Distance:", los["obstructing_distance_m"], "meters")

# 3. Compute 360° Local Terrain Horizon Profile
horizon = dem.calculate_horizon_profile(
    observer_coord=(0.0, 0.0),
    max_distance_m=30000.0,
    azimuth_step_deg=5.0,
    observer_height_m=2.0
)
print("Max Horizon Obstacle:", horizon["max_horizon_elevation_deg"], "deg at az", horizon["azimuth_of_max_elevation_deg"])
```

### New REST API Endpoints
1. `POST /api/v1/terrain/ray`: Radial terrain ray transect with distance-dependent curvature drop.
2. `POST /api/v1/terrain/check-los`: Topographic line-of-sight obstruction query with tri-state response.
3. `POST /api/v1/terrain/dem-horizon`: Full 360° local horizon elevation profile computed directly from loaded DEM.

---

## 7. Full First Fresnel-Zone and RF Obstruction Clearance Geometry

### Theoretical Formulation
For lunar surface-to-Earth radio frequency communication (S-band, Ka-band), line-of-sight clearance requires evaluating diffraction clearance relative to the first Fresnel ellipsoid:

$$r_{F_1} = \sqrt{\frac{\lambda d_1 d_2}{d_1 + d_2}}$$

Where:
- $d_1$: Distance from the lunar surface transmitter to the obstructing terrain knife-edge ridge (meters).
- $d_2$: Distance from the ridge to the Earth receiving station ($d_{\text{Earth}} - d_1$ meters).
- $d_{\text{Earth}}$: Total lunar transmitter-to-Earth distance ($3.844 \times 10^8\text{ m}$). Because $d_1 \le 100\text{ km} \ll d_{\text{Earth}}$ ($d_1 / d_{\text{Earth}} < 0.026\%$), the path partition $d_{\text{Earth}} = d_1 + d_2$ is valid to sub-millimeter precision ($O((d_1/d_{\text{Earth}})^2) < 10^{-7}$).
- $\lambda = c / f$: Carrier wavelength in meters ($c = 299,792,458\text{ m/s}$).

### Geometric Clearance Definition
Let $\theta_{\text{Earth}}$ and $\theta_{\text{obs}}$ be the topocentric elevation angles of the Earth and the obstacle ridge peak, measured from the **same local horizontal tangent plane** at the lunar transmitter.

1. **Exact Vertical Clearance**:
   At horizontal distance $d_1$, the line-of-sight ray has altitude $z_{\text{LOS}} = d_1 \tan(\theta_{\text{Earth}})$ and the obstacle peak has altitude $z_{\text{obs}} = d_1 \tan(\theta_{\text{obs}})$.
   The exact vertical clearance is:
   $$\Delta z_{\text{vert}} = d_1 \left(\tan(\theta_{\text{Earth}}) - \tan(\theta_{\text{obs}})\right)$$

   *Elimination of Previous Approximation Error*:
   The previously used formula $\Delta z_{\text{approx}} = d_1 \tan(\theta_{\text{Earth}} - \theta_{\text{obs}})$ introduced an angular denominator error:
   $$\tan(\alpha - \beta) = \frac{\tan\alpha - \tan\beta}{1 + \tan\alpha \tan\beta} \implies \Delta z_{\text{exact}} - \Delta z_{\text{approx}} = \Delta z_{\text{approx}} \cdot (\tan\alpha \tan\beta)$$
   At $d_1 = 10\text{ km}$, $\theta_{\text{Earth}} = 5.0^\circ$, and $\theta_{\text{obs}} = 3.0^\circ$:
   - $\Delta z_{\text{exact}} = 10,000 \cdot (\tan 5^\circ - \tan 3^\circ) = 350.81\text{ m}$
   - $\Delta z_{\text{approx}} = 10,000 \cdot \tan(2^\circ) = 349.21\text{ m}$
   - Error eliminated: $+1.60\text{ m}$.

2. **Perpendicular Clearance & Transverse Screening**:
   The first Fresnel zone is a transverse circle oriented perpendicular to the direct propagation path. The perpendicular distance from the obstacle peak to the direct ray axis is:
   $$\Delta h_{\perp} = \Delta z_{\text{vert}} \cos(\theta_{\text{Earth}})$$

   Screening against 60% first Fresnel zone clearance (ITU-R P.526 knife-edge screening criterion):
   $$z_{\text{req}} = 0.6 \cdot r_{F_1}, \quad \text{Pass} \iff \Delta h_{\perp} \ge z_{\text{req}}$$
   Margin is defined as $\Delta h_{\perp} - z_{\text{req}}$.

### Verified Analytical Benchmarks
At $d_1 = 10,000\text{ m}$ and $d_{\text{Earth}} = 384,400,000\text{ m}$:
- **S-Band** ($2.2\text{ GHz}, \lambda = 0.136269\text{ m}$): $r_{F_1} = 36.914\text{ m}$, $0.6 r_{F_1} = 22.148\text{ m}$.
- **Ka-Band** ($32.0\text{ GHz}, \lambda = 0.0093685\text{ m}$): $r_{F_1} = 9.678\text{ m}$, $0.6 r_{F_1} = 5.807\text{ m}$.
- **Equal Elevations** ($\theta_{\text{Earth}} = \theta_{\text{obs}}$): $\Delta z_{\text{vert}} \equiv 0.0\text{ m}$, $\Delta h_{\perp} \equiv 0.0\text{ m}$.
- **Obstacle Above Path** ($\theta_{\text{obs}} > \theta_{\text{Earth}}$): $\Delta z_{\text{vert}} < 0$, $\Delta h_{\perp} < 0$ (direct knife-edge obstruction).


