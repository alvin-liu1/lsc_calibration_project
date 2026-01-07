# LSC Algorithm Documentation

## Algorithm Overview

This document provides detailed technical documentation for the Lens Shading Correction (LSC) algorithm used in this project.

## Core Algorithm: Radial Polynomial Fitting (V5.1)

### Problem Statement

Traditional LSC calibration methods use direct grid statistics:
1. Divide image into N×M cells
2. Calculate average brightness for each cell
3. Use average as gain table

**Problems with this approach:**
- Grid statistics are noisy (especially in dark corners)
- Produces visible stripe artifacts
- Color fringing at edges
- Requires heavy smoothing (loses accuracy)

### Our Solution: Radial Polynomial Fitting

Instead of using raw grid statistics, we fit the lens vignetting to a smooth mathematical curve based on optical physics.

### Algorithm Steps

#### Step 1: Grid Statistics Collection
```python
# Divide image into 16×12 cells (17×13 vertices)
for each cell in grid:
    valid_pixels = cell_pixels[mask == 1]
    cell_brightness[r, c] = mean(valid_pixels)
```

This produces a **noisy brightness map** - but this is only intermediate data.

#### Step 2: Radial Coordinate Transform
```python
# Convert grid coordinates to radial distance
for each vertex (r, c):
    pixel_x = c * (image_width / grid_cols)
    pixel_y = r * (image_height / grid_rows)

    # Distance from fisheye center
    distance = sqrt((pixel_x - cx)^2 + (pixel_y - cy)^2)

    # Normalize to [0, 1]
    normalized_radius = distance / fisheye_radius
```

#### Step 3: Adaptive Data Filtering
```python
# Only use high SNR data for fitting
valid_mask = (brightness > 0.001) & (normalized_radius < 0.92)

# This excludes:
# - Completely black pixels (brightness too low)
# - Outermost 8% of circle (low SNR, prone to noise)
```

**Why 0.92?**
- Edge pixels have lowest SNR
- Noise causes polynomial to "overshoot" at edges
- By excluding edge 8%, we get stable fit that naturally extends to edges

#### Step 4: 4th Order Polynomial Fit
```python
# Fit brightness falloff curve
coefficients = polyfit(normalized_radius, brightness, degree=4)

# Polynomial form:
# brightness(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
```

**Why 4th order?**
- Lower orders (2-3): Can't capture complex vignetting patterns
- Higher orders (6+): Overfitting → edge oscillations
- 4th order: Sweet spot for fisheye lenses

#### Step 5: Gain Calculation
```python
# Reconstruct smooth brightness for ALL grid points
fitted_brightness = poly4(normalized_radius)

# Calculate gain
max_brightness = max(brightness)  # Center of image
gain = max_brightness / fitted_brightness

# Result: Mathematically smooth gain table
```

#### Step 6: Geometric Damping
```python
# For pixels OUTSIDE fisheye circle:
start_damp_radius = fisheye_radius * 1.05  # Start at 105%
end_damp_radius = start_damp_radius + damping_width

# Smoothly transition gain from fitted_value → 1.0
weight = clip((end - distance) / (end - start), 0, 1)
final_gain = fitted_gain * weight + 1.0 * (1 - weight)
```

**Purpose:**
- Prevent noise amplification in dead black area
- Eliminate Bicubic interpolation ringing artifacts
- Smooth transition to unity gain

## ISP Pipeline Integration

### Standard ISP Pipeline Order

```
RAW → BLC → WB → LSC → Demosaic → CCM → Gamma → ...
        ↑         ↑
     Ideal    Current
```

**Note:** Industry best practice is WB before LSC (LSC gains are color-temperature dependent). Current implementation does LSC before WB for simplicity. For production, consider multi-temperature LSC tables.

### Bayer Domain Processing

```python
# Extract 4 Bayer channels
R  = bayer[0::2, 0::2]   # Red
Gr = bayer[0::2, 1::2]   # Green-Red row
Gb = bayer[1::2, 0::2]   # Green-Blue row
B  = bayer[1::2, 1::2]   # Blue

# Independent BLC for each channel
for ch in [R, Gr, Gb, B]:
    ch_blc = max(0, ch - black_level[ch])
    ch_norm = ch_blc / (sensor_max - black_level[ch])

# Calculate separate gain tables for each channel
for ch in [R, Gr, Gb, B]:
    gain_table[ch] = radial_polynomial_fit(ch_norm)

# Apply gains in Bayer domain
compensated_bayer = bayer_blc * gain_map

# CRITICAL: Zero out-of-circle pixels
compensated_bayer[mask == 0] = 0

# Then demosaic
rgb = demosaic(compensated_bayer)
```

### Hard Mask Protection

**Why zero out-of-circle pixels?**

```
Without hard mask:
1. Out-of-circle pixels contain random noise
2. LSC amplifies noise (gain can be 2-8x)
3. Demosaic interpolates amplified noise into valid region
4. Result: Bright noise ring around fisheye circle

With hard mask:
1. Apply LSC gain to entire image
2. Force out-of-circle pixels to 0.0
3. Demosaic sees clean zero boundary
4. Result: Clean fisheye edge
```

## Qualcomm ISP Format: 13uQ10

### Format Specification

```
13uQ10 = 13-bit unsigned, 10 fractional bits

Integer representation:
int_value = round(float_gain * 1024)

Valid range:
- Minimum: 1024 (represents gain 1.0)
- Maximum: 8191 (represents gain 7.99...)

Conversion:
float_gain = int_value / 1024.0
```

### Example Values

| Float Gain | Q10 Integer | Hex    |
|------------|-------------|--------|
| 1.0        | 1024        | 0x0400 |
| 1.5        | 1536        | 0x0600 |
| 2.0        | 2048        | 0x0800 |
| 4.0        | 4096        | 0x1000 |
| 7.99       | 8191        | 0x1FFF |

### Output Format

```
# Qualcomm Chromatix LSC Table Format
# Each line: 17 values (horizontal vertices)
# Total lines: 13 (vertical vertices)

1024 1050 1089 1142 1210 ... (17 values for R channel)
1050 1075 1115 1167 1234 ...
...
(13 lines)

[Next channel: Gr, then Gb, then B]
```

## Post-Processing Steps

### 1. Extrapolation and Smoothing

**Problem:** Some grid cells might have failed (too dark, out of mask)

```python
def extrapolate_and_smooth_gains(gain_matrix):
    # Find invalid cells (gain = 1.0 = failed calculation)
    invalid_mask = (gain_matrix == 1.0)

    # Use KD-tree to find nearest valid neighbor
    valid_points = get_valid_coordinates()
    kdtree = cKDTree(valid_points)

    for invalid_point in invalid_points:
        nearest = kdtree.query(invalid_point)
        gain_matrix[invalid_point] = gain_matrix[nearest]

    # Now apply Gaussian smoothing on filled surface
    smoothed = gaussian_filter(gain_matrix, sigma)
```

### 2. Symmetrization

**Problem:** Real-world imperfections (lens decentering, uneven illumination)

```python
def symmetrize_table(table):
    rows, cols = table.shape
    center_r = rows // 2
    center_c = cols // 2

    for r in range(rows):
        for c in range(cols):
            # Find symmetric point
            sym_r = 2 * center_r - r
            sym_c = 2 * center_c - c

            # Average with symmetric point
            if in_bounds(sym_r, sym_c):
                avg = (table[r,c] + table[sym_r, sym_c]) / 2
                table[r,c] = table[sym_r, sym_c] = avg
```

**Trade-off:**
- Pro: Smoother, more physically correct
- Con: May hide real lens asymmetry issues

## Performance Analysis

### Computational Complexity

| Operation | Complexity | Time (3MP) |
|-----------|------------|------------|
| Grid Statistics | O(W×H) | ~100ms |
| Polynomial Fit | O(N log N) per channel | ~10ms |
| Geometric Damping | O(grid_size) | ~1ms |
| Bicubic Resize | O(W×H) | ~50ms |
| Bayer Gain Application | O(W×H) | ~80ms |
| Demosaic | O(W×H) | ~200ms |

**Total:** ~450ms per image (single-threaded on Intel i7)

### Optimization Opportunities

1. **Vectorize Grid Statistics**
```python
# Current: Double loop (slow)
for r in range(grid_rows):
    for c in range(grid_cols):
        cell_mean = ...

# Optimized: Numpy advanced indexing
cell_indices = ...
cell_means = np.bincount(cell_indices, weights=values)
```

2. **Parallelize Channels**
```python
from multiprocessing import Pool

with Pool(4) as p:
    gains = p.map(fit_radial_gain, [R_ch, Gr_ch, Gb_ch, B_ch])
```

3. **GPU Acceleration**
```python
import cupy as cp  # CUDA acceleration

# Move to GPU
bayer_gpu = cp.array(bayer)
gain_map_gpu = cp.array(gain_map)

# Fast element-wise multiply
result_gpu = bayer_gpu * gain_map_gpu
```

## Validation and Quality Metrics

### 1. Uniformity Ratio

```python
def calculate_uniformity(image, mask):
    center_region = image[mask & (r < 0.3 * radius)]
    edge_region = image[mask & (r > 0.7 * radius)]

    uniformity = mean(edge_region) / mean(center_region)
    # Target: > 0.95
```

### 2. Color Ratio Stability

```python
def evaluate_color_ratio(corrected_image, mask):
    R = corrected_image[:,:,0][mask]
    G = corrected_image[:,:,1][mask]
    B = corrected_image[:,:,2][mask]

    RG_ratio_std = std(R / G)
    BG_ratio_std = std(B / G)
    # Target: < 0.05
```

### 3. Gain Smoothness

```python
def check_gain_smoothness(gain_table):
    # Calculate gradient magnitude
    dy, dx = np.gradient(gain_table)
    gradient_mag = np.sqrt(dx**2 + dy**2)

    max_gradient = np.max(gradient_mag)
    # Target: < 0.2 per grid cell
```

## Limitations and Future Work

### Current Limitations

1. **Single Color Temperature**
   - Calibration specific to D65 (or whatever illumination used)
   - Real lenses have color-temperature-dependent vignetting

2. **Circular Fisheye Assumption**
   - Algorithm assumes radially symmetric vignetting
   - Not suitable for:
     - Rectangular sensors (16:9 with corner clipping)
     - Decentered lenses
     - Elliptical fisheye

3. **No Dead Pixel Correction**
   - Should be done before LSC in production pipeline

4. **No Noise Model**
   - LSC amplifies dark corner noise
   - Should be paired with spatially-varying NR

### Future Enhancements

1. **Multi-Temperature Calibration**
```python
# Capture multiple illuminations
tables = {
    'D65': calibrate(raw_d65),
    'A': calibrate(raw_tungsten),
    'TL84': calibrate(raw_fluorescent)
}

# Runtime interpolation based on AWB CCT
```

2. **Elliptical Fitting**
```python
# Use 2D polynomial instead of radial
gain(x, y) = poly2d(x, y)
```

3. **Automatic Quality Assessment**
```python
def auto_validate(result):
    if uniformity < 0.90:
        raise Warning("Poor correction quality")
    if max_gain > 6.0:
        raise Warning("Excessive vignetting - check lens")
```

## References

- **OpenCV**: Bayer demosaicing and image processing
- **Qualcomm ISP**: Chromatix tuning documentation
- **Lens Vignetting**: "Vignetting in Digital Cameras" - H. Zhang et al.
- **Polynomial Fitting**: NumPy documentation

---

**Author:** LSC Calibration Project
**Version:** 5.1
**Last Updated:** 2025-01-07
