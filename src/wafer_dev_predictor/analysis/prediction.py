"""
2D polynomial surface fit and Gaussian low-pass prediction for wafer anomaly analysis.

For each image, fits a 2D polynomial to the clean (non-anomalous) region and extrapolates
across the full surface to estimate what the surface would look like without the defect.

Why polynomial?
- Wafer surfaces are smooth optical surfaces: tilt + curvature + higher-order terms
- Fitting ONLY to the clean region and extrapolating gives the most physically meaningful
  prediction of the defect-free baseline
- Fast: < 1 second for a 1102×70 image via scipy.linalg.lstsq (no new dependencies)

Alternative: Gaussian low-pass via fastlibrary.gauss_low_pass — a global filter that
removes high-frequency content but may miss large-scale drift in the anomalous region.
"""

import numpy as np
from scipy import linalg


def compute_metrics(anomaly_z: np.ndarray, anom_mask: np.ndarray) -> dict:
    """
    Compute anomaly metrics over the masked region.

    Args:
        anomaly_z: 2D array of anomaly heights (nm). NaN where no valid data.
        anom_mask: Boolean mask — True for pixels to include in the metric.

    Returns:
        dict with keys: rms_nm, pv_nm, max_nm, mean_nm, anomaly_area_pct.
        anomaly_area_pct is the percentage of masked pixels with |deviation| > 10 nm.
    """
    vals = anomaly_z[anom_mask & ~np.isnan(anomaly_z)]
    if vals.size == 0:
        return {
            "rms_nm": 0.0,
            "pv_nm": 0.0,
            "max_nm": 0.0,
            "mean_nm": 0.0,
            "anomaly_area_pct": 0.0,
        }

    rms = float(np.sqrt(np.mean(vals**2)))
    pv = float(np.max(vals) - np.min(vals))
    max_abs = float(np.max(np.abs(vals)))
    mean = float(np.mean(vals))
    threshold_nm = 10.0
    anomaly_area_pct = float(np.mean(np.abs(vals) > threshold_nm) * 100)

    return {
        "rms_nm": rms,
        "pv_nm": pv,
        "max_nm": max_abs,
        "mean_nm": mean,
        "anomaly_area_pct": anomaly_area_pct,
    }


def _poly_design_matrix(
    rows: np.ndarray,
    cols: np.ndarray,
    n_rows: int,
    n_cols: int,
    degree: int,
) -> np.ndarray:
    """
    Build a 2D polynomial design matrix for the given pixel coordinates.

    Coordinates are normalized to [-1, 1] to improve numerical conditioning.
    For degree d, includes all monomials x^i * y^j where i + j <= d.
    """
    x = (cols / max(n_cols - 1, 1)) * 2.0 - 1.0  # column direction → x
    y = (rows / max(n_rows - 1, 1)) * 2.0 - 1.0  # row direction → y

    features = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            features.append((x**i) * (y**j))
    return np.column_stack(features)


def predict_normal_polynomial(
    z: np.ndarray,
    anom_col_start: int,
    anom_col_end: int,
    degree: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit a 2D polynomial to the clean region and extrapolate to predict the full surface.

    The polynomial is fit using scipy.linalg.lstsq on all pixels OUTSIDE the
    anomalous column range that are not NaN. It is then evaluated on the full grid
    to produce predicted_z. The anomaly map is actual_z - predicted_z.

    Args:
        z: 2D height map in nm, shape (n_rows, n_cols). NaN where no valid data.
        anom_col_start: First anomalous column (inclusive).
        anom_col_end: Last anomalous column (exclusive).
        degree: Polynomial degree. Default 5 gives tilt + curvature + higher-order terms.

    Returns:
        (predicted_z, anomaly_z, metrics) where metrics are computed on the anomalous region.
    """
    n_rows, n_cols = z.shape
    row_idx, col_idx = np.mgrid[0:n_rows, 0:n_cols]

    # Clean mask: outside anomalous columns AND not NaN
    clean_mask = np.ones((n_rows, n_cols), dtype=bool)
    if anom_col_start < anom_col_end:
        clean_mask[:, anom_col_start:anom_col_end] = False
    clean_mask &= ~np.isnan(z)

    clean_rows = row_idx[clean_mask].astype(float)
    clean_cols = col_idx[clean_mask].astype(float)
    clean_z = z[clean_mask]

    # Fit polynomial to clean pixels
    A_clean = _poly_design_matrix(clean_rows, clean_cols, n_rows, n_cols, degree)
    coeffs, _, _, _ = linalg.lstsq(A_clean, clean_z)

    # Evaluate on full grid
    all_rows = row_idx.flatten().astype(float)
    all_cols = col_idx.flatten().astype(float)
    A_full = _poly_design_matrix(all_rows, all_cols, n_rows, n_cols, degree)
    predicted_z = (A_full @ coeffs).reshape(n_rows, n_cols)

    # Preserve NaN structure from original
    predicted_z = np.where(np.isnan(z), np.nan, predicted_z)

    anomaly_z = z - predicted_z

    # Metrics on the anomalous region only
    anom_mask = np.zeros((n_rows, n_cols), dtype=bool)
    if anom_col_start < anom_col_end:
        anom_mask[:, anom_col_start:anom_col_end] = True

    metrics = compute_metrics(anomaly_z, anom_mask)
    return predicted_z, anomaly_z, metrics


def _weighted_lstsq(A: np.ndarray, z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Solve weighted least squares by rescaling rows by sqrt(weight)."""
    w_sqrt = np.sqrt(weights)
    coeffs, _, _, _ = linalg.lstsq(A * w_sqrt[:, np.newaxis], z * w_sqrt)
    return coeffs


def predict_normal_robust_polynomial(
    z: np.ndarray,
    degree: int = 5,
    n_iter: int = 5,
    k_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit a robust 2D polynomial to the FULL image using IRLS (Huber weights).

    No manual region selection required. The robust estimator automatically
    downweights pixels that deviate from the smooth process background, so the
    fitted polynomial captures the process signature only — not the defect.

    Why this works:
        Every wafer diff map = smooth process component + localised defect.
        The process component is low spatial frequency (well described by a
        low-degree polynomial). The defect is a localised deviation on top.
        IRLS finds the polynomial that best fits the MAJORITY of pixels while
        treating the defect pixels as outliers — without knowing where they are.

    Algorithm (Huber IRLS):
        1. Initial unweighted polynomial fit to all valid pixels.
        2. Compute residuals → robust sigma via MAD * 1.4826.
        3. Huber weights: w_i = min(1,  k_sigma * sigma / |r_i|)
           Pixels with large residuals (defects) get weight < 1.
        4. Weighted least-squares refit.
        5. Repeat steps 2-4 for n_iter iterations.
        Defect pixels converge to near-zero weight; clean pixels stay at 1.

    Args:
        z: 2D height map in nm. NaN where no valid data.
        degree: Polynomial degree (default 5).
        n_iter: IRLS iterations (default 5 — converges in 3-4).
        k_sigma: Huber threshold in units of robust sigma (default 2.0).
                 Lower → more aggressive outlier rejection (catches subtle defects).
                 Higher → more conservative (only flags obvious defects).

    Returns:
        (predicted_z, anomaly_z, metrics)
        anomaly_z = z - predicted_z  (the residual after removing process component)
        metrics are computed on pixels where |residual| > k_sigma * sigma_robust,
        i.e. exactly the pixels IRLS identified as non-process-like.
    """
    n_rows, n_cols = z.shape
    row_idx, col_idx = np.mgrid[0:n_rows, 0:n_cols]

    valid_mask = ~np.isnan(z)
    valid_rows = row_idx[valid_mask].astype(float)
    valid_cols = col_idx[valid_mask].astype(float)
    valid_z = z[valid_mask]

    A_valid = _poly_design_matrix(valid_rows, valid_cols, n_rows, n_cols, degree)

    # Initial unweighted fit
    coeffs, _, _, _ = linalg.lstsq(A_valid, valid_z)

    # IRLS iterations
    for _ in range(n_iter):
        residuals = valid_z - A_valid @ coeffs
        sigma = np.median(np.abs(residuals)) * 1.4826  # robust sigma via MAD
        if sigma < 1e-10:
            break
        weights = np.minimum(1.0, (k_sigma * sigma) / np.abs(residuals))
        coeffs = _weighted_lstsq(A_valid, valid_z, weights)

    # Final residuals and robust sigma for anomaly mask
    final_residuals = valid_z - A_valid @ coeffs
    sigma_final = np.median(np.abs(final_residuals)) * 1.4826

    # Evaluate polynomial on full grid
    all_rows = row_idx.flatten().astype(float)
    all_cols = col_idx.flatten().astype(float)
    A_full = _poly_design_matrix(all_rows, all_cols, n_rows, n_cols, degree)
    predicted_z = (A_full @ coeffs).reshape(n_rows, n_cols)
    predicted_z = np.where(np.isnan(z), np.nan, predicted_z)

    anomaly_z = z - predicted_z

    # Anomaly mask: pixels whose residual exceeds the IRLS outlier threshold —
    # exactly the pixels the robust fit identified as not fitting the process shape
    anom_mask = (np.abs(anomaly_z) > k_sigma * sigma_final) & valid_mask

    metrics = compute_metrics(anomaly_z, anom_mask)
    return predicted_z, anomaly_z, metrics


def inpaint_region(
    z: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace a rectangular region with a polynomial surface fit from surrounding pixels.

    Fits a 2D polynomial to all valid pixels OUTSIDE the rectangle, then
    evaluates it on the full grid. Returns the cleaned image (original with
    rectangle replaced by polynomial prediction) and the full predicted surface.
    """
    n_rows, n_cols = z.shape
    row_idx, col_idx = np.mgrid[0:n_rows, 0:n_cols]

    clean_mask = np.ones((n_rows, n_cols), dtype=bool)
    clean_mask[row_start:row_end, col_start:col_end] = False
    clean_mask &= ~np.isnan(z)

    clean_rows = row_idx[clean_mask].astype(float)
    clean_cols = col_idx[clean_mask].astype(float)
    clean_z = z[clean_mask]

    A_clean = _poly_design_matrix(clean_rows, clean_cols, n_rows, n_cols, degree)
    coeffs, _, _, _ = linalg.lstsq(A_clean, clean_z)

    all_rows = row_idx.flatten().astype(float)
    all_cols = col_idx.flatten().astype(float)
    A_full = _poly_design_matrix(all_rows, all_cols, n_rows, n_cols, degree)
    predicted_z = (A_full @ coeffs).reshape(n_rows, n_cols)
    predicted_z = np.where(np.isnan(z), np.nan, predicted_z)

    cleaned_z = z.copy()
    cleaned_z[row_start:row_end, col_start:col_end] = predicted_z[row_start:row_end, col_start:col_end]

    return cleaned_z, predicted_z


def predict_normal_gaussian(
    topo: "fl.Topography",
    fwhm_m: float = 0.005,
    anom_col_start: int = 0,
    anom_col_end: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply a Gaussian low-pass filter as the normal surface estimate.

    The Gaussian filter is applied globally (not just to the clean region), so it
    may be influenced by the anomaly itself. It is provided as a comparison method.

    Args:
        topo: fastlibrary Topography object.
        fwhm_m: FWHM of the Gaussian filter in metres (default 5 mm).
        anom_col_start: First anomalous column (for metrics only).
        anom_col_end: Last anomalous column (for metrics only).

    Returns:
        (predicted_z, anomaly_z, metrics) where metrics are computed on the anomalous region.
    """
    filtered_topo = topo.gauss_low_pass(fwhm_m=fwhm_m)
    predicted_z = filtered_topo.z_map * 1e9
    actual_z = topo.z_map * 1e9
    anomaly_z = actual_z - predicted_z

    n_rows, n_cols = actual_z.shape
    anom_mask = np.zeros((n_rows, n_cols), dtype=bool)
    if anom_col_start < anom_col_end:
        anom_mask[:, anom_col_start:anom_col_end] = True

    metrics = compute_metrics(anomaly_z, anom_mask)
    return predicted_z, anomaly_z, metrics
