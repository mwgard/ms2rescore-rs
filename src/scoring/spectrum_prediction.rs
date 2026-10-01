// Spectrum prediction comparison features (batch + flexible NumPy inputs):
//
// - Minimize peak memory usage by processing in blocks, and keeping f32 arrays only
//   while holding the GIL.
// - Use Rayon for parallelism outside the GIL.
// - Support arbitrary array-like inputs (lists, different dtypes, non-contiguous arrays)
//   by converting to contiguous np.ndarray(float32) once per input spectrum.

use std::collections::HashMap;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use rayon::prelude::*;

const CLIP_LOG2_MIN: f64 = crate::utils::LOG2_FLOOR_F64; // (0.001_f64).log2()
const SPECTRA_ANGLE_EPS: f64 = 1e-7;
/// Intensities at or below this value count as absent peaks in the NIST ratio of peak pairs.
/// Unlogged floor values (2^log2(0.001) - 0.001) are ~0 up to floating point noise.
const NIST_NOISE_THRESHOLD: f64 = 1e-9;

#[inline]
fn clip_min_f32(x: f32) -> f64 {
    let xf = x as f64;
    if xf < CLIP_LOG2_MIN {
        CLIP_LOG2_MIN
    } else {
        xf
    }
}

#[inline]
fn pow2_unlog(x: f64) -> f64 {
    // matches Python: 2**x - 0.001
    x.exp2() - 0.001
}

#[inline]
fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[inline]
fn any_to_vec_f32<'py>(
    np: &'py Bound<'py, PyModule>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Vec<f32>> {
    // np.ascontiguousarray(obj, dtype=np.float32)
    let arr_any = np.getattr("ascontiguousarray")?.call1((obj, "float32"))?;

    // Convert to a Bound<PyArray1<f32>>
    let arr = arr_any.cast::<PyArray1<f32>>()?;
    let ro = arr.readonly();

    if let Ok(slice) = ro.as_slice() {
        Ok(slice.to_vec())
    } else {
        Ok(ro.as_array().iter().copied().collect())
    }
}

#[inline]
fn any_to_vec_f64<'py>(
    np: &'py Bound<'py, PyModule>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Vec<f64>> {
    let arr_any = np.getattr("ascontiguousarray")?.call1((obj, "float64"))?;
    let arr = arr_any.cast::<PyArray1<f64>>()?;
    let ro = arr.readonly();
    if let Ok(slice) = ro.as_slice() {
        Ok(slice.to_vec())
    } else {
        Ok(ro.as_array().iter().copied().collect())
    }
}

/// Positive log-like transform used for the m/z-weighted log-space features, so that
/// fractional exponents are defined: ln(unlogged + 1.001).
#[inline]
fn positive_log(x: f64) -> f64 {
    (x + 1.001).ln()
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n < 2 {
        return f64::NAN;
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if den_x <= 0.0 || den_y <= 0.0 {
        return f64::NAN;
    }
    num / (den_x.sqrt() * den_y.sqrt())
}

fn mse(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n == 0 {
        return f64::NAN;
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }

    let mut s = 0.0;
    for i in 0..n {
        let d = x[i] - y[i];
        s += d * d;
    }
    s / n as f64
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() {
        return f64::NAN;
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn l2_norm(x: &[f64]) -> f64 {
    if x.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn cosine_similarity(x: &[f64], y: &[f64]) -> f64 {
    let d = dot(x, y);
    let nx = l2_norm(x);
    let ny = l2_norm(y);
    if !d.is_finite() || !nx.is_finite() || !ny.is_finite() || nx <= 0.0 || ny <= 0.0 {
        return f64::NAN;
    }
    d / (nx * ny)
}

#[inline]
fn dot2(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> f64 {
    if a1.len() != b1.len() || a2.len() != b2.len() {
        return f64::NAN;
    }
    let d1 = dot(a1, b1);
    let d2 = dot(a2, b2);
    if !d1.is_finite() || !d2.is_finite() {
        return f64::NAN;
    }
    d1 + d2
}

#[inline]
fn mse2(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> f64 {
    if a1.len() != b1.len() || a2.len() != b2.len() {
        return f64::NAN;
    }
    let n = a1.len() + a2.len();
    if n == 0 {
        return f64::NAN;
    }
    if a1.iter().any(|v| !v.is_finite())
        || a2.iter().any(|v| !v.is_finite())
        || b1.iter().any(|v| !v.is_finite())
        || b2.iter().any(|v| !v.is_finite())
    {
        return f64::NAN;
    }

    let mut s = 0.0;
    for (x, y) in a1.iter().zip(b1.iter()) {
        let d = x - y;
        s += d * d;
    }
    for (x, y) in a2.iter().zip(b2.iter()) {
        let d = x - y;
        s += d * d;
    }
    s / (n as f64)
}

#[inline]
fn pearson2(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> f64 {
    if a1.len() != b1.len() || a2.len() != b2.len() {
        return f64::NAN;
    }
    let n = a1.len() + a2.len();
    if n < 2 {
        return f64::NAN;
    }
    if a1.iter().any(|v| !v.is_finite())
        || a2.iter().any(|v| !v.is_finite())
        || b1.iter().any(|v| !v.is_finite())
        || b2.iter().any(|v| !v.is_finite())
    {
        return f64::NAN;
    }

    let sum_x = a1.iter().sum::<f64>() + a2.iter().sum::<f64>();
    let sum_y = b1.iter().sum::<f64>() + b2.iter().sum::<f64>();
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for (x, y) in a1.iter().zip(b1.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    for (x, y) in a2.iter().zip(b2.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if den_x <= 0.0 || den_y <= 0.0 {
        return f64::NAN;
    }
    num / (den_x.sqrt() * den_y.sqrt())
}

#[inline]
fn cosine2(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> f64 {
    let d = dot2(a1, a2, b1, b2);
    if !d.is_finite() {
        return f64::NAN;
    }
    let nx1 = l2_norm(a1);
    let nx2 = l2_norm(a2);
    let ny1 = l2_norm(b1);
    let ny2 = l2_norm(b2);
    if !nx1.is_finite() || !nx2.is_finite() || !ny1.is_finite() || !ny2.is_finite() {
        return f64::NAN;
    }
    let nx = (nx1 * nx1 + nx2 * nx2).sqrt();
    let ny = (ny1 * ny1 + ny2 * ny2).sqrt();
    if nx <= 0.0 || ny <= 0.0 {
        return f64::NAN;
    }
    d / (nx * ny)
}

/// Spectral angle similarity (Toprak et al., 2014): 1 - 2 * arccos(cos) / pi.
fn spectral_angle(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return f64::NAN;
    }
    let d = dot(x, y);
    if !d.is_finite() {
        return f64::NAN;
    }
    // epsilon safeguard on the squared norms, as in the reference implementation
    let nx = x.iter().map(|v| v * v).sum::<f64>().max(SPECTRA_ANGLE_EPS).sqrt();
    let ny = y.iter().map(|v| v * v).sum::<f64>().max(SPECTRA_ANGLE_EPS).sqrt();
    let c = (d / (nx * ny)).clamp(-1.0, 1.0);
    1.0 - 2.0 * c.acos() / std::f64::consts::PI
}

/// SpectraST dot product (Lam et al., 2007): squared cosine of the unit-normalized spectra.
fn spectrast(x: &[f64], y: &[f64]) -> f64 {
    let c = cosine_similarity(x, y);
    if !c.is_finite() {
        return f64::NAN;
    }
    c * c
}

#[inline]
fn weight_peaks(mz: &[f64], intensity: &[f64], mz_weight: f64, intensity_weight: f64) -> Vec<f64> {
    mz.iter()
        .zip(intensity.iter())
        .map(|(m, i)| i.powf(intensity_weight) * m.powf(mz_weight))
        .collect()
}

/// Weighted (Sokolow: m/z^1, intensity^0.5) normalized dot product.
fn weighted_dotprod(mz: &[f64], x: &[f64], y: &[f64]) -> f64 {
    if mz.len() != x.len() || x.len() != y.len() || x.is_empty() {
        return f64::NAN;
    }
    let a = weight_peaks(mz, x, 1.0, 0.5);
    let b = weight_peaks(mz, y, 1.0, 0.5);
    cosine_similarity(&a, &b)
}

/// NIST MS/MS match factor (Stein & Scott, 1994): weighted dot product cosine (DPC; m/z^1,
/// intensity^0.5) combined with the ratio of peak pairs (RPP; intensity^1) of consecutive
/// peaks present in both spectra (intensity > `NIST_NOISE_THRESHOLD`).
///
/// `x` plays the role of the "unknown" spectrum: the number of its peaks weights the DPC.
fn nist_match_factor(mz: &[f64], x: &[f64], y: &[f64]) -> f64 {
    let n = mz.len();
    if n != x.len() || n != y.len() || n == 0 {
        return f64::NAN;
    }
    if mz.iter().chain(x.iter()).chain(y.iter()).any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| mz[i].total_cmp(&mz[j]));
    let mz_s: Vec<f64> = order.iter().map(|&i| mz[i]).collect();
    let x_s: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let y_s: Vec<f64> = order.iter().map(|&i| y[i]).collect();

    // DPC
    let a = weight_peaks(&mz_s, &x_s, 1.0, 0.5);
    let b = weight_peaks(&mz_s, &y_s, 1.0, 0.5);
    let num = dot(&a, &b).powi(2);
    let den = a.iter().map(|v| v * v).sum::<f64>() * b.iter().map(|v| v * v).sum::<f64>();
    let dpc = if den > 0.0 && num.is_finite() { num / den } else { 0.0 };

    // RPP over consecutive peaks common to both spectra
    let common: Vec<usize> = (0..n)
        .filter(|&i| x_s[i] > NIST_NOISE_THRESHOLD && y_s[i] > NIST_NOISE_THRESHOLD)
        .collect();
    let n_common = common.len();
    if n_common < 2 {
        return 0.0;
    }
    let n_x = x_s.iter().filter(|&&v| v > NIST_NOISE_THRESHOLD).count();
    let rpp_sum: f64 = common
        .windows(2)
        .map(|w| {
            let r = (y_s[w[1]] / y_s[w[0]]) * (x_s[w[0]] / x_s[w[1]]);
            if r < 1.0 {
                r
            } else {
                1.0 / r
            }
        })
        .sum();
    let rpp = rpp_sum / n_x as f64;
    (n_x as f64 * dpc + n_common as f64 * rpp) / (n_x + n_common) as f64
}

fn mean_std(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64; // ddof=0 like numpy default
    (mean, var.sqrt())
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = (n as f64 - 1.0) * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

fn ranks_average_ties(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();

    // Deterministic ordering
    idx.sort_by(|&i, &j| values[i].total_cmp(&values[j]));

    let mut ranks = vec![0.0; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // average rank for ties, ranks are 1-based
        let r_lo = (i + 1) as f64;
        let r_hi = j as f64;
        let r_avg = (r_lo + r_hi) / 2.0;
        for k in i..j {
            ranks[idx[k]] = r_avg;
        }
        i = j;
    }
    ranks
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let rx = ranks_average_ties(x);
    let ry = ranks_average_ties(y);
    pearson(&rx, &ry)
}

/// Compute MS²PIP spectrum comparison features from predicted and observed b/y intensities
/// (log2(TIC-normalized + 0.001), as returned by MS²PIP).
///
/// If ``theoretical_mz_b`` and ``theoretical_mz_y`` are given, the m/z-weighted features
/// (``weighted_dotprod*`` and ``nist_match_factor*``) are added as well.
/// Spectral angle, SpectraST, weighted dot product, and NIST match factor features in log
/// (``*_norm``) and unlogged space. The m/z-weighted features are only added if the
/// concatenated (b then y) theoretical m/z values are given.
#[allow(clippy::too_many_arguments)]
fn add_similarity_features(
    feats: &mut HashMap<String, f64>,
    tb: &[f64],
    ty: &[f64],
    pb: &[f64],
    pyv: &[f64],
    tb_u: &[f64],
    ty_u: &[f64],
    pb_u: &[f64],
    py_u: &[f64],
    t_all_u: &[f64],
    p_all_u: &[f64],
    mz_all: Option<&[f64]>,
) {
    let t_all: Vec<f64> = tb.iter().chain(ty.iter()).copied().collect();
    let p_all: Vec<f64> = pb.iter().chain(pyv.iter()).copied().collect();
    let mut put = |name: &str, value: f64| {
        feats.insert(name.into(), finite_or_zero(value));
    };

    // log space
    put("spectrast_norm", spectrast(&t_all, &p_all));
    put("spectrast_ionb_norm", spectrast(tb, pb));
    put("spectrast_iony_norm", spectrast(ty, pyv));
    put("spectral_angle_norm", spectral_angle(&t_all, &p_all));
    put("spectral_angle_ionb_norm", spectral_angle(tb, pb));
    put("spectral_angle_iony_norm", spectral_angle(ty, pyv));
    // normal space
    put("spectrast", spectrast(t_all_u, p_all_u));
    put("spectrast_ionb", spectrast(tb_u, pb_u));
    put("spectrast_iony", spectrast(ty_u, py_u));
    put("spectral_angle", spectral_angle(t_all_u, p_all_u));
    put("spectral_angle_ionb", spectral_angle(tb_u, pb_u));
    put("spectral_angle_iony", spectral_angle(ty_u, py_u));

    let Some(mz_all) = mz_all else {
        return;
    };
    if mz_all.len() != t_all.len() {
        return;
    }
    let (mz_b, mz_y) = mz_all.split_at(tb.len());
    // positive log-like values for fractional exponents in log space
    let pos = |v: &[f64]| -> Vec<f64> { v.iter().copied().map(positive_log).collect() };
    let (tb_p, ty_p, pb_p, py_p) = (pos(tb_u), pos(ty_u), pos(pb_u), pos(py_u));
    let (t_all_p, p_all_p) = (pos(t_all_u), pos(p_all_u));

    put("weighted_dotprod_norm", weighted_dotprod(mz_all, &t_all_p, &p_all_p));
    put("weighted_dotprod_ionb_norm", weighted_dotprod(mz_b, &tb_p, &pb_p));
    put("weighted_dotprod_iony_norm", weighted_dotprod(mz_y, &ty_p, &py_p));
    put("nist_match_factor_norm", nist_match_factor(mz_all, &t_all_p, &p_all_p));
    put("weighted_dotprod", weighted_dotprod(mz_all, t_all_u, p_all_u));
    put("weighted_dotprod_ionb", weighted_dotprod(mz_b, tb_u, pb_u));
    put("weighted_dotprod_iony", weighted_dotprod(mz_y, ty_u, py_u));
    put("nist_match_factor", nist_match_factor(mz_all, t_all_u, p_all_u));
}

#[pyfunction]
#[pyo3(signature = (
    psm_indices, predicted_b, predicted_y, observed_b, observed_y,
    theoretical_mz_b=None, theoretical_mz_y=None
))]
#[allow(clippy::too_many_arguments)]
pub fn ms2pip_features_from_prediction_peak_arrays(
    py: Python<'_>,
    psm_indices: Vec<usize>,
    predicted_b: Vec<Py<PyAny>>,
    predicted_y: Vec<Py<PyAny>>,
    observed_b: Vec<Py<PyAny>>,
    observed_y: Vec<Py<PyAny>>,
    theoretical_mz_b: Option<Vec<Py<PyAny>>>,
    theoretical_mz_y: Option<Vec<Py<PyAny>>>,
) -> PyResult<Vec<(usize, HashMap<String, f64>)>> {
    let n = psm_indices.len();
    if predicted_b.len() != n
        || predicted_y.len() != n
        || observed_b.len() != n
        || observed_y.len() != n
    {
        return Err(PyValueError::new_err(
            "All inputs must have identical length: psm_indices, predicted_b, predicted_y, observed_b, observed_y",
        ));
    }
    let theoretical_mz = match (theoretical_mz_b, theoretical_mz_y) {
        (Some(mb), Some(my)) => {
            if mb.len() != n || my.len() != n {
                return Err(PyValueError::new_err(
                    "theoretical_mz_b and theoretical_mz_y must have the same length as psm_indices",
                ));
            }
            Some((mb, my))
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "theoretical_mz_b and theoretical_mz_y must be given together",
            ))
        }
    };

    #[derive(Clone)]
    struct Owned {
        idx: usize,
        pb: Vec<f32>,
        py: Vec<f32>,
        ob: Vec<f32>,
        oy: Vec<f32>,
        mz: Option<Vec<f64>>,
    }

    let mut out: Vec<(usize, HashMap<String, f64>)> = Vec::with_capacity(n);

    // Tune for peak memory vs overhead.
    let block_size: usize = 4096;

    // Import numpy once per call.
    let np: Bound<'_, PyModule> = PyModule::import(py, "numpy")?;

    for start in (0..n).step_by(block_size) {
        let end = (start + block_size).min(n);

        // ---- Convert/copy while holding the GIL (only this block) ----
        let mut owned: Vec<Owned> = Vec::with_capacity(end - start);
        for i in start..end {
            let pb_obj = predicted_b[i].bind(py);
            let py_obj = predicted_y[i].bind(py);
            let ob_obj = observed_b[i].bind(py);
            let oy_obj = observed_y[i].bind(py);

            let pb_vec = any_to_vec_f32(&np, pb_obj)?;
            let py_vec = any_to_vec_f32(&np, py_obj)?;
            let ob_vec = any_to_vec_f32(&np, ob_obj)?;
            let oy_vec = any_to_vec_f32(&np, oy_obj)?;

            let mz_vec = match &theoretical_mz {
                Some((mb, my)) => {
                    let mut v = any_to_vec_f64(&np, mb[i].bind(py))?;
                    v.extend(any_to_vec_f64(&np, my[i].bind(py))?);
                    Some(v)
                }
                None => None,
            };

            owned.push(Owned {
                idx: psm_indices[i],
                pb: pb_vec,
                py: py_vec,
                ob: ob_vec,
                oy: oy_vec,
                mz: mz_vec,
            });
        }

        // ---- Compute this block without the GIL ----
        let mut block_out: Vec<(usize, HashMap<String, f64>)> = py.detach(|| {
            owned
                .into_par_iter()
                .map(|it| {
                    if it.pb.len() != it.ob.len() || it.py.len() != it.oy.len() {
                        return (it.idx, HashMap::new());
                    }
                    if it.pb.is_empty() && it.py.is_empty() {
                        return (it.idx, HashMap::new());
                    }

                    // clip in log2 space
                    let tb: Vec<f64> = it.pb.into_iter().map(clip_min_f32).collect();
                    let ty: Vec<f64> = it.py.into_iter().map(clip_min_f32).collect();
                    let pb: Vec<f64> = it.ob.into_iter().map(clip_min_f32).collect();
                    let pyv: Vec<f64> = it.oy.into_iter().map(clip_min_f32).collect();

                    // unlog arrays
                    let tb_u: Vec<f64> = tb.iter().copied().map(pow2_unlog).collect();
                    let ty_u: Vec<f64> = ty.iter().copied().map(pow2_unlog).collect();
                    let pb_u: Vec<f64> = pb.iter().copied().map(pow2_unlog).collect();
                    let py_u: Vec<f64> = pyv.iter().copied().map(pow2_unlog).collect();

                    // abs diffs (log)
                    let mut abs_b: Vec<f64> = tb
                        .iter()
                        .zip(pb.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect();
                    let mut abs_y: Vec<f64> = ty
                        .iter()
                        .zip(pyv.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect();
                    let mut abs_all: Vec<f64> = Vec::with_capacity(abs_b.len() + abs_y.len());
                    abs_all.extend_from_slice(&abs_b);
                    abs_all.extend_from_slice(&abs_y);

                    // abs diffs (unlog)
                    let mut abs_b_u: Vec<f64> = tb_u
                        .iter()
                        .zip(pb_u.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect();
                    let mut abs_y_u: Vec<f64> = ty_u
                        .iter()
                        .zip(py_u.iter())
                        .map(|(a, b)| (a - b).abs())
                        .collect();
                    let mut abs_all_u: Vec<f64> = Vec::with_capacity(abs_b_u.len() + abs_y_u.len());
                    abs_all_u.extend_from_slice(&abs_b_u);
                    abs_all_u.extend_from_slice(&abs_y_u);

                    let (mean_abs_all, std_abs_all) = mean_std(&abs_all);
                    let (mean_abs_b, std_abs_b) = mean_std(&abs_b);
                    let (mean_abs_y, std_abs_y) = mean_std(&abs_y);

                    let (mean_abs_all_u, std_abs_all_u) = mean_std(&abs_all_u);
                    let (mean_abs_b_u, std_abs_b_u) = mean_std(&abs_b_u);
                    let (mean_abs_y_u, std_abs_y_u) = mean_std(&abs_y_u);

                    abs_all.sort_by(|a, b| a.total_cmp(b));
                    abs_b.sort_by(|a, b| a.total_cmp(b));
                    abs_y.sort_by(|a, b| a.total_cmp(b));

                    abs_all_u.sort_by(|a, b| a.total_cmp(b));
                    abs_b_u.sort_by(|a, b| a.total_cmp(b));
                    abs_y_u.sort_by(|a, b| a.total_cmp(b));

                    let min_abs_all = abs_all.first().copied().unwrap_or(f64::NAN);
                    let max_abs_all = abs_all.last().copied().unwrap_or(f64::NAN);
                    let q1_all = quantile_sorted(&abs_all, 0.25);
                    let q2_all = quantile_sorted(&abs_all, 0.5);
                    let q3_all = quantile_sorted(&abs_all, 0.75);

                    let min_abs_b = abs_b.first().copied().unwrap_or(f64::NAN);
                    let max_abs_b = abs_b.last().copied().unwrap_or(f64::NAN);
                    let q1_b = quantile_sorted(&abs_b, 0.25);
                    let q2_b = quantile_sorted(&abs_b, 0.5);
                    let q3_b = quantile_sorted(&abs_b, 0.75);

                    let min_abs_y = abs_y.first().copied().unwrap_or(f64::NAN);
                    let max_abs_y = abs_y.last().copied().unwrap_or(f64::NAN);
                    let q1_y = quantile_sorted(&abs_y, 0.25);
                    let q2_y = quantile_sorted(&abs_y, 0.5);
                    let q3_y = quantile_sorted(&abs_y, 0.75);

                    let min_abs_all_u = abs_all_u.first().copied().unwrap_or(f64::NAN);
                    let max_abs_all_u = abs_all_u.last().copied().unwrap_or(f64::NAN);
                    let q1_all_u = quantile_sorted(&abs_all_u, 0.25);
                    let q2_all_u = quantile_sorted(&abs_all_u, 0.5);
                    let q3_all_u = quantile_sorted(&abs_all_u, 0.75);

                    let min_abs_b_u = abs_b_u.first().copied().unwrap_or(f64::NAN);
                    let max_abs_b_u = abs_b_u.last().copied().unwrap_or(f64::NAN);
                    let q1_b_u = quantile_sorted(&abs_b_u, 0.25);
                    let q2_b_u = quantile_sorted(&abs_b_u, 0.5);
                    let q3_b_u = quantile_sorted(&abs_b_u, 0.75);

                    let min_abs_y_u = abs_y_u.first().copied().unwrap_or(f64::NAN);
                    let max_abs_y_u = abs_y_u.last().copied().unwrap_or(f64::NAN);
                    let q1_y_u = quantile_sorted(&abs_y_u, 0.25);
                    let q2_y_u = quantile_sorted(&abs_y_u, 0.5);
                    let q3_y_u = quantile_sorted(&abs_y_u, 0.75);

                    let spec_pearson_norm = pearson2(&tb, &ty, &pb, &pyv);
                    let ionb_pearson_norm = pearson(&tb, &pb);
                    let iony_pearson_norm = pearson(&ty, &pyv);

                    let spec_mse_norm = mse2(&tb, &ty, &pb, &pyv);
                    let ionb_mse_norm = mse(&tb, &pb);
                    let iony_mse_norm = mse(&ty, &pyv);

                    let dotprod_norm = dot2(&tb, &ty, &pb, &pyv);
                    let dotprod_ionb_norm = dot(&tb, &pb);
                    let dotprod_iony_norm = dot(&ty, &pyv);

                    let cos_norm = cosine2(&tb, &ty, &pb, &pyv);
                    let cos_ionb_norm = cosine_similarity(&tb, &pb);
                    let cos_iony_norm = cosine_similarity(&ty, &pyv);

                    let spec_pearson = pearson2(&tb_u, &ty_u, &pb_u, &py_u);
                    let ionb_pearson = pearson(&tb_u, &pb_u);
                    let iony_pearson = pearson(&ty_u, &py_u);

                    let mut t_all_u = Vec::with_capacity(tb_u.len() + ty_u.len());
                    t_all_u.extend_from_slice(&tb_u);
                    t_all_u.extend_from_slice(&ty_u);
                    let mut p_all_u = Vec::with_capacity(pb_u.len() + py_u.len());
                    p_all_u.extend_from_slice(&pb_u);
                    p_all_u.extend_from_slice(&py_u);

                    let spec_spearman = spearman(&t_all_u, &p_all_u);
                    let ionb_spearman = spearman(&tb_u, &pb_u);
                    let iony_spearman = spearman(&ty_u, &py_u);

                    let spec_mse = mse2(&tb_u, &ty_u, &pb_u, &py_u);
                    let ionb_mse = mse(&tb_u, &pb_u);
                    let iony_mse = mse(&ty_u, &py_u);

                    let dotprod = dot2(&tb_u, &ty_u, &pb_u, &py_u);
                    let dotprod_ionb = dot(&tb_u, &pb_u);
                    let dotprod_iony = dot(&ty_u, &py_u);

                    let cos = cosine2(&tb_u, &ty_u, &pb_u, &py_u);
                    let cos_ionb = cosine_similarity(&tb_u, &pb_u);
                    let cos_iony = cosine_similarity(&ty_u, &py_u);

                    // iontype min/max (unlog) like Python: 0 if b else 1 if y
                    let min_abs_diff_iontype = if min_abs_b_u <= min_abs_y_u { 0.0 } else { 1.0 };
                    let max_abs_diff_iontype = if max_abs_b_u >= max_abs_y_u { 0.0 } else { 1.0 };

                    let mut feats: HashMap<String, f64> = HashMap::with_capacity(90);

                    // log space
                    feats.insert(
                        "spec_pearson_norm".into(),
                        finite_or_zero(spec_pearson_norm),
                    );
                    feats.insert(
                        "ionb_pearson_norm".into(),
                        finite_or_zero(ionb_pearson_norm),
                    );
                    feats.insert(
                        "iony_pearson_norm".into(),
                        finite_or_zero(iony_pearson_norm),
                    );
                    feats.insert("spec_mse_norm".into(), finite_or_zero(spec_mse_norm));
                    feats.insert("ionb_mse_norm".into(), finite_or_zero(ionb_mse_norm));
                    feats.insert("iony_mse_norm".into(), finite_or_zero(iony_mse_norm));

                    feats.insert("min_abs_diff_norm".into(), finite_or_zero(min_abs_all));
                    feats.insert("max_abs_diff_norm".into(), finite_or_zero(max_abs_all));
                    feats.insert("abs_diff_Q1_norm".into(), finite_or_zero(q1_all));
                    feats.insert("abs_diff_Q2_norm".into(), finite_or_zero(q2_all));
                    feats.insert("abs_diff_Q3_norm".into(), finite_or_zero(q3_all));
                    feats.insert("mean_abs_diff_norm".into(), finite_or_zero(mean_abs_all));
                    feats.insert("std_abs_diff_norm".into(), finite_or_zero(std_abs_all));

                    feats.insert("ionb_min_abs_diff_norm".into(), finite_or_zero(min_abs_b));
                    feats.insert("ionb_max_abs_diff_norm".into(), finite_or_zero(max_abs_b));
                    feats.insert("ionb_abs_diff_Q1_norm".into(), finite_or_zero(q1_b));
                    feats.insert("ionb_abs_diff_Q2_norm".into(), finite_or_zero(q2_b));
                    feats.insert("ionb_abs_diff_Q3_norm".into(), finite_or_zero(q3_b));
                    feats.insert("ionb_mean_abs_diff_norm".into(), finite_or_zero(mean_abs_b));
                    feats.insert("ionb_std_abs_diff_norm".into(), finite_or_zero(std_abs_b));

                    feats.insert("iony_min_abs_diff_norm".into(), finite_or_zero(min_abs_y));
                    feats.insert("iony_max_abs_diff_norm".into(), finite_or_zero(max_abs_y));
                    feats.insert("iony_abs_diff_Q1_norm".into(), finite_or_zero(q1_y));
                    feats.insert("iony_abs_diff_Q2_norm".into(), finite_or_zero(q2_y));
                    feats.insert("iony_abs_diff_Q3_norm".into(), finite_or_zero(q3_y));
                    feats.insert("iony_mean_abs_diff_norm".into(), finite_or_zero(mean_abs_y));
                    feats.insert("iony_std_abs_diff_norm".into(), finite_or_zero(std_abs_y));

                    feats.insert("dotprod_norm".into(), finite_or_zero(dotprod_norm));
                    feats.insert(
                        "dotprod_ionb_norm".into(),
                        finite_or_zero(dotprod_ionb_norm),
                    );
                    feats.insert(
                        "dotprod_iony_norm".into(),
                        finite_or_zero(dotprod_iony_norm),
                    );
                    feats.insert("cos_norm".into(), finite_or_zero(cos_norm));
                    feats.insert("cos_ionb_norm".into(), finite_or_zero(cos_ionb_norm));
                    feats.insert("cos_iony_norm".into(), finite_or_zero(cos_iony_norm));

                    // normal space
                    feats.insert("spec_pearson".into(), finite_or_zero(spec_pearson));
                    feats.insert("ionb_pearson".into(), finite_or_zero(ionb_pearson));
                    feats.insert("iony_pearson".into(), finite_or_zero(iony_pearson));
                    feats.insert("spec_spearman".into(), finite_or_zero(spec_spearman));
                    feats.insert("ionb_spearman".into(), finite_or_zero(ionb_spearman));
                    feats.insert("iony_spearman".into(), finite_or_zero(iony_spearman));
                    feats.insert("spec_mse".into(), finite_or_zero(spec_mse));
                    feats.insert("ionb_mse".into(), finite_or_zero(ionb_mse));
                    feats.insert("iony_mse".into(), finite_or_zero(iony_mse));

                    feats.insert("min_abs_diff_iontype".into(), min_abs_diff_iontype);
                    feats.insert("max_abs_diff_iontype".into(), max_abs_diff_iontype);

                    feats.insert("min_abs_diff".into(), finite_or_zero(min_abs_all_u));
                    feats.insert("max_abs_diff".into(), finite_or_zero(max_abs_all_u));
                    feats.insert("abs_diff_Q1".into(), finite_or_zero(q1_all_u));
                    feats.insert("abs_diff_Q2".into(), finite_or_zero(q2_all_u));
                    feats.insert("abs_diff_Q3".into(), finite_or_zero(q3_all_u));
                    feats.insert("mean_abs_diff".into(), finite_or_zero(mean_abs_all_u));
                    feats.insert("std_abs_diff".into(), finite_or_zero(std_abs_all_u));

                    feats.insert("ionb_min_abs_diff".into(), finite_or_zero(min_abs_b_u));
                    feats.insert("ionb_max_abs_diff".into(), finite_or_zero(max_abs_b_u));
                    feats.insert("ionb_abs_diff_Q1".into(), finite_or_zero(q1_b_u));
                    feats.insert("ionb_abs_diff_Q2".into(), finite_or_zero(q2_b_u));
                    feats.insert("ionb_abs_diff_Q3".into(), finite_or_zero(q3_b_u));
                    feats.insert("ionb_mean_abs_diff".into(), finite_or_zero(mean_abs_b_u));
                    feats.insert("ionb_std_abs_diff".into(), finite_or_zero(std_abs_b_u));

                    feats.insert("iony_min_abs_diff".into(), finite_or_zero(min_abs_y_u));
                    feats.insert("iony_max_abs_diff".into(), finite_or_zero(max_abs_y_u));
                    feats.insert("iony_abs_diff_Q1".into(), finite_or_zero(q1_y_u));
                    feats.insert("iony_abs_diff_Q2".into(), finite_or_zero(q2_y_u));
                    feats.insert("iony_abs_diff_Q3".into(), finite_or_zero(q3_y_u));
                    feats.insert("iony_mean_abs_diff".into(), finite_or_zero(mean_abs_y_u));
                    feats.insert("iony_std_abs_diff".into(), finite_or_zero(std_abs_y_u));

                    feats.insert("dotprod".into(), finite_or_zero(dotprod));
                    feats.insert("dotprod_ionb".into(), finite_or_zero(dotprod_ionb));
                    feats.insert("dotprod_iony".into(), finite_or_zero(dotprod_iony));
                    feats.insert("cos".into(), finite_or_zero(cos));
                    feats.insert("cos_ionb".into(), finite_or_zero(cos_ionb));
                    feats.insert("cos_iony".into(), finite_or_zero(cos_iony));

                    add_similarity_features(
                        &mut feats, &tb, &ty, &pb, &pyv, &tb_u, &ty_u, &pb_u, &py_u,
                        &t_all_u, &p_all_u, it.mz.as_deref(),
                    );

                    (it.idx, feats)
                })
                .collect::<Vec<_>>()
        });

        out.append(&mut block_out);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from the original NumPy implementation in mwgard/ms2rescore@7f3d5df
    // (`_spectra_angle_calc`, `_spectrast_match`, `_weighted_dot_product`, `_nist_ms_match`).
    const MZ: [f64; 6] = [175.119, 288.203, 401.287, 147.113, 260.197, 373.281];
    const X: [f64; 6] = [0.20, 0.05, 0.10, 0.30, 0.15, 0.20];
    const Y: [f64; 6] = [0.25, 0.04, 0.12, 0.28, 0.10, 0.21];

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn test_spectral_angle() {
        assert_close(spectral_angle(&X, &Y), 0.8921315119149698);
        let lx: Vec<f64> = X.iter().map(|v| (v + 0.001).log2()).collect();
        let ly: Vec<f64> = Y.iter().map(|v| (v + 0.001).log2()).collect();
        assert_close(spectral_angle(&lx, &ly), 0.9334764009012824);
        assert!((spectral_angle(&X, &X) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_spectrast() {
        assert_close(spectrast(&X, &Y), 0.9715639810426541);
        assert!(spectrast(&[0.0, 0.0], &[1.0, 2.0]).is_nan());
    }

    #[test]
    fn test_weighted_dotprod() {
        assert_close(weighted_dotprod(&MZ, &X, &Y), 0.9955546637147498);
        assert!(weighted_dotprod(&MZ[..2], &X, &Y).is_nan());
    }

    #[test]
    fn test_nist_match_factor() {
        assert_close(nist_match_factor(&MZ, &X, &Y), 0.8080843854919356);
        let px: Vec<f64> = X.iter().copied().map(positive_log).collect();
        let py: Vec<f64> = Y.iter().copied().map(positive_log).collect();
        assert_close(nist_match_factor(&MZ, &px, &py), 0.8151864674677505);
        // identical spectra: DPC = 1, RPP = (n - 1) / n, as the RPP sum over the n - 1
        // consecutive pairs is divided by the number of peaks (reference implementation)
        assert_close(nist_match_factor(&MZ, &X, &X), (6.0 + 5.0) / 12.0);
    }

    #[test]
    fn test_nist_match_factor_missing_peaks() {
        // Peaks missing in either spectrum are excluded from the ratio of peak pairs
        // (the NumPy reference raised a broadcasting error for this input).
        let x = [1.0, 2.0, 0.0, 4.0, 5.0];
        let y = [1.0, 0.0, 3.0, 4.0, 5.0];
        let mz = [100.0, 200.0, 300.0, 400.0, 500.0];
        let mf = nist_match_factor(&mz, &x, &y);
        assert!(mf.is_finite() && mf > 0.0 && mf < 1.0);
        // fewer than two common peaks -> 0
        assert_close(nist_match_factor(&mz, &[1.0, 0.0, 0.0, 0.0, 0.0], &y), 0.0);
    }
}
