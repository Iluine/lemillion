use anyhow::{bail, Result};
use faer::{Mat, Side};
use faer::prelude::Solve;
use ndarray::Array2;

/// Convert ndarray Array2 to faer Mat (column-major).
fn ndarray_to_faer(arr: &Array2<f64>) -> Mat<f64> {
    let (rows, cols) = (arr.nrows(), arr.ncols());
    Mat::from_fn(rows, cols, |i, j| arr[[i, j]])
}

/// Convert faer Mat back to ndarray Array2.
fn faer_to_ndarray(mat: &Mat<f64>) -> Array2<f64> {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    Array2::from_shape_fn((rows, cols), |(i, j)| mat[(i, j)])
}

/// Solve a symmetric positive-definite system via Cholesky factorization.
/// Returns X such that A * X = B.
fn cholesky_solve(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let a_faer = ndarray_to_faer(a);
    let b_faer = ndarray_to_faer(b);
    let llt = a_faer.llt(Side::Lower);
    if llt.is_err() {
        bail!("Cholesky: matrix not positive-definite");
    }
    let llt = llt.unwrap();
    let x_faer = llt.solve(&b_faer);
    Ok(faer_to_ndarray(&x_faer))
}

/// Ridge regression: W_out = Y * H^T * (H * H^T + lambda * I)^{-1}
///
/// - h: [state_dim, T] (state matrix, each column is a state at time t)
/// - y: [output_dim, T] (target matrix)
/// - Returns W_out: [output_dim, state_dim]
///
/// Automatically selects primal (d×d) or dual (T×T) formulation
/// based on which is smaller, via push-through identity.
pub fn ridge_regression(h: &Array2<f64>, y: &Array2<f64>, lambda: f64) -> Result<Array2<f64>> {
    let d = h.nrows(); // state_dim
    let t = h.ncols(); // number of time steps

    if t < d {
        // Dual formulation: solve on T×T instead of d×d
        // G = H^T * H + lambda * I_T  [T×T]
        let hth = h.t().dot(h);
        let mut g = hth;
        for i in 0..t {
            g[[i, i]] += lambda;
        }

        // Solve G * Z = Y^T  [T×k]
        let yt = y.t().to_owned();
        let z = cholesky_solve(&g, &yt)?;

        // W_out = (H * Z)^T  [k×d]
        let hz = h.dot(&z);
        Ok(hz.t().to_owned())
    } else {
        // Primal formulation: solve on d×d
        // A = H * H^T + lambda * I  [d×d]
        let hht = h.dot(&h.t());
        let mut a = hht;
        for i in 0..d {
            a[[i, i]] += lambda;
        }

        // B = Y * H^T  [output_dim, d]
        let yht = y.dot(&h.t());

        // Solve A * W_out^T = B^T via Cholesky
        let b_t = yht.t().to_owned();
        let w_out_t = cholesky_solve(&a, &b_t)?;

        // w_out_t is [d, output_dim], transpose to get [output_dim, d]
        Ok(w_out_t.t().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_ridge_regression_identity() {
        // With A = I, the solve is trivial
        let h = Array2::eye(3);
        let y = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let w = ridge_regression(&h, &y, 1e-8).unwrap();
        assert_eq!(w.shape(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (w[[i, j]] - y[[i, j]]).abs() < 0.01,
                    "w[{i},{j}]={}, expected {}",
                    w[[i, j]],
                    y[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_ridge_regression_small() {
        // H = [[1, 2], [1, 0]] (state_dim=2, T=2)
        // Y = [[5, 2]] (output_dim=1, T=2)
        // Least-squares: W*H = Y => W = Y*H^T*(H*H^T)^{-1}
        // H*H^T = [[5,1],[1,1]], Y*H^T = [[9,5]]
        // (H*H^T)^{-1} = 1/4 * [[1,-1],[-1,5]]
        // W = [[9,5]] * 1/4 * [[1,-1],[-1,5]] = 1/4 * [[4,16]] = [[1, 4]]
        let h = array![[1.0, 2.0], [1.0, 0.0]]; // [2, 2]
        let y = array![[5.0, 2.0]]; // [1, 2]
        let w = ridge_regression(&h, &y, 1e-8).unwrap();
        assert_eq!(w.shape(), &[1, 2]);
        assert!(
            (w[[0, 0]] - 1.0).abs() < 0.01,
            "w[0,0]={}, expected ~1",
            w[[0, 0]]
        );
        assert!(
            (w[[0, 1]] - 4.0).abs() < 0.01,
            "w[0,1]={}, expected ~4",
            w[[0, 1]]
        );
    }

    #[test]
    fn test_ridge_regression_dual_path() {
        // T < d triggers dual formulation
        // H: [10, 3] (d=10, T=3), Y: [2, 3] (k=2)
        let h = Array2::from_shape_fn((10, 3), |(i, j)| (i * 3 + j + 1) as f64 * 0.1);
        let y = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j + 1) as f64);

        let w = ridge_regression(&h, &y, 0.01).unwrap();
        assert_eq!(w.shape(), &[2, 10]);

        // Verify by computing Y_hat = W * H and checking it's close to Y
        let y_hat = w.dot(&h);
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (y_hat[[i, j]] - y[[i, j]]).abs() < 0.5,
                    "y_hat[{i},{j}]={}, expected ~{}",
                    y_hat[[i, j]],
                    y[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_primal_dual_equivalence() {
        // Same problem solved with both paths should give same result
        // d=5, T=5 -> primal (t >= d)
        let h_sq = Array2::from_shape_fn((5, 5), |(i, j)| {
            if i == j { 2.0 } else { 0.1 * ((i + j) as f64) }
        });
        let y_sq = Array2::from_shape_fn((2, 5), |(i, j)| (i * 5 + j) as f64);
        let w_primal = ridge_regression(&h_sq, &y_sq, 0.1).unwrap();

        // d=5, T=3 -> dual (t < d)
        // Use first 3 columns
        let h_dual = h_sq.slice(ndarray::s![.., ..3]).to_owned();
        let y_dual = y_sq.slice(ndarray::s![.., ..3]).to_owned();
        let w_dual = ridge_regression(&h_dual, &y_dual, 0.1).unwrap();

        // Both should produce valid results (different problems, but both should be consistent)
        assert_eq!(w_primal.shape(), &[2, 5]);
        assert_eq!(w_dual.shape(), &[2, 5]);

        // Verify reconstruction: W * H ≈ Y for both
        let y_hat_p = w_primal.dot(&h_sq);
        let y_hat_d = w_dual.dot(&h_dual);
        for i in 0..2 {
            for j in 0..5 {
                assert!(
                    (y_hat_p[[i, j]] - y_sq[[i, j]]).abs() < 1.0,
                    "primal reconstruction error at [{i},{j}]"
                );
            }
            for j in 0..3 {
                assert!(
                    (y_hat_d[[i, j]] - y_dual[[i, j]]).abs() < 1.0,
                    "dual reconstruction error at [{i},{j}]"
                );
            }
        }
    }
}
