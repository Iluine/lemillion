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

/// Ridge regression: W_out = Y * H^T * (H * H^T + lambda * I)^{-1}
///
/// - h: [state_dim, T] (state matrix, each column is a state at time t)
/// - y: [output_dim, T] (target matrix)
/// - Returns W_out: [output_dim, state_dim]
pub fn ridge_regression(h: &Array2<f64>, y: &Array2<f64>, lambda: f64) -> Result<Array2<f64>> {
    let state_dim = h.nrows();

    // A = H * H^T + lambda * I  [state_dim, state_dim]
    let hht = h.dot(&h.t());
    let mut a = hht;
    for i in 0..state_dim {
        a[[i, i]] += lambda;
    }

    // B = Y * H^T  [output_dim, state_dim]
    let yht = y.dot(&h.t());

    // Solve: A * W_out^T = B^T via Cholesky
    let a_faer = ndarray_to_faer(&a);
    let b_t = yht.t().to_owned();
    let b_faer = ndarray_to_faer(&b_t);

    let llt = a_faer.llt(Side::Lower);
    if llt.is_err() {
        bail!("Cholesky: matrix not positive-definite");
    }
    let llt = llt.unwrap();
    let x_faer = llt.solve(&b_faer);

    // x_faer is [state_dim, output_dim], transpose to get [output_dim, state_dim]
    let w_out_t = faer_to_ndarray(&x_faer);
    Ok(w_out_t.t().to_owned())
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
}
