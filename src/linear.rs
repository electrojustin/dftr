use std::iter::zip;
use std::ops;

use num::complex::Complex;

#[derive(Debug, Clone)]
pub struct Vector(Vec<Complex<f64>>);

impl Vector {
    // Compare two vectors, returning true if each element is equal to within the specified
    // tolerance.
    pub fn compare(&self, other: &Vector, tolerance: f64) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        for i in 0..self.0.len() {
            if (self.0[i] - other.0[i]).norm_sqr() > tolerance {
                return false;
            }
        }
        true
    }

    // Compute dot product of two vectors.
    pub fn dot(&self, other: &Vector) -> Complex<f64> {
        zip(self.0.iter(), other.0.iter())
            .map(|(x, y)| -> Complex<f64> { x * y })
            .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x })
    }

    // Project one vector onto another, i.e. A.B / B.B
    pub fn proj(&self, other: &Vector) -> Complex<f64> {
        other.dot(self) / other.dot(other)
    }

    // L2 (Euclidean) norm.
    pub fn l2(&self) -> Complex<f64> {
        self.dot(self).sqrt()
    }

    // Special case of compare() for checking if a vector is the zero vector.
    pub fn is_zero(&self, zero_threshold: f64) -> bool {
        for element in self.0.iter() {
            if element.norm_sqr() > zero_threshold {
                return false;
            }
        }
        true
    }
}

impl From<Vec<Complex<f64>>> for Vector {
    fn from(val: Vec<Complex<f64>>) -> Self {
        Self(val)
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<Complex<f64>>,
    width: usize,
    height: usize,
}

impl Matrix {
    // Construct a matrix from its rows.
    pub fn from_row_vecs(rows: Vec<Vector>) -> Self {
        assert!(
            rows.len() != 0,
            "Cannot initialize Matrix with no row vector!"
        );
        let height = rows.len();
        let width = rows[0].0.len();
        let mut data: Vec<Complex<f64>> = Vec::with_capacity(width * height);
        for mut row in rows.into_iter() {
            assert!(
                row.0.len() == width,
                "Cannot initialize Matrix with rows of different lengths!"
            );
            data.append(&mut row.0);
        }

        Matrix {
            data,
            width,
            height,
        }
    }

    // Construct a diagonal matrix with the specified entries on its diagonal.
    pub fn from_diagonal(diag: Vec<Complex<f64>>) -> Self {
        let width = diag.len();
        let mut data: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); width * width];
        for i in 0..diag.len() {
            data[i * width + i] = diag[i];
        }

        Matrix {
            data,
            width,
            height: width,
        }
    }

    // Special case of a diagonal matrix where the entries are all the same value.
    pub fn identity(x: Complex<f64>, dim: usize) -> Self {
        Matrix::from_diagonal(vec![x; dim])
    }

    // Row view of the matrix.
    pub fn to_row_vecs(&self) -> Vec<Vector> {
        (0..self.height)
            .map(|y| -> Vector {
                self.data[y * self.width..(y + 1) * self.width]
                    .to_vec()
                    .into()
            })
            .collect()
    }

    // Column view of the matrix.
    pub fn to_col_vecs(&self) -> Vec<Vector> {
        (0..self.width)
            .map(|x| -> Vector {
                (0..self.height)
                    .map(|y| -> Complex<f64> { self.data[y * self.width + x] })
                    .collect::<Vec<_>>()
                    .into()
            })
            .collect()
    }

    // Compute the transpose of the matrix.
    pub fn transpose(self) -> Matrix {
        Matrix::from_row_vecs(self.to_col_vecs())
    }

    // Compare two matrices, returning true if each element is equal to within the specified
    // tolerance.
    pub fn compare(&self, other: &Matrix, tolerance: f64) -> bool {
        if self.width != other.width || self.height != other.height {
            return false;
        }
        for y in 0..self.height {
            for x in 0..self.width {
                if (self.data[y * self.width + x] - other.data[y * self.width + x]).norm_sqr()
                    > tolerance
                {
                    return false;
                }
            }
        }
        true
    }

    // Orthonormal decomposition via the Gram-Schmidt algorithm.
    pub fn qr_decomp(self) -> (Matrix, Matrix) {
        assert_eq!(self.width, self.height);

        let cols = self.to_col_vecs();
        let mut triangle_matrix: Vec<Vector> = Vec::with_capacity(self.width);
        let mut orthogonal_matrix: Vec<Vector> = Vec::with_capacity(self.width);
        for mut col in cols.into_iter() {
            let mut triangle_row: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.width];
            let mut i: usize = 0;
            for basis in orthogonal_matrix.iter() {
                let coefficient = col.proj(basis);
                col = col - coefficient * basis.clone();
                triangle_row[i] = coefficient;
                i += 1;
            }
            let coefficient = col.l2();
            triangle_row[i] = coefficient;
            let new_basis = (Complex::new(1.0, 0.0) / coefficient) * col;
            triangle_matrix.push(triangle_row.into());
            orthogonal_matrix.push(new_basis);
        }

        (
            Matrix::from_row_vecs(orthogonal_matrix).transpose(),
            Matrix::from_row_vecs(triangle_matrix).transpose(),
        )
    }

    // Returns true if a matrix is upper triangular, that is, the lower left triangle of the matrix
    // contains entries that are all zero to within the specified tolerance.
    pub fn is_triangular(&self, tolerance: f64) -> bool {
        if self.width != self.height {
            return false;
        }

        for y in 0..self.height {
            for x in 0..y {
                if self.data[y * self.width + x].norm_sqr() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    // Basic Gaussian elimination algorithm.
    fn row_echelon(&self, augmentation: &Matrix, zero_threshold: f64) -> (Matrix, Matrix) {
        assert_eq!(self.height, augmentation.height);

        let mut lhs_rows = self.to_row_vecs();
        let mut rhs_rows = augmentation.to_row_vecs();
        for y in 0..self.height {
            let offset = 'find_offset: {
                for x in y..self.width {
                    for y2 in y..self.height {
                        if lhs_rows[y2].0[x].norm_sqr() > zero_threshold {
                            // Swap rows so the current row has the furthest left non-zero element.
                            if y2 != y {
                                lhs_rows.swap(y, y2);
                                rhs_rows.swap(y, y2);
                            }
                            break 'find_offset Some(x);
                        }
                    }
                }
                None
            };
            if let Some(x) = offset {
                let coefficient = Complex::new(1.0, 0.0) / lhs_rows[y].0[x];
                lhs_rows[y] = coefficient * lhs_rows[y].clone();
                rhs_rows[y] = coefficient * rhs_rows[y].clone();
                for y2 in (y + 1)..self.height {
                    let coefficient = lhs_rows[y2].0[x];
                    lhs_rows[y2] = lhs_rows[y2].clone() - coefficient * lhs_rows[y].clone();
                    rhs_rows[y2] = rhs_rows[y2].clone() - coefficient * rhs_rows[y].clone();
                }
            }
        }

        (
            Matrix::from_row_vecs(lhs_rows),
            Matrix::from_row_vecs(rhs_rows),
        )
    }

    // Compute the inverse of the matrix using Gaussian elimination.
    pub fn inverse(self, zero_threshold: f64) -> Matrix {
        assert_eq!(self.width, self.height);

        let height = self.height;
        let (lhs, rhs) = self.row_echelon(
            &Matrix::identity(Complex::new(1.0, 0.0), self.width),
            zero_threshold,
        );

        let mut lhs_rows = lhs.to_row_vecs();
        let mut rhs_rows = rhs.to_row_vecs();
        for i in 0..height {
            let y = height - 1 - i;
            for y2 in 0..y {
                let coefficient = lhs_rows[y2].0[y];
                lhs_rows[y2] = lhs_rows[y2].clone() - coefficient * lhs_rows[y].clone();
                rhs_rows[y2] = rhs_rows[y2].clone() - coefficient * rhs_rows[y].clone();
            }
        }

        Matrix::from_row_vecs(rhs_rows)
    }

    // Compute the kernel (null space) of the matrix using Gaussian elimination.
    pub fn kernel(&self, zero_threshold: f64) -> Vec<Vector> {
        let transpose_self = self.clone().transpose();
        let (lhs, rhs) = transpose_self.row_echelon(
            &Matrix::identity(Complex::new(1.0, 0.0), self.width),
            zero_threshold,
        );
        let lhs_rows = lhs.to_row_vecs();
        let rhs_rows = rhs.to_row_vecs();
        zip(lhs_rows.into_iter(), rhs_rows.into_iter())
            .filter_map(|(l, r)| {
                if l.is_zero(zero_threshold) {
                    Some(r)
                } else {
                    None
                }
            })
            .collect()
    }

    // Implementation of Eigendecomposition by QR algorithm.
    pub fn eigen(&self, zero_threshold: f64, max_iters: usize) -> (Vec<Complex<f64>>, Vec<Vector>) {
        assert_eq!(self.width, self.height);

        let mut similar_matrix = self.clone();
        let mut i = 0;
        loop {
            if i == max_iters {
                return (vec![], vec![]);
            }
            // The eigenvalues of triangular matrices are the diagonal.
            if similar_matrix.is_triangular(zero_threshold) {
                break;
            }

            // Iteratively transform the given matrix into a triangular similar matrix.
            // For a given matrix A = QR, B = RQ is similar to A, which means it has the same
            // eigenvalues.
            let (q, r) = similar_matrix.clone().qr_decomp();
            similar_matrix = r * q;
            i += 1;
        }

        let mut eigenvalues: Vec<Complex<f64>> = Vec::new();
        for i in 0..self.width {
            let diag_val = similar_matrix.data[i * self.width + i];
            if diag_val.norm_sqr() > zero_threshold {
                eigenvalues.push(diag_val);
            }
        }

        // Now that we have the eigenvalues for the matrix, we can find the eigenvectors by solving
        // the equation (A - lambda * I) * x = 0, where A is the original matrix, lambda is an
        // eigenvalue. The non-zero vectors x which satisfy this equation, i.e. the null space of
        // (A - lambda * I), are the eigenvectors.
        let mut eigenvectors: Vec<Vector> = Vec::new();
        for eigenval in eigenvalues.iter() {
            let char_matrix = self.clone() - Matrix::identity(*eigenval, self.width);
            eigenvectors.append(&mut char_matrix.kernel(zero_threshold));
        }

        (eigenvalues, eigenvectors)
    }
}

impl ops::Sub<Vector> for Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        assert_eq!(self.0.len(), rhs.0.len());
        zip(self.0.into_iter(), rhs.0.into_iter())
            .map(|(left, right)| -> Complex<f64> { left - right })
            .collect::<Vec<_>>()
            .into()
    }
}

impl ops::Add<Vector> for Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Self::Output {
        assert_eq!(self.0.len(), rhs.0.len());
        zip(self.0.into_iter(), rhs.0.into_iter())
            .map(|(left, right)| -> Complex<f64> { left + right })
            .collect::<Vec<_>>()
            .into()
    }
}

impl ops::Mul<Vector> for Complex<f64> {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        rhs.0
            .into_iter()
            .map(|x| -> Complex<f64> { self * x })
            .collect::<Vec<_>>()
            .into()
    }
}

impl ops::Mul<Vector> for Matrix {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        self.to_row_vecs()
            .into_iter()
            .map(|row| -> Complex<f64> { row.dot(&rhs) })
            .collect::<Vec<_>>()
            .into()
    }
}

impl ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Self::Output {
        assert_eq!(self.width, rhs.height);
        assert_eq!(self.height, rhs.width);

        let lhs = self.to_row_vecs();
        let rhs = rhs.to_col_vecs();
        let mut ret: Vec<Vector> = Vec::with_capacity(lhs.len());
        for row in lhs.into_iter() {
            let mut out_row: Vec<Complex<f64>> = Vec::with_capacity(rhs.len());
            for col in rhs.iter() {
                out_row.push(col.dot(&row))
            }
            ret.push(out_row.into());
        }
        Matrix::from_row_vecs(ret)
    }
}

impl ops::Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Matrix) -> Self::Output {
        assert_eq!(self.width, rhs.width);
        assert_eq!(self.height, rhs.height);

        let width = self.width;
        let height = self.height;
        Self {
            data: zip(self.data.into_iter(), rhs.data.into_iter())
                .map(|(x, y)| -> Complex<f64> { x - y })
                .collect(),
            width,
            height,
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_vector_arithmetic() {
        let test1: Vector = vec![Complex::new(2.0, 0.0), Complex::new(2.0, 0.0)].into();
        let test2: Vector = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)].into();
        let sub = test1.clone() - test2.clone();
        assert!(
            sub.compare(&test2, 1E-10),
            "Error subtracting vectors!\nExpected: {:?}\nActual: {:?}",
            test2,
            sub
        );
        let add = sub + test2;
        assert!(
            add.compare(&test1, 1E-10),
            "Error adding vectors!\nExpected: {:?}\nActual: {:?}",
            test1,
            add
        );
        let dot = test1.dot(&test1);
        assert!(
            (dot - Complex::new(8.0, 0.0)).norm_sqr() <= 1E-10,
            "Error computing dot product!\nExpected: {:?}\nActual: {:?}",
            Complex::new(8.0, 0.0),
            dot
        );
    }

    #[test]
    fn test_transpose() {
        let input = Matrix::from_row_vecs(vec![
            vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)].into(),
            vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)].into(),
        ]);
        let expected = Matrix::from_row_vecs(vec![
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)].into(),
            vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)].into(),
        ]);
        let actual = input.transpose();
        assert!(
            actual.compare(&expected, 1E-10),
            "Error transposing matrix!\nExpected: {:?}\nActual: {:?}",
            expected,
            actual
        );
    }

    #[test]
    fn test_matrix_arithmetic() {
        let lhs = Matrix::from_row_vecs(vec![
            vec![Complex::new(0.77551606, 0.0), Complex::new(0.19238363, 0.0)].into(),
            vec![Complex::new(0.1340687, 0.0), Complex::new(0.53814676, 0.0)].into(),
        ]);
        let rhs: Vector = vec![Complex::new(0.28871166, 0.0), Complex::new(0.90987051, 0.0)].into();
        let expected: Vector =
            vec![Complex::new(0.39894472, 0.0), Complex::new(0.52835106, 0.0)].into();
        let actual = lhs.clone() * rhs;
        assert!(
            actual.compare(&expected, 1E-10),
            "Error in matrix-vector multiplication!\nExpected: {:?}\nActual: {:?}",
            expected,
            actual
        );

        let rhs = Matrix::from_row_vecs(vec![
            vec![Complex::new(0.98896077, 0.0), Complex::new(0.96173138, 0.0)].into(),
            vec![Complex::new(0.49584558, 0.0), Complex::new(0.67387721, 0.0)].into(),
        ]);
        let expected = Matrix::from_row_vecs(vec![
            vec![Complex::new(0.86234754, 0.0), Complex::new(0.87548108, 0.0)].into(),
            vec![Complex::new(0.39942637, 0.0), Complex::new(0.49158291, 0.0)].into(),
        ]);
        let actual = lhs * rhs;
        assert!(
            actual.compare(&expected, 1E-10),
            "Error in matix multiplication!\nExpected: {:?}\nActual: {:?}",
            expected,
            actual
        );
    }

    #[test]
    fn test_qr() {
        let lhs = Matrix::from_row_vecs(vec![
            vec![Complex::new(0.77551606, 0.0), Complex::new(0.19238363, 0.0)].into(),
            vec![Complex::new(0.1340687, 0.0), Complex::new(0.53814676, 0.0)].into(),
        ]);
        let (actual_q, actual_r) = lhs.clone().qr_decomp();
        let q_cols = actual_q.to_col_vecs();
        let q_dot = q_cols[0].dot(&q_cols[1]);
        assert!(
            q_dot.norm_sqr() < 1E-10,
            "Error in QR decomp! Q matrix is not orthogonal. Q matrix: {:?}",
            actual_q
        );
        assert!(
            (q_cols[0].l2().norm_sqr() - 1.0).abs() < 1E-10
                || (q_cols[1].l2().norm_sqr() - 1.0).abs() < 1E-10,
            "Error in QR decomp! Q matrix is not normalized. Q matrix: {:?}",
            actual_q
        );
        assert!(
            actual_r.is_triangular(1E-10),
            "Error in QR decomp! R matrix is not triangular. R matrix: {:?}",
            actual_r
        );
        let reconstructed = actual_q * actual_r;
        assert!(
            lhs.compare(&reconstructed, 1E-10),
            "Error in QR decomp! A != QR.\nExpected: {:?}\nActual: {:?}",
            lhs,
            reconstructed,
        );
    }

    #[test]
    fn test_row_echelon() {
        let lhs = Matrix::from_row_vecs(vec![
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(9.0, 0.0),
            ]
            .into(),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
                Complex::new(1.0, 0.0),
            ]
            .into(),
            vec![
                Complex::new(3.0, 0.0),
                Complex::new(11.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(35.0, 0.0),
            ]
            .into(),
        ]);
        let rhs = Matrix::from_row_vecs(vec![
            vec![Complex::new(0.0, 0.0)].into(),
            vec![Complex::new(0.0, 0.0)].into(),
            vec![Complex::new(0.0, 0.0)].into(),
        ]);
        let expected = Matrix::from_row_vecs(vec![
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(9.0, 0.0),
            ]
            .into(),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(4.0, 0.0),
            ]
            .into(),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ]
            .into(),
        ]);
        let (actual, _) = lhs.row_echelon(&rhs, 1E-10);
        assert!(
            actual.compare(&expected, 1E-10),
            "Error in row echelon!\nExpected: {:?}\nActual: {:?}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_inverse() {
        let input = Matrix::from_row_vecs(vec![
            vec![Complex::new(-1.0, 0.0), Complex::new(1.5, 0.0)].into(),
            vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)].into(),
        ]);
        let expected = Matrix::from_row_vecs(vec![
            vec![Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)].into(),
            vec![Complex::new(2.0, 0.0), Complex::new(2.0, 0.0)].into(),
        ]);
        let actual = input.inverse(1E-10);
        assert!(
            actual.compare(&expected, 1E-10),
            "Error in inverse!\nExpected: {:?}\nActual: {:?}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_kernel() {
        let input = Matrix::from_row_vecs(vec![
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-3.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(-8.0, 0.0),
            ]
            .into(),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-1.0, 0.0),
                Complex::new(4.0, 0.0),
            ]
            .into(),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(-9.0, 0.0),
            ]
            .into(),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ]
            .into(),
        ]);
        let expected = vec![
            Vector::from(vec![
                Complex::new(3.0, 0.0),
                Complex::new(-5.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ]),
            Vector::from(vec![
                Complex::new(-2.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-7.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ]),
            Vector::from(vec![
                Complex::new(8.0, 0.0),
                Complex::new(-4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(9.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ]),
        ];
        let actual = input.kernel(1E-10);
        let mut is_correct = expected.len() == actual.len();
        if is_correct {
            for i in 0..expected.len() {
                if !actual[i].compare(&expected[i], 1E-10) {
                    is_correct = false;
                    break;
                }
            }
        }
        assert!(
            is_correct,
            "Error in kernel space!\nExpected: {:?}\nActual: {:?}",
            expected, actual
        );
    }

    #[test]
    fn test_eigen() {
        let input = Matrix::from_row_vecs(vec![
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)].into(),
            vec![Complex::new(1.0, 0.0), Complex::new(3.0, 0.0)].into(),
        ]);
        let expected_eigenvals = vec![Complex::new(3.0, 0.0), Complex::new(1.0, 0.0)];
        let expected_eigenvecs = vec![
            Vector::from(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]),
            Vector::from(vec![Complex::new(-2.0, 0.0), Complex::new(1.0, 0.0)]),
        ];
        let (actual_eigenvals, actual_eigenvecs) = input.eigen(1E-10, 1000);
        let is_correct = Vector::from(expected_eigenvals.clone())
            .compare(&Vector::from(actual_eigenvals.clone()), 1E-10);
        assert!(
            is_correct,
            "Error in eigen: incorrect eigenvalues!\nExpected: {:?}\nActual: {:?}",
            expected_eigenvals, actual_eigenvals
        );
        let is_correct = Matrix::from_row_vecs(expected_eigenvecs.clone())
            .compare(&Matrix::from_row_vecs(actual_eigenvecs.clone()), 1E-10);
        assert!(
            is_correct,
            "Error in eigen: incorrect eigenvecs!\nExpected: {:?}\nActual: {:?}",
            expected_eigenvecs, actual_eigenvecs
        );
    }
}
