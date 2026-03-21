use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::iter::zip;
use std::ops;

use num::complex::Complex;

#[derive(Clone)]
pub struct Vector(pub Vec<Complex<f64>>);

impl Vector {
    // Compare two vectors, returning true if each element is equal to within the specified
    // rcond.
    pub fn compare(&self, other: &Vector, rcond: f64) -> bool {
        let max = self.max_norm().max(other.max_norm());
        if self.0.len() != other.0.len() {
            return false;
        }
        for i in 0..self.0.len() {
            if (self.0[i] - other.0[i]).norm_sqr() > rcond * max {
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
    pub fn is_zero(&self, tolerance: f64) -> bool {
        for element in self.0.iter() {
            if element.norm_sqr() > tolerance {
                return false;
            }
        }
        true
    }

    fn max_norm(&self) -> f64 {
        self.0
            .iter()
            .map(|x| -> f64 { x.norm_sqr() })
            .fold(f64::MIN, |acc, x| -> f64 { x.max(acc) })
    }
}

impl Debug for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Vector: [")?;
        for i in 0..self.0.len() {
            if i != self.0.len() - 1 {
                write!(f, "{}, ", self.0[i].re)?;
            } else {
                write!(f, "{}", self.0[i].re)?;
            }
        }
        write!(f, "]")
    }
}

impl From<Vec<Complex<f64>>> for Vector {
    fn from(val: Vec<Complex<f64>>) -> Self {
        Self(val)
    }
}

#[derive(Clone)]
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

    fn max_norm(&self) -> f64 {
        self.data
            .iter()
            .map(|x| -> f64 { x.norm_sqr() })
            .fold(f64::MIN, |acc, x| -> f64 { x.max(acc) })
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
    // rcond.
    pub fn compare(&self, other: &Matrix, rcond: f64) -> bool {
        let max = self.max_norm().max(other.max_norm());
        if self.width != other.width || self.height != other.height {
            return false;
        }
        for y in 0..self.height {
            for x in 0..self.width {
                if (self.data[y * self.width + x] - other.data[y * self.width + x]).norm_sqr()
                    > rcond * max
                {
                    return false;
                }
            }
        }
        true
    }

    // Returns true if the matrix is symmetric within the specified rcond.
    pub fn is_symmetric(&self, rcond: f64) -> bool {
        if self.width != self.height {
            return false;
        }

        let max = self.max_norm();
        for i in 0..self.width {
            for j in i..self.width {
                if (self.data[i * self.width + j] - self.data[j * self.width + i]).norm_sqr()
                    > rcond * max
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
    // contains entries that are all zero to within the specified rcond.
    pub fn is_triangular(&self, rcond: f64) -> bool {
        if self.width != self.height {
            return false;
        }

        let max = self.max_norm();
        for y in 0..self.height {
            for x in 0..y {
                if self.data[y * self.width + x].norm_sqr() > rcond * max {
                    return false;
                }
            }
        }
        true
    }

    // Basic Gaussian elimination algorithm.
    fn row_echelon(&self, augmentation: &Matrix, tolerance: f64) -> (Matrix, Matrix) {
        assert_eq!(self.height, augmentation.height);

        let mut lhs_rows = self.to_row_vecs();
        let mut rhs_rows = augmentation.to_row_vecs();
        for y in 0..self.height {
            let offset = 'find_offset: {
                for x in y..self.width {
                    for y2 in y..self.height {
                        if lhs_rows[y2].0[x].norm_sqr() > tolerance {
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
                for y2 in (y + 1)..self.height {
                    let coefficient = lhs_rows[y2].0[x] / lhs_rows[y].0[x];
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
    pub fn inverse(self, rcond: f64) -> Matrix {
        assert_eq!(self.width, self.height);
        let max = self.max_norm();

        let height = self.height;
        let (lhs, rhs) = self.row_echelon(
            &Matrix::identity(Complex::new(1.0, 0.0), self.width),
            rcond * max,
        );

        let mut lhs_rows = lhs.to_row_vecs();
        let mut rhs_rows = rhs.to_row_vecs();
        for i in 0..height {
            let y = height - 1 - i;
            for y2 in 0..y {
                let coefficient = lhs_rows[y2].0[y] / lhs_rows[y].0[y];
                lhs_rows[y2] = lhs_rows[y2].clone() - coefficient * lhs_rows[y].clone();
                rhs_rows[y2] = rhs_rows[y2].clone() - coefficient * rhs_rows[y].clone();
            }
        }

        for y in 0..height {
            let coefficient = Complex::new(1.0, 0.0) / lhs_rows[y].0[y];
            rhs_rows[y] = coefficient * rhs_rows[y].clone();
        }

        Matrix::from_row_vecs(rhs_rows)
    }

    // Compute the kernel (null space) of the matrix using Gaussian elimination.
    pub fn kernel(&self, rcond: f64) -> Vec<Vector> {
        let max = self.max_norm();
        let transpose_self = self.clone().transpose();
        let (lhs, rhs) = transpose_self.row_echelon(
            &Matrix::identity(Complex::new(1.0, 0.0), self.width),
            rcond * max,
        );
        let lhs_rows = lhs.to_row_vecs();
        let rhs_rows = rhs.to_row_vecs();
        zip(lhs_rows.into_iter(), rhs_rows.into_iter())
            .filter_map(|(l, r)| {
                if l.is_zero(rcond * max) {
                    Some(r)
                } else {
                    None
                }
            })
            .collect()
    }

    // Implementation of Eigendecomposition by QR algorithm.
    pub fn eigen(&self, rcond: f64, max_iters: usize) -> (Vec<Complex<f64>>, Vec<Vector>) {
        assert_eq!(self.width, self.height);

        let mut similar_matrix = self.clone();
        let mut i = 0;
        loop {
            if i == max_iters {
                return (vec![], vec![]);
            }
            // The eigenvalues of triangular matrices are the diagonal.
            if similar_matrix.is_triangular(rcond) {
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
        let max = similar_matrix.max_norm();
        for i in 0..self.width {
            let diag_val = similar_matrix.data[i * self.width + i];
            if diag_val.norm_sqr() > rcond * max {
                eigenvalues.push(diag_val);
            }
        }

        let mut dedup_eigenvals: Vec<Complex<f64>> = Vec::new();
        let mut multiplicities: Vec<usize> = Vec::new();
        let max = eigenvalues
            .iter()
            .map(|x| -> f64 { x.norm_sqr() })
            .fold(f64::MIN, |acc, x| -> f64 { x.max(acc) });
        for eigenval in eigenvalues.iter() {
            let mut dup = false;
            for prev_eigenval in dedup_eigenvals.iter() {
                if (eigenval - prev_eigenval).norm_sqr() < rcond * max {
                    dup = true;
                    *multiplicities.last_mut().unwrap() += 1;
                    break;
                }
            }
            if !dup {
                dedup_eigenvals.push(*eigenval);
                multiplicities.push(1);
            }
        }

        // Now that we have the eigenvalues for the matrix, we can find the eigenvectors by solving
        // the equation (A - lambda * I) * x = 0, where A is the original matrix, lambda is an
        // eigenvalue. The non-zero vectors x which satisfy this equation, i.e. the null space of
        // (A - lambda * I), are the eigenvectors.
        let mut eigenvectors: Vec<Vector> = Vec::new();
        for (eigenval, multiplicity) in zip(dedup_eigenvals.iter(), multiplicities.iter()) {
            // We can have multiple eigenvectors per eigenvalue. In this case, we should return the
            // orthogonal basis for the eigenspace of the corresponding eigenvalues, or else
            // diagonalization won't work because our eigenvectors won't be linearly independent.
            let char_matrix = self.clone() - Matrix::identity(*eigenval, self.width);
            let max = char_matrix.max_norm();
            let (_, rhs) = char_matrix.transpose().row_echelon(
                &Matrix::identity(Complex::new(1.0, 0.0), self.width),
                rcond * max,
            );
            let curr_eigenspace = rhs.to_row_vecs()[self.width - multiplicity..].to_vec();
            let mut ortho_eigenspace: Vec<Vector> = Vec::new();
            for mut eigenvec in curr_eigenspace.into_iter() {
                for prev_eigenvec in ortho_eigenspace.iter() {
                    let coeff = eigenvec.proj(prev_eigenvec);
                    eigenvec = eigenvec - (coeff * prev_eigenvec.clone());
                }
                let eigenvec = Complex::new(1.0, 0.0) / eigenvec.l2() * eigenvec;
                ortho_eigenspace.push(eigenvec.clone());
            }
            eigenvectors.append(&mut ortho_eigenspace);
        }
        (eigenvalues, eigenvectors)
    }
}

impl TryFrom<Vec<Vec<Complex<f64>>>> for Matrix {
    type Error = String;

    fn try_from(value: Vec<Vec<Complex<f64>>>) -> Result<Self, Self::Error> {
        if value.len() == 0 {
            return Err("Cannot initialize empty matrix!".into());
        }
        let mut row_vecs: Vec<Vector> = Vec::new();
        let row_len = value[0].len();
        for row in value.into_iter() {
            if row.len() != row_len {
                return Err("Matrix has inconsistent dimensions!".into());
            }
            row_vecs.push(row.into());
        }
        Ok(Matrix::from_row_vecs(row_vecs))
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

impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix: [\n")?;
        for y in 0..self.height {
            write!(f, "\t[")?;
            for x in 0..self.width {
                if x != self.width - 1 {
                    write!(f, "{}, ", self.data[y * self.width + x].re)?;
                } else {
                    write!(f, "{}", self.data[y * self.width + x].re)?;
                }
            }
            write!(f, "],\n")?;
        }
        write!(f, "]")
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
                Complex::new(-2.0, 0.0),
                Complex::new(-2.0, 0.0),
                Complex::new(-8.0, 0.0),
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
            Vector::from(vec![
                Complex::new(-0.8944271909999159, 0.0),
                Complex::new(0.4472135954999579, 0.0),
            ]),
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
