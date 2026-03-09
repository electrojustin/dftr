use std::iter::zip;
use std::ops;

use num::complex::Complex;

#[derive(Debug, Clone)]
pub struct Vector(Vec<Complex<f64>>);

impl Vector {
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

    pub fn dot(&self, other: &Vector) -> Complex<f64> {
        zip(self.0.iter(), other.0.iter())
            .map(|(x, y)| -> Complex<f64> { x * y })
            .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x })
    }

    pub fn proj(&self, other: &Vector) -> Complex<f64> {
        other.dot(self) / other.dot(other)
    }

    pub fn l2(&self) -> Complex<f64> {
        self.dot(self).sqrt()
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

    pub fn to_row_vecs(&self) -> Vec<Vector> {
        (0..self.height)
            .map(|y| -> Vector {
                self.data[y * self.height..(y + 1) * self.height]
                    .to_vec()
                    .into()
            })
            .collect()
    }

    pub fn to_col_vecs(&self) -> Vec<Vector> {
        (0..self.width)
            .map(|x| -> Vector {
                (0..self.height)
                    .map(|y| -> Complex<f64> { self.data[y * self.height + x] })
                    .collect::<Vec<_>>()
                    .into()
            })
            .collect()
    }

    pub fn transpose(self) -> Matrix {
        Matrix::from_row_vecs(self.to_col_vecs())
    }

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
        let (actual_q, actual_r) = lhs.qr_decomp();
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
    }
}
