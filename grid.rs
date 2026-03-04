use std::hash::Hash;
use std::hash::Hasher;
use std::iter::zip;
use std::ops;

use num::complex::Complex;

#[derive(Debug, Clone)]
pub struct GridConfig {
    pub start_x: f64,
    pub start_y: f64,
    pub start_z: f64,
    pub end_x: f64,
    pub end_y: f64,
    pub end_z: f64,
    pub width_voxels: usize,
    pub height_voxels: usize,
    pub depth_voxels: usize,
}

impl PartialEq for GridConfig {
    fn eq(&self, other: &Self) -> bool {
        self.start_x.to_bits() == other.start_x.to_bits()
            && self.start_y.to_bits() == other.start_y.to_bits()
            && self.start_z.to_bits() == other.start_z.to_bits()
            && self.end_x.to_bits() == other.end_x.to_bits()
            && self.end_y.to_bits() == other.end_y.to_bits()
            && self.end_z.to_bits() == other.end_z.to_bits()
            && self.width_voxels == other.width_voxels
            && self.height_voxels == other.height_voxels
            && self.depth_voxels == other.depth_voxels
    }
}

impl Eq for GridConfig {}

impl Hash for GridConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start_x.to_bits().hash(state);
        self.start_y.to_bits().hash(state);
        self.start_z.to_bits().hash(state);
        self.end_x.to_bits().hash(state);
        self.end_y.to_bits().hash(state);
        self.end_z.to_bits().hash(state);
        self.width_voxels.hash(state);
        self.height_voxels.hash(state);
        self.depth_voxels.hash(state);
    }
}

#[derive(Debug, Clone)]
pub struct Grid {
    data: Vec<Complex<f64>>,
    config: GridConfig,
}

impl Grid {
    pub fn new(config: GridConfig) -> Self {
        Self {
            data: vec![
                Complex::new(0.0, 0.0);
                config.width_voxels * config.height_voxels * config.depth_voxels
            ],
            config,
        }
    }

    pub fn fill(&mut self, func: &impl Fn(f64, f64, f64) -> Complex<f64>) {
        self.map(&|x, y, z, _old| -> Complex<f64> { func(x, y, z) });
    }

    pub fn integrate(&self) -> Complex<f64> {
        let x_res = (self.config.end_x - self.config.start_x) / self.config.width_voxels as f64;
        let y_res = (self.config.end_y - self.config.start_y) / self.config.height_voxels as f64;
        let z_res = (self.config.end_z - self.config.start_z) / self.config.depth_voxels as f64;
        let cubic_res = Complex::new(x_res * y_res * z_res, 0.0);
        self.data
            .iter()
            .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> {
                acc + x * cubic_res
            })
    }

    pub fn map(&mut self, func: &impl Fn(f64, f64, f64, Complex<f64>) -> Complex<f64>) {
        let x_res = (self.config.end_x - self.config.start_x) / self.config.width_voxels as f64;
        let y_res = (self.config.end_y - self.config.start_y) / self.config.height_voxels as f64;
        let z_res = (self.config.end_z - self.config.start_z) / self.config.depth_voxels as f64;
        let mut z = self.config.start_z;
        for z_idx in 0..self.config.depth_voxels {
            let mut y = self.config.start_y;
            for y_idx in 0..self.config.height_voxels {
                let mut x = self.config.start_x;
                for x_idx in 0..self.config.width_voxels {
                    let old_data =
                        self.data[z_idx * self.config.width_voxels * self.config.height_voxels
                            + y_idx * self.config.width_voxels
                            + x_idx];
                    self.data[z_idx * self.config.width_voxels * self.config.height_voxels
                        + y_idx * self.config.width_voxels
                        + x_idx] = func(x, y, z, old_data);
                    x += x_res;
                }
                y += y_res;
            }
            z += z_res;
        }
    }
}

impl ops::Add<Grid> for Complex<f64> {
    type Output = Grid;

    fn add(self, rhs: Grid) -> Self::Output {
        Grid {
            data: rhs
                .data
                .into_iter()
                .map(|x| -> Complex<f64> { self + x })
                .collect(),
            config: rhs.config,
        }
    }
}

impl ops::Add<Complex<f64>> for Grid {
    type Output = Grid;

    fn add(self, rhs: Complex<f64>) -> Self::Output {
        rhs + self
    }
}

impl ops::Sub<Grid> for Complex<f64> {
    type Output = Grid;

    fn sub(self, rhs: Grid) -> Self::Output {
        Grid {
            data: rhs
                .data
                .into_iter()
                .map(|x| -> Complex<f64> { self - x })
                .collect(),
            config: rhs.config,
        }
    }
}

impl ops::Sub<Complex<f64>> for Grid {
    type Output = Grid;

    fn sub(self, rhs: Complex<f64>) -> Self::Output {
        Grid {
            data: self
                .data
                .into_iter()
                .map(|x| -> Complex<f64> { x - rhs })
                .collect(),
            config: self.config,
        }
    }
}

impl ops::Mul<Grid> for Complex<f64> {
    type Output = Grid;

    fn mul(self, rhs: Grid) -> Self::Output {
        Grid {
            data: rhs
                .data
                .into_iter()
                .map(|x| -> Complex<f64> { self * x })
                .collect(),
            config: rhs.config,
        }
    }
}

impl ops::Mul<Complex<f64>> for Grid {
    type Output = Grid;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        rhs * self
    }
}

impl ops::Add<Grid> for Grid {
    type Output = Grid;

    fn add(self, rhs: Grid) -> Self::Output {
        assert_eq!(self.config.start_x, rhs.config.start_x);
        assert_eq!(self.config.start_y, rhs.config.start_y);
        assert_eq!(self.config.start_z, rhs.config.start_z);
        assert_eq!(self.config.end_x, rhs.config.end_x);
        assert_eq!(self.config.end_y, rhs.config.end_y);
        assert_eq!(self.config.end_z, rhs.config.end_z);
        assert_eq!(self.config.width_voxels, rhs.config.width_voxels);
        assert_eq!(self.config.height_voxels, rhs.config.height_voxels);
        assert_eq!(self.config.depth_voxels, rhs.config.depth_voxels);

        Grid {
            data: zip(self.data.into_iter(), rhs.data.into_iter())
                .map(|(x, y)| -> Complex<f64> { x + y })
                .collect(),
            config: self.config,
        }
    }
}

impl ops::Sub<Grid> for Grid {
    type Output = Grid;

    fn sub(self, rhs: Grid) -> Self::Output {
        assert_eq!(self.config.start_x, rhs.config.start_x);
        assert_eq!(self.config.start_y, rhs.config.start_y);
        assert_eq!(self.config.start_z, rhs.config.start_z);
        assert_eq!(self.config.end_x, rhs.config.end_x);
        assert_eq!(self.config.end_y, rhs.config.end_y);
        assert_eq!(self.config.end_z, rhs.config.end_z);
        assert_eq!(self.config.width_voxels, rhs.config.width_voxels);
        assert_eq!(self.config.height_voxels, rhs.config.height_voxels);
        assert_eq!(self.config.depth_voxels, rhs.config.depth_voxels);

        Grid {
            data: zip(self.data.into_iter(), rhs.data.into_iter())
                .map(|(x, y)| -> Complex<f64> { x - y })
                .collect(),
            config: self.config,
        }
    }
}

impl ops::Mul<Grid> for Grid {
    type Output = Grid;

    fn mul(self, rhs: Grid) -> Self::Output {
        assert_eq!(self.config.start_x, rhs.config.start_x);
        assert_eq!(self.config.start_y, rhs.config.start_y);
        assert_eq!(self.config.start_z, rhs.config.start_z);
        assert_eq!(self.config.end_x, rhs.config.end_x);
        assert_eq!(self.config.end_y, rhs.config.end_y);
        assert_eq!(self.config.end_z, rhs.config.end_z);
        assert_eq!(self.config.width_voxels, rhs.config.width_voxels);
        assert_eq!(self.config.height_voxels, rhs.config.height_voxels);
        assert_eq!(self.config.depth_voxels, rhs.config.depth_voxels);

        Grid {
            data: zip(self.data.into_iter(), rhs.data.into_iter())
                .map(|(x, y)| -> Complex<f64> { x * y })
                .collect(),
            config: self.config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_GRID_CONFIG: GridConfig = GridConfig {
        start_x: 0.0,
        start_y: 0.0,
        start_z: 0.0,
        end_x: 1.0,
        end_y: 1.0,
        end_z: 1.0,
        width_voxels: 100,
        height_voxels: 100,
        depth_voxels: 100,
    };

    #[test]
    fn test_fill_and_integrate() {
        let mut test = Grid::new(K_GRID_CONFIG.clone());
        test.fill(&|_x, _y, _z| -> Complex<f64> { Complex::new(1.0, 0.0) });
        let integral = test.integrate().re;
        assert!(
            (integral - 1.0).abs() < 0.1,
            "Fill with constant failed! Expected {} Actual {}",
            1.0,
            integral
        );
        test.fill(&|x, _y, _z| -> Complex<f64> { Complex::new(x, 0.0) });
        let integral = test.integrate().re;
        assert!(
            (integral - 0.5).abs() < 0.1,
            "Fill with f(x, y, z) = x failed! Expected {} Actual {}",
            0.5,
            integral
        );
    }

    #[test]
    fn test_add_sub_mul() {
        let mut orig_test = Grid::new(K_GRID_CONFIG.clone());
        orig_test.fill(&|_x, _y, _z| -> Complex<f64> { Complex::new(2.0, 0.0) });
        let test = orig_test.clone();
        let test2 = orig_test.clone();
        let integral = (test + test2).integrate().re;
        assert!(
            (integral - 4.0).abs() < 0.1,
            "Addition failed! Expected {} Actual {}",
            4.0,
            integral
        );
        let test = orig_test.clone();
        let test2 = orig_test.clone();
        let integral = (test - test2).integrate().re;
        assert!(
            integral.abs() < 0.1,
            "Addition failed! Expected {} Actual {}",
            0.0,
            integral
        );
        let test = orig_test.clone();
        let test2 = orig_test.clone();
        let integral = (test * test2).integrate().re;
        assert!(
            (integral - 4.0).abs() < 0.1,
            "Multiplication failed! Expected {} Actual {}",
            4.0,
            integral
        );
    }

    #[test]
    fn test_scalar_arithmetic() {
        let mut orig_test = Grid::new(K_GRID_CONFIG.clone());
        orig_test.fill(&|_x, _y, _z| -> Complex<f64> { Complex::new(2.0, 0.0) });
        let integral = (orig_test.clone() + Complex::new(1.0, 0.0)).integrate().re;
        assert!(
            (integral - 3.0).abs() < 0.1,
            "Grid-scalar addition failed! Expected {} Actual {}",
            3.0,
            integral
        );
        let integral = (Complex::new(1.0, 0.0) + orig_test.clone()).integrate().re;
        assert!(
            (integral - 3.0).abs() < 0.1,
            "Scalar-grid addition failed! Expected {} Actual {}",
            3.0,
            integral
        );
        let integral = (orig_test.clone() * Complex::new(2.0, 0.0)).integrate().re;
        assert!(
            (integral - 4.0).abs() < 0.1,
            "Grid-scalar multiplication failed! Expected {} Actual {}",
            4.0,
            integral
        );
        let integral = (Complex::new(2.0, 0.0) * orig_test.clone()).integrate().re;
        assert!(
            (integral - 4.0).abs() < 0.1,
            "Scalar-grid multiplication failed! Expected {} Actual {}",
            4.0,
            integral
        );
        let integral = (orig_test.clone() - Complex::new(1.0, 0.0)).integrate().re;
        assert!(
            (integral - 1.0).abs() < 0.1,
            "Grid-scalar subtraction failed! Expected {} Actual {}",
            1.0,
            integral
        );
        let integral = (Complex::new(1.0, 0.0) - orig_test.clone()).integrate().re;
        assert!(
            (integral + 1.0).abs() < 0.1,
            "Scalar-grid subtraction failed! Expected {} Actual {}",
            -1.0,
            integral
        );
    }
}
