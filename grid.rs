use std::iter::zip;
use std::ops;

use num::complex::Complex;

#[derive(Debug, Clone)]
pub struct Grid {
    data: Vec<Complex<f64>>,
    start_x: f64,
    start_y: f64,
    start_z: f64,
    end_x: f64,
    end_y: f64,
    end_z: f64,
    width_voxels: usize,
    height_voxels: usize,
    depth_voxels: usize,
}

impl Grid {
    pub fn new(
        start_x: f64,
        start_y: f64,
        start_z: f64,
        end_x: f64,
        end_y: f64,
        end_z: f64,
        width_voxels: usize,
        height_voxels: usize,
        depth_voxels: usize,
    ) -> Self {
        Self {
            data: vec![Complex::new(0.0, 0.0); width_voxels * height_voxels * depth_voxels],
            start_x,
            start_y,
            start_z,
            end_x,
            end_y,
            end_z,
            width_voxels,
            height_voxels,
            depth_voxels,
        }
    }

    pub fn fill(&mut self, func: &impl Fn(f64, f64, f64) -> Complex<f64>) {
        self.map(&|x, y, z, _old| -> Complex<f64> { func(x, y, z) });
    }

    pub fn integrate(&self) -> Complex<f64> {
        let x_res = (self.end_x - self.start_x) / self.width_voxels as f64;
        let y_res = (self.end_y - self.start_y) / self.height_voxels as f64;
        let z_res = (self.end_z - self.start_z) / self.depth_voxels as f64;
        let cubic_res = Complex::new(x_res * y_res * z_res, 0.0);
        self.data
            .iter()
            .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> {
                acc + x * cubic_res
            })
    }

    pub fn map(&mut self, func: &impl Fn(f64, f64, f64, Complex<f64>) -> Complex<f64>) {
        let x_res = (self.end_x - self.start_x) / self.width_voxels as f64;
        let y_res = (self.end_y - self.start_y) / self.height_voxels as f64;
        let z_res = (self.end_z - self.start_z) / self.depth_voxels as f64;
        let mut z = self.start_z;
        for z_idx in 0..self.depth_voxels {
            let mut y = self.start_y;
            for y_idx in 0..self.height_voxels {
                let mut x = self.start_x;
                for x_idx in 0..self.width_voxels {
                    let old_data = self.data[z_idx * self.width_voxels * self.height_voxels
                        + y_idx * self.width_voxels
                        + x_idx];
                    self.data[z_idx * self.width_voxels * self.height_voxels
                        + y_idx * self.width_voxels
                        + x_idx] = func(x, y, z, old_data);
                    x += x_res;
                }
                y += y_res;
            }
            z += z_res;
        }
    }
}

impl ops::Add<Grid> for Grid {
    type Output = Grid;

    fn add(self, rhs: Grid) -> Self::Output {
        assert_eq!(self.start_x, rhs.start_x);
        assert_eq!(self.start_y, rhs.start_y);
        assert_eq!(self.start_z, rhs.start_z);
        assert_eq!(self.end_x, rhs.end_x);
        assert_eq!(self.end_y, rhs.end_y);
        assert_eq!(self.end_z, rhs.end_z);
        assert_eq!(self.width_voxels, rhs.width_voxels);
        assert_eq!(self.height_voxels, rhs.height_voxels);
        assert_eq!(self.depth_voxels, rhs.depth_voxels);

        let start_x = self.start_x;
        let start_y = self.start_y;
        let start_z = self.start_z;
        let end_x = self.end_x;
        let end_y = self.end_y;
        let end_z = self.end_z;
        let width_voxels = self.width_voxels;
        let height_voxels = self.height_voxels;
        let depth_voxels = self.depth_voxels;

        Grid {
            data: zip(self.data.into_iter(), rhs.data.into_iter())
                .map(|(x, y)| -> Complex<f64> { x + y })
                .collect(),
            start_x,
            start_y,
            start_z,
            end_x,
            end_y,
            end_z,
            width_voxels,
            height_voxels,
            depth_voxels,
        }
    }
}

impl ops::Mul<Grid> for Grid {
    type Output = Grid;

    fn mul(self, rhs: Grid) -> Self::Output {
        assert_eq!(self.start_x, rhs.start_x);
        assert_eq!(self.start_y, rhs.start_y);
        assert_eq!(self.start_z, rhs.start_z);
        assert_eq!(self.end_x, rhs.end_x);
        assert_eq!(self.end_y, rhs.end_y);
        assert_eq!(self.end_z, rhs.end_z);
        assert_eq!(self.width_voxels, rhs.width_voxels);
        assert_eq!(self.height_voxels, rhs.height_voxels);
        assert_eq!(self.depth_voxels, rhs.depth_voxels);

        let start_x = self.start_x;
        let start_y = self.start_y;
        let start_z = self.start_z;
        let end_x = self.end_x;
        let end_y = self.end_y;
        let end_z = self.end_z;
        let width_voxels = self.width_voxels;
        let height_voxels = self.height_voxels;
        let depth_voxels = self.depth_voxels;

        Grid {
            data: zip(self.data.into_iter(), rhs.data.into_iter())
                .map(|(x, y)| -> Complex<f64> { x * y })
                .collect(),
            start_x,
            start_y,
            start_z,
            end_x,
            end_y,
            end_z,
            width_voxels,
            height_voxels,
            depth_voxels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_and_integrate() {
        let mut test = Grid::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 100, 100, 100);
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
    fn test_add_and_mul() {
        let mut orig_test = Grid::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 100, 100, 100);
        orig_test.fill(&|_x, _y, _z| -> Complex<f64> { Complex::new(2.0, 0.0) });
        let test = orig_test.clone();
        let test2 = orig_test.clone();
        let integral = (test + test2).integrate().re;
        assert!(
            (integral - 4.0).abs() < 0.1,
            "Addition failed! Expected {} Actual {}",
            2.0,
            integral
        );
        let test = orig_test.clone();
        let test2 = orig_test.clone();
        let integral = (test * test2).integrate().re;
        assert!(
            (integral - 4.0).abs() < 0.1,
            "Multiplication failed! Expected {} Actual {}",
            2.0,
            integral
        );
    }
}
