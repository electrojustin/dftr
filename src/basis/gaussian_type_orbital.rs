use num::complex::Complex;

use crate::basis::Basis;
use crate::utils::factorial;

// Factor out the core calculations so we can call it multiple times for computing the Laplacian.
// Taken from https://en.wikipedia.org/wiki/Gaussian_orbital

// Returns the normalization factor. Only needs computed once per instantiation.
fn gto_norm_helper(alpha: f64, i: i32, j: i32, k: i32) -> f64 {
    (2.0 * alpha / std::f64::consts::PI).powf(0.75)
        * (((8.0 * alpha).powf((i + j + k) as f64)
            * factorial(i) as f64
            * factorial(j) as f64
            * factorial(k) as f64)
            / (factorial(2 * i) as f64 * factorial(2 * j) as f64 * factorial(2 * k) as f64))
            .sqrt()
}

// Returns the unnormalized, real, cartesian gaussian type orbital at (x, y, z).
fn gto_helper(x: f64, y: f64, z: f64, alpha: f64, i: i32, j: i32, k: i32) -> f64 {
    let x = if x.abs() < 0.01 { 0.01 } else { x };
    let y = if y.abs() < 0.01 { 0.01 } else { y };
    let z = if z.abs() < 0.01 { 0.01 } else { z };
    x.powi(i) * y.powi(j) * z.powi(k) * (-alpha * (x * x + y * y + z * z)).exp()
}

#[derive(Debug, Clone)]
pub struct GTO {
    x: f64,
    y: f64,
    z: f64,
    alpha: f64,
    i: i32,
    j: i32,
    k: i32,
    norm: f64,
}

impl GTO {
    pub fn new(x: f64, y: f64, z: f64, alpha: f64, i: i32, j: i32, k: i32) -> Self {
        GTO {
            x,
            y,
            z,
            alpha,
            i,
            j,
            k,
            norm: gto_norm_helper(alpha, i, j, k),
        }
    }
}

impl Basis for GTO {
    fn pos(&self, x: f64, y: f64, z: f64) -> Complex<f64> {
        Complex::new(
            self.norm
                * gto_helper(
                    x - self.x,
                    y - self.y,
                    z - self.z,
                    self.alpha,
                    self.i,
                    self.j,
                    self.k,
                ),
            0.0,
        )
    }

    // Adapted from https://gqcg-res.github.io/knowdes/gaussian-type-orbitals.html
    fn laplacian(&self, x: f64, y: f64, z: f64) -> Complex<f64> {
        // This term is used in all 3 components, so we compute it only once.
        let intermediate = -2.0
            * self.alpha
            * gto_helper(
                x - self.x,
                y - self.y,
                z - self.z,
                self.alpha,
                self.i,
                self.j,
                self.k,
            );

        let x_component = 4.0
            * self.alpha
            * self.alpha
            * gto_helper(
                x - self.x,
                y - self.y,
                z - self.z,
                self.alpha,
                self.i + 2,
                self.j,
                self.k,
            )
            + (2.0 * self.i as f64 + 1.0) * intermediate
            + self.i as f64
                * (self.i as f64 - 1.0)
                * gto_helper(
                    x - self.x,
                    y - self.y,
                    z - self.z,
                    self.alpha,
                    self.i - 2,
                    self.j,
                    self.k,
                );
        let y_component = 4.0
            * self.alpha
            * self.alpha
            * gto_helper(
                x - self.x,
                y - self.y,
                z - self.z,
                self.alpha,
                self.i,
                self.j + 2,
                self.k,
            )
            + (2.0 * self.j as f64 + 1.0) * intermediate
            + self.j as f64
                * (self.j as f64 - 1.0)
                * gto_helper(
                    x - self.x,
                    y - self.y,
                    z - self.z,
                    self.alpha,
                    self.i,
                    self.j - 2,
                    self.k,
                );
        let z_component = 4.0
            * self.alpha
            * self.alpha
            * gto_helper(
                x - self.x,
                y - self.y,
                z - self.z,
                self.alpha,
                self.i,
                self.j,
                self.k + 2,
            )
            + (2.0 * self.k as f64 + 1.0) * intermediate
            + self.k as f64
                * (self.k as f64 - 1.0)
                * gto_helper(
                    x - self.x,
                    y - self.y,
                    z - self.z,
                    self.alpha,
                    self.i,
                    self.j,
                    self.k - 2,
                );
        Complex::new(self.norm * (x_component + y_component + z_component), 0.0)
    }
}

mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::grid::GridConfig;

    const K_GRID_CONFIG: GridConfig = GridConfig {
        start_x: -5.0,
        start_y: -5.0,
        start_z: -5.0,
        end_x: 5.0,
        end_y: 5.0,
        end_z: 5.0,
        width_voxels: 64,
        height_voxels: 64,
        depth_voxels: 64,
    };

    #[test]
    fn test_normalized() {
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, 0.25, 0, 0, 0);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let integral = (bra.clone() * ket.clone()).integrate().re;
        assert!(
            (integral - 1.0).abs() < 0.1,
            "s-type GTO is not normalized! Expected {} Actual {}",
            1.0,
            integral
        );

        let mut test_gto = GTO::new(0.0, 0.0, 0.0, 0.25, 1, 0, 0);
        test_gto.bra(K_GRID_CONFIG);
        test_gto.ket(K_GRID_CONFIG);
        let integral = (bra * ket).integrate().re;
        assert!(
            (integral - 1.0).abs() < 0.1,
            "p-type GTO is not normalized! Expected {} Actual {}",
            1.0,
            integral
        );
    }

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    #[test]
    fn test_kinetic_energy() {
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, 0.25, 0, 0, 0);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let kinetic_energy = test_gto.kinetic_energy(K_GRID_CONFIG);
        let integral = (bra * kinetic_energy).integrate().re;
        assert!(
            (integral - 1.5 * 0.25).abs() < 0.1,
            "Incorrect kinetic energy! Expected {} Actual {}",
            1.5 * 0.25,
            integral
        );
    }
}
