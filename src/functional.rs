use num::Complex;

use crate::grid::Grid;
use crate::grid::GridConfig;

pub mod lda;

pub struct RepulsionCache {
    inverse_func_fourier: Grid,
}

impl RepulsionCache {
    pub fn new(grid_config: GridConfig) -> Self {
        let mut inverse_func = Grid::new(grid_config.clone());
        let x_offset = grid_config.start_x;
        let y_offset = grid_config.start_y;
        let z_offset = grid_config.start_z;
        inverse_func.fill(&|x, y, z| -> Complex<f64> {
            Complex::new(1.0 / (x * x + y * y + z * z).sqrt().max(0.01), 0.0)
        });
        inverse_func.fourier(false, false);
        Self {
            inverse_func_fourier: inverse_func,
        }
    }
}

pub fn repulsion_potential_functional(mut electron_density: Grid, cache: &RepulsionCache) -> Grid {
    // Instead of actually performing the double integral, which is O(N^2) with respect to the
    // grid size, we treat the repulsion potential as a convolution between 1/|r| and p(r),
    // which we can compute efficiently in the frequency domain as a simple multiplication. This
    // reduces the time complexity to that of the FFT algorithm, which is O(N log N).
    // I first found this idea referenced here: https://docs.onetep.org/cutoff_coulomb.html
    // But the analytic fourier transforms in this source don't look correct to me, so I just
    // implemented it numerically...
    assert_eq!(electron_density.config, cache.inverse_func_fourier.config);
    electron_density.fourier(false, false);
    let mut potential = cache.inverse_func_fourier.clone() * electron_density;
    potential.fourier(true, true);
    potential
}

mod tests {
    use test_log::test;

    use super::*;
    use crate::basis::gaussian_type_orbital::GTO;
    use crate::basis::Basis;
    use crate::grid::GridConfig;

    const K_GRID_CONFIG: GridConfig = GridConfig {
        start_x: -3.0,
        start_y: -3.0,
        start_z: -3.0,
        end_x: 3.0,
        end_y: 3.0,
        end_z: 3.0,
        width_voxels: 32,
        height_voxels: 32,
        depth_voxels: 32,
    };

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    #[test]
    fn test_hydrogen_repulsion_potential() {
        let alpha = 0.25;
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, alpha, 0, 0, 0, true);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let potential =
            repulsion_potential_functional(electron_density, &RepulsionCache::new(K_GRID_CONFIG));
        let integral = (bra * potential * ket).integrate().re;
        let expected = 1.128 * alpha.sqrt();
        assert!(
            (integral - expected).abs() < 0.01,
            "Incorrect hydrogen electron repulsion energy! Expected {} Actual {}",
            expected,
            integral
        );
    }
}
