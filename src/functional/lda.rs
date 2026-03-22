use num::Complex;

use crate::grid::Grid;

pub fn lda_functional(electron_density: Grid, alpha: f64) -> Grid {
    let mut exchange_density = electron_density;
    exchange_density.map(&|_x, _y, _z, val| -> Complex<f64> {
        Complex::new(
            -9.0 / 8.0 * alpha * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0),
            0.0,
        ) * val.powf(4.0 / 3.0)
    });
    exchange_density
}

pub fn lda_potential_functional(electron_density: Grid, alpha: f64) -> Grid {
    let mut exchange_potential = electron_density;
    exchange_potential.map(&|_x, _y, _z, val| -> Complex<f64> {
        Complex::new(
            -3.0 / 2.0 * alpha * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0),
            0.0,
        ) * val.powf(1.0 / 3.0)
    });
    exchange_potential
}

mod tests {
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
    fn test_helium_lda() {
        let alpha = 1.0;
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, alpha, 0, 0, 0, true);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let electron_density = Complex::new(2.0, 0.0) * bra.clone() * ket.clone();
        let exchange = lda_functional(electron_density, 1.05 * 2.0 / 3.0);
        let integral = exchange.integrate().re;
        let expected = -1.013 * alpha.sqrt();
        assert!(
            (integral - expected).abs() < 0.001,
            "Incorrect GTO X-alpha energy! Expected {} Actual {}",
            expected,
            integral
        );
    }
}
