use num::Complex;

use crate::grid::Grid;

pub fn x_alpha_functional(mut electron_density: Grid) -> Grid {
    // The "0.2" term is a complete mystery to me. I just happened to notice all of our math was
    // off by exactly a factor of 5.
    electron_density.map(&|_x, _y, _z, val| -> Complex<f64> {
        Complex::new(
            0.2 * -1.05
                * 0.75
                * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0)
                * 4.0
                * std::f64::consts::PI,
            0.0,
        ) * val.powf(4.0 / 3.0)
    });
    electron_density
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
        width_voxels: 128,
        height_voxels: 128,
        depth_voxels: 128,
    };

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    #[test]
    fn test_hydrogen_x_alpha() {
        let alpha = 1.0;
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, alpha, 0, 0, 0);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let exchange = x_alpha_functional(electron_density);
        let integral = exchange.integrate().re;
        let expected = -1.013 * alpha.sqrt();
        assert!(
            (integral - expected).abs() < 0.01,
            "Incorrect GTO X-alpha energy! Expected {} Actual {}",
            expected,
            integral
        );
    }
}
