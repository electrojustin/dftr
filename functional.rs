use num::Complex;

use crate::grid::Grid;

pub fn repulsion_potential_functional(mut electron_density: Grid) -> Grid {
    // Instead of actually performing the double integral, which is O(N^2) with respect to the
    // grid size, we treat the repulsion potential as a convolution between 1/|r| and p(r),
    // which we can compute efficiently in the frequency domain as a simple multiplication. This
    // reduces the time complexity to that of the FFT algorithm, which is O(N log N).
    let mut potential = electron_density.clone();
    let x_offset = potential.config.start_x;
    let y_offset = potential.config.start_y;
    let z_offset = potential.config.start_z;
    potential.fill(&|x, y, z| -> Complex<f64> {
        Complex::new(1.0 / (x * x + y * y + z * z).sqrt().max(0.01), 0.0)
    });
    potential.fourier(false, false);
    electron_density.fourier(false, false);
    let mut test = electron_density.clone();
    potential = potential * electron_density;
    potential.fourier(true, true);
    potential
}

mod tests {
    use super::*;
    use crate::ansatz::gaussian_type_orbital::GTO;
    use crate::ansatz::Ansatz;
    use crate::grid::GridConfig;

    const K_GRID_CONFIG: GridConfig = GridConfig {
        start_x: -3.0,
        start_y: -3.0,
        start_z: -3.0,
        end_x: 3.0,
        end_y: 3.0,
        end_z: 3.0,
        width_voxels: 64,
        height_voxels: 64,
        depth_voxels: 64,
    };

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    #[test]
    fn test_hydrogen_repulsion_potential() {
        let alpha = 0.25;
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, alpha, 0, 0, 0);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let potential = repulsion_potential_functional(electron_density);
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
