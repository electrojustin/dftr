use num::Complex;

use crate::grid::Grid;

pub fn repulsion_potential_functional(electron_density: Grid) -> Grid {
    let mut potential = electron_density;
    potential.convolve(
        &|x1, y1, z1, x2, y2, z2, val| -> Complex<f64> {
            let dx = x2 - x1;
            let dy = y2 - y1;
            let dz = z2 - z1;
            // Cap the distance at 0.1 A to avoid divide by 0 numerical instability.
            let distance: f64 = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
            val / Complex::new(distance, 0.0)
        },
        None,
        None,
        None,
    );
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
        width_voxels: 30,
        height_voxels: 30,
        depth_voxels: 30,
    };

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    #[test]
    fn test_hydrogen_repulsion_potential() {
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, 0.25, 0, 0, 0);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let potential = repulsion_potential_functional(electron_density);
        let integral = (bra * potential * ket).integrate().re;
        assert!(
            (integral - 0.564).abs() < 0.01,
            "Incorrect hydrogen electron repulsion energy! Expected {} Actual {}",
            0.564,
            integral
        );
    }
}
