use num::complex::Complex;

use crate::ansatz::Ansatz;
use crate::grid::Grid;
use crate::grid::GridConfig;

pub struct Nucleus {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub charge: f64,
}

// Populate a grid with the nuclear-electron coulombic energy. Note that this grid alone is
// insufficient for the calculation, it needs to be combined with wavefunction bra and kets.
pub fn nuclear_potential(nuclei: &Vec<Nucleus>, grid_config: GridConfig) -> Grid {
    let mut grid = Grid::new(grid_config);
    grid.fill(&|x, y, z| -> Complex<f64> {
        nuclei
            .iter()
            .map(|nucleus| -> Complex<f64> {
                let dx = nucleus.x - x;
                let dy = nucleus.y - y;
                let dz = nucleus.z - z;
                // We experience numericals instability the closer we get to a nucleus since the
                // distance will be 0, triggering a divide by 0 error. As a practical solution, we
                // just set the minimum distance to 0.01 A.
                let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);
                Complex::new(-nucleus.charge / distance, 0.0)
            })
            .fold(Complex::new(0.0, 0.0), |acc, e| -> Complex<f64> { acc + e })
    });
    grid
}

mod tests {
    use super::*;
    use crate::ansatz::gaussian_type_orbital::GTO;
    use crate::grid::GridConfig;

    const K_GRID_CONFIG: GridConfig = GridConfig {
        start_x: -5.0,
        start_y: -5.0,
        start_z: -5.0,
        end_x: 5.0,
        end_y: 5.0,
        end_z: 5.0,
        width_voxels: 100,
        height_voxels: 100,
        depth_voxels: 100,
    };

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    #[test]
    fn test_hydrogen_nuclear_potential() {
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, 0.25, 0, 0, 0);
        let bra = test_gto.bra(K_GRID_CONFIG);
        let ket = test_gto.ket(K_GRID_CONFIG);
        let potential = nuclear_potential(
            &vec![Nucleus {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                charge: 1.0,
            }],
            K_GRID_CONFIG,
        );
        let integral = (bra * potential * ket).integrate().re;
        assert!(
            (integral - -0.798).abs() < 0.1,
            "Incorrect hydrogen nuclear-electron energy! Expected {} Actual {}",
            -0.798,
            integral
        );
    }
}
