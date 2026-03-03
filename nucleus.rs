use num::complex::Complex;

use crate::ansatz::Ansatz;
use crate::grid::Grid;

pub struct Nucleus {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub charge: f64,
}

// Populate a grid with the nuclear-electron coulombic energy. Note that this grid alone is
// insufficient for the calculation, it needs to be combined with wavefunction bra and kets.
pub fn nuclear_coulomb_grid(grid: &mut Grid, nuclei: &Vec<Nucleus>) {
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
}

mod tests {
    use super::*;
    use crate::ansatz::gaussian_type_orbital::GTO;

    #[test]
    fn test_hydrogen_nuclear_electron_energy() {
        let test_grid = Grid::new(-5.0, -5.0, -5.0, 5.0, 5.0, 5.0, 100, 100, 100);
        let test_gto = GTO::new(0.0, 0.0, 0.0, 1.0, 0.25, 0, 0, 0);
        let mut bra = test_grid.clone();
        let mut ket = test_grid.clone();
        let mut nuclear_electron_energy = test_grid.clone();
        test_gto.bra(&mut bra);
        test_gto.ket(&mut ket);
        nuclear_coulomb_grid(
            &mut nuclear_electron_energy,
            &vec![Nucleus {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                charge: 1.0,
            }],
        );
        let integral = (bra * nuclear_electron_energy * ket).integrate().re;
        assert!(
            (integral - -0.798).abs() < 0.1,
            "Incorrect hydrogen nuclear-electron energy! Expected {} Actual {}",
            -0.798,
            integral
        );
    }
}
