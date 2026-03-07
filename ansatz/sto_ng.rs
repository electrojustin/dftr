use std::iter::zip;

use num::complex::Complex;

use crate::ansatz::gaussian_type_orbital::GTO;
use crate::ansatz::Ansatz;

// Fixed linear combination of more primitive orbitals, usually GTOs.
pub struct STONG<T: Ansatz> {
    delegates: Vec<T>,
    coefficients: Vec<Complex<f64>>,
}

impl<T: Ansatz> STONG<T> {
    pub fn new(delegates: Vec<T>, coefficients: Vec<Complex<f64>>) -> Self {
        STONG {
            delegates,
            coefficients,
        }
    }
}

impl STONG<GTO> {
    // Adapted from https://en.wikipedia.org/wiki/STO-nG_basis_sets
    pub fn sto_3g(x: f64, y: f64, z: f64, shell: &str) -> Result<STONG<GTO>, String> {
        match shell {
            "1s" => Ok(STONG::new(
                vec![
                    GTO::new(x, y, z, 2.22766, 0, 0, 0),
                    GTO::new(x, y, z, 0.405771, 0, 0, 0),
                    GTO::new(x, y, z, 0.109818, 0, 0, 0),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
            _ => Err(format!("STO-3G presets do not yet support {shell} shell")),
        }
    }
}

impl<T: Ansatz> Ansatz for STONG<T> {
    fn pos(&self, x: f64, y: f64, z: f64) -> Complex<f64> {
        zip(self.delegates.iter(), self.coefficients.iter())
            .fold(Complex::new(0.0, 0.0), |acc, (d, c)| -> Complex<f64> {
                acc + d.pos(x, y, z) * c
            })
    }

    fn laplacian(&self, x: f64, y: f64, z: f64) -> Complex<f64> {
        zip(self.delegates.iter(), self.coefficients.iter())
            .fold(Complex::new(0.0, 0.0), |acc, (d, c)| -> Complex<f64> {
                acc + d.laplacian(x, y, z) * c
            })
    }
}
/*
mod tests {
    use super::*;
    use crate::ansatz::gaussian_type_orbital::GTO;
    use crate::ansatz::Ansatz;
    use crate::functional::repulsion_potential_functional;
    use crate::grid::GridConfig;
    use crate::nucleus::nuclear_potential;
    use crate::nucleus::Nucleus;

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
    #[ignore]
    #[test]
    fn test_hydrogen_sto3g() {
        let mut test = STONG::sto_3g(0.0, 0.0, 0.0, "1s").expect("Failed to create STO-3G 1s!");
        let bra = test.bra(K_GRID_CONFIG);
        let ket = test.ket(K_GRID_CONFIG);
        let ke = test.kinetic_energy(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let repulsion_pe = repulsion_potential_functional(electron_density);
        let nuclear_pe = nuclear_potential(
            &vec![Nucleus {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                charge: 1.0,
            }],
            K_GRID_CONFIG,
        );
        let hamiltonian = ((bra.clone() * ke.clone()).integrate()
            + (bra.clone() * repulsion_pe.clone() * ket.clone()).integrate()
            + (bra * nuclear_pe * ket).integrate())
        .re;

        // TODO: This currently fails. I think it's because I'm missing an exchange-correlation
        // functional in the hamiltonian?
        assert!(
            (hamiltonian - -0.5).abs() < 0.01,
            "Incorrect hydrogen atom energy! Expected {} Actual {}",
            -0.5,
            hamiltonian
        );
    }
}*/
