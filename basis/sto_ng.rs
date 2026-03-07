use std::iter::zip;

use num::complex::Complex;

use crate::basis::gaussian_type_orbital::GTO;
use crate::basis::Basis;

// Fixed linear combination of more primitive orbitals, usually GTOs.
pub struct STONG<T: Basis> {
    delegates: Vec<T>,
    coefficients: Vec<Complex<f64>>,
}

impl<T: Basis> STONG<T> {
    pub fn new(delegates: Vec<T>, coefficients: Vec<Complex<f64>>) -> Self {
        STONG {
            delegates,
            coefficients,
        }
    }
}

impl STONG<GTO> {
    // Adapted from https://en.wikipedia.org/wiki/STO-nG_basis_sets
    pub fn sto_2g(x: f64, y: f64, z: f64, shell: &str) -> Result<STONG<GTO>, String> {
        match shell {
            "1s" => Ok(STONG::new(
                vec![
                    GTO::new(x, y, z, 0.151632, 0, 0, 0),
                    GTO::new(x, y, z, 0.851819, 0, 0, 0),
                ],
                vec![Complex::new(0.678914, 0.0), Complex::new(0.430129, 0.0)],
            )),
            _ => Err(format!("STO-2G presets do not yet support {shell} shell")),
        }
    }

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

impl<T: Basis> Basis for STONG<T> {
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

mod tests {
    use super::*;
    use crate::basis::gaussian_type_orbital::GTO;
    use crate::basis::Basis;
    use crate::functional::lda::x_alpha_functional;
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
        width_voxels: 64,
        height_voxels: 64,
        depth_voxels: 64,
    };

    // X-Alpha exchange is a poor approximation, and monoatomic hydrogen isn't a good fit for
    // DFT anyway, so we expect our values to not be super accurate.
    #[test]
    fn test_hydrogen_sto2g() {
        let mut test = STONG::sto_2g(0.0, 0.0, 0.0, "1s").expect("Failed to create STO-3G 1s!");
        let bra = test.bra(K_GRID_CONFIG);
        let ket = test.ket(K_GRID_CONFIG);
        let ke = test.kinetic_energy(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let repulsion_pe = repulsion_potential_functional(electron_density.clone());
        let nuclear_pe = nuclear_potential(
            &vec![Nucleus {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                charge: 1.0,
            }],
            K_GRID_CONFIG,
        );
        let exchange = x_alpha_functional(electron_density);
        let hamiltonian = ((bra.clone() * ke.clone()).integrate()
            + (bra.clone() * nuclear_pe * ket.clone()).integrate()
            + (bra * repulsion_pe * ket).integrate()
            + exchange.integrate())
        .re;

        assert!(
            (hamiltonian - -0.5).abs() < 0.1,
            "Incorrect hydrogen atom energy! Expected {} Actual {}",
            -0.5,
            hamiltonian
        );
    }

    #[test]
    fn test_hydrogen_sto3g() {
        let mut test = STONG::sto_3g(0.0, 0.0, 0.0, "1s").expect("Failed to create STO-3G 1s!");
        let bra = test.bra(K_GRID_CONFIG);
        let ket = test.ket(K_GRID_CONFIG);
        let ke = test.kinetic_energy(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let repulsion_pe = repulsion_potential_functional(electron_density.clone());
        let nuclear_pe = nuclear_potential(
            &vec![Nucleus {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                charge: 1.0,
            }],
            K_GRID_CONFIG,
        );
        let exchange = x_alpha_functional(electron_density);
        let hamiltonian = ((bra.clone() * ke.clone()).integrate()
            + (bra.clone() * nuclear_pe * ket.clone()).integrate()
            + (bra * repulsion_pe * ket).integrate()
            + exchange.integrate())
        .re;

        assert!(
            (hamiltonian - -0.5).abs() < 0.1,
            "Incorrect hydrogen atom energy! Expected {} Actual {}",
            -0.5,
            hamiltonian
        );
    }

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    // TODO: Something seems off here, we're further off from the experimental values than
    // literature indicates we should be.
    #[test]
    fn test_helium_sto3g() {
        let mut test = STONG::sto_3g(0.0, 0.0, 0.0, "1s").expect("Failed to create STO-3G 1s!");
        let bra = test.bra(K_GRID_CONFIG);
        let ket = test.ket(K_GRID_CONFIG);
        let ke = test.kinetic_energy(K_GRID_CONFIG);
        let electron_density = bra.clone() * ket.clone();
        let repulsion_pe = repulsion_potential_functional(electron_density.clone());
        let nuclear_pe = nuclear_potential(
            &vec![Nucleus {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                charge: 2.0,
            }],
            K_GRID_CONFIG,
        );
        let exchange = x_alpha_functional(electron_density).integrate();
        let ke = Complex::new(2.0, 0.0) * (bra.clone() * ke.clone()).integrate();
        let nuclear_pe =
            Complex::new(2.0, 0.0) * (bra.clone() * nuclear_pe * ket.clone()).integrate();
        let repulsion_pe = Complex::new(2.0, 0.0) * (bra * repulsion_pe * ket).integrate();
        //println!("{:?} {:?} {:?} {:?}", ke, nuclear_pe, repulsion_pe, exchange);

        let hamiltonian = ke + nuclear_pe + repulsion_pe + exchange;
        let expected = -2.9034;
        assert!(
            (hamiltonian.re - expected).abs() < 0.7,
            "Incorrect helium atom energy! Expected {} Actual {}",
            expected,
            hamiltonian
        );
    }
}
