use std::iter::zip;

use anyhow::anyhow;
use anyhow::Result;
use num::complex::Complex;

use crate::basis::Basis;
use crate::functional::repulsion_potential_functional;
use crate::grid::Grid;
use crate::grid::GridConfig;
use crate::linear::Matrix;
use crate::linear::Vector;
use crate::nucleus::nuclear_potential;
use crate::nucleus::nuclear_repulsion;
use crate::nucleus::Nucleus;

pub struct SCF<XC1: Fn(Grid) -> Grid, XC2: Fn(Grid) -> Grid, B: Basis> {
    pub electron_density: Grid,
    pub energy: Complex<f64>,
    num_electrons: usize,
    nuclear_repulsion_energy: Complex<f64>,
    exchange_correlation_functional: XC1,
    exchange_correlation_potential_functional: XC2,
    basis: Vec<B>,
    coeff_matrix: Matrix,
    grid_config: GridConfig,
    nuclear_potential: Grid,
    repulsion_potential: Grid,
    exchange_correlation_potential: Grid,
    orthogonalizer: Matrix,
    inverse_orthogonalizer: Matrix,
    fock_cache: Vec<Complex<f64>>,
}

impl<XC1: Fn(Grid) -> Grid, XC2: Fn(Grid) -> Grid, B: Basis> SCF<XC1, XC2, B> {
    pub fn new(
        nuclei: &Vec<Nucleus>,
        mut basis: Vec<B>,
        num_electrons: usize,
        exchange_correlation_functional: XC1,
        exchange_correlation_potential_functional: XC2,
        grid_config: GridConfig,
    ) -> Self {
        // In order to solve the general eigenvector problem, we make an orthonormal basis. We do
        // this using the Lowdin decomposition of the overlap matrix.
        log::debug!("Computing overlap matrix");
        let mut overlap_matrix = vec![vec![Complex::new(0.0, 0.0); basis.len()]; basis.len()];
        for i in 0..basis.len() {
            for j in i..basis.len() {
                let overlap = (basis[i].bra(grid_config.clone())
                    * basis[j].ket(grid_config.clone()))
                .integrate();
                overlap_matrix[i][j] = overlap;
                overlap_matrix[j][i] = overlap;
            }
        }
        let overlap_matrix: Matrix = overlap_matrix
            .try_into()
            .expect("Error creating overlap matrix!");
        log::debug!("Calculating Lowdin decomposition");
        let (eigenvals, eigenvecs) = overlap_matrix.clone().eigen(1E-10, basis.len() * 10000);
        let test_eigenvals = eigenvals.clone();
        let eigenvals = Matrix::from_diagonal(
            eigenvals
                .iter()
                .map(|x| -> Complex<f64> { x.powf(-0.5) })
                .collect(),
        );
        let eigenvecs = Matrix::from_row_vecs(eigenvecs).transpose();
        let orthogonalizer = eigenvecs.clone() * eigenvals.clone() * eigenvecs.clone().transpose();
        let inverse_orthogonalizer = orthogonalizer.clone().inverse(1E-10);
        assert!(
            Matrix::identity(Complex::new(1.0, 0.0), basis.len()).compare(
                &(inverse_orthogonalizer.clone() * orthogonalizer.clone()),
                1E-4
            ),
            "Error orthogonalizing basis! Cannot invert orthogonalizer."
        );
        assert!(
            overlap_matrix.compare(
                &(orthogonalizer.clone().transpose() * orthogonalizer.clone()).inverse(1E-10),
                1E-4
            ),
            "Error orthogonalizing basis! Cannot reconstruct overlap matrix from orthogonalizer."
        );

        // The nuclear coulombic attraction and kinetic energy terms of the Fock matrix elements
        // don't depend on electron density and therefore don't change every SCF iteration. We can
        // calculate these once and skip redoing all the integrals.
        log::debug!("Computing core Hamiltonian");
        let mut fock_cache = vec![Complex::new(0.0, 0.0); basis.len() * basis.len()];
        let nuclear_potential_grid = nuclear_potential(&nuclei, grid_config.clone());
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                let core_hamiltonian = (basis[i].bra(grid_config.clone())
                    * basis[j].kinetic_energy(grid_config.clone()))
                .integrate()
                    + (basis[i].bra(grid_config.clone())
                        * nuclear_potential_grid.clone()
                        * basis[j].ket(grid_config.clone()))
                    .integrate();
                fock_cache[i * basis.len() + j] = core_hamiltonian * Complex::new(1.0, 0.0);
            }
        }

        let default_grid = Grid::new(grid_config.clone());

        Self {
            electron_density: default_grid.clone(),
            energy: Complex::new(0.0, 0.0),
            nuclear_repulsion_energy: nuclear_repulsion(&nuclei),
            exchange_correlation_functional,
            exchange_correlation_potential_functional,
            // Arbitrary guess for the coefficient matrix: normalized diagonal matrix.
            coeff_matrix: Matrix::identity(
                Complex::new(
                    ((num_electrons as f64) / 2.0 / (basis.len() as f64)).sqrt(),
                    0.0,
                ),
                basis.len(),
            ),
            nuclear_potential: nuclear_potential_grid,
            grid_config,
            basis,
            orthogonalizer,
            inverse_orthogonalizer,
            repulsion_potential: default_grid.clone(),
            exchange_correlation_potential: default_grid,
            fock_cache,
            num_electrons,
        }
    }

    fn compute_energy_and_density(&mut self) {
        // Compute molecular orbitals from basis function and coefficients.
        let coeffs = self.coeff_matrix.to_col_vecs();
        let mut bras: Vec<Grid> = Vec::new();
        let mut kinetic_energies: Vec<Grid> = Vec::new();
        let mut kets: Vec<Grid> = Vec::new();
        // In molecules, we expected to have more molecular orbitals than electrons to fill them.
        // This results in unoccupied orbitals. We do not count these for our total ground state
        // energy and electron density calculations.
        for orbital in 0..(self.num_electrons / 2) {
            bras.push(
                zip(self.basis.iter_mut(), coeffs[orbital].0.iter())
                    .fold(Grid::new(self.grid_config.clone()), |acc, (x, c)| -> Grid {
                        acc + *c * x.bra(self.grid_config.clone())
                    }),
            );
            kinetic_energies.push(
                zip(self.basis.iter_mut(), coeffs[orbital].0.iter())
                    .fold(Grid::new(self.grid_config.clone()), |acc, (x, c)| -> Grid {
                        acc + *c * x.kinetic_energy(self.grid_config.clone())
                    }),
            );
            kets.push(
                zip(self.basis.iter_mut(), coeffs[orbital].0.iter())
                    .fold(Grid::new(self.grid_config.clone()), |acc, (x, c)| -> Grid {
                        acc + *c * x.ket(self.grid_config.clone())
                    }),
            );
        }

        // Currently we only support closed shell systems, so we double the electron density,
        // kinetic energy, and potential energy to account for 2 electrons being present in each
        // molecular orbital. Note that repulsion and exchange energies are not doubled because the
        // electron density already accounts for that.
        self.electron_density = zip(bras.iter(), kets.iter()).fold(
            Grid::new(self.grid_config.clone()),
            |acc, (bra, ket)| -> Grid { acc + Complex::new(2.0, 0.0) * bra.clone() * ket.clone() },
        );
        log::debug!("Total electrons: {}", self.electron_density.integrate().re);

        let kinetic_energy = zip(bras.iter(), kinetic_energies.into_iter()).fold(
            Complex::new(0.0, 0.0),
            |acc, (bra, kinetic_energy)| -> Complex<f64> {
                acc + (bra.clone() * kinetic_energy).integrate()
            },
        );

        let nuclear_potential_energy =
            (self.electron_density.clone() * self.nuclear_potential.clone()).integrate();

        self.repulsion_potential = repulsion_potential_functional(self.electron_density.clone());
        let repulsion_potential_energy = zip(bras.into_iter(), kets.into_iter()).fold(
            Complex::new(0.0, 0.0),
            |acc, (bra, ket)| -> Complex<f64> {
                acc + (bra * self.repulsion_potential.clone() * ket).integrate()
            },
        );

        self.exchange_correlation_potential =
            (self.exchange_correlation_potential_functional)(self.electron_density.clone());
        let exchange_correlation_energy =
            (self.exchange_correlation_functional)(self.electron_density.clone()).integrate();

        log::debug!(
            "Nucleus-Nucleus Repulsion: {}",
            self.nuclear_repulsion_energy.re
        );
        log::debug!("Kinetic Energy: {}", kinetic_energy.re);
        log::debug!(
            "Nucleus-Electron Attraction: {}",
            nuclear_potential_energy.re
        );
        log::debug!(
            "Electron-Electron Repulsion: {}",
            repulsion_potential_energy.re
        );
        log::debug!("Exchange-Correlation: {}", exchange_correlation_energy.re);

        self.energy = self.nuclear_repulsion_energy
            + kinetic_energy
            + nuclear_potential_energy
            + repulsion_potential_energy
            + exchange_correlation_energy;
    }

    fn fock_matrix(&mut self) -> Matrix {
        let mut fock_matrix =
            vec![vec![Complex::new(0.0, 0.0); self.basis.len()]; self.basis.len()];
        for i in 0..self.basis.len() {
            for j in 0..self.basis.len() {
                let entry = (self.basis[i].bra(self.grid_config.clone())
                    * (self.repulsion_potential.clone()
                        + self.exchange_correlation_potential.clone())
                    * self.basis[j].ket(self.grid_config.clone()))
                .integrate()
                    + self.fock_cache[i * self.basis.len() + j];
                fock_matrix[i][j] = entry;
            }
        }
        fock_matrix
            .try_into()
            .expect("Error creating overlap matrix!")
    }

    // Adapted from https://enccs.github.io/veloxchem-workshop/notebooks/rh-scf/ and
    // https://mattermodeling.stackexchange.com/questions/13574/decomposing-hartree-fock-into-orthogonalization-step-and-unitary-transformation
    // Returns true if orbitals have degenerated, so we should break the SCF loop.
    fn compute_coeff_matrix(&mut self) -> Result<()> {
        let fock = self.fock_matrix();
        let ortho_fock = self.orthogonalizer.clone() * fock * self.orthogonalizer.clone();
        let (energy_levels, eigenvecs) = ortho_fock.eigen(1E-10, self.basis.len() * 10000);
        if eigenvecs.is_empty() {
            Err(anyhow!(
                "Empty eigenvecs when solving for coefficient matrix! Orbitals have degenerated"
            ))
        } else {
            self.coeff_matrix =
                self.orthogonalizer.clone() * (Matrix::from_row_vecs(eigenvecs).transpose());
            Ok(())
        }
    }

    // Returns true on convergence.
    pub fn iterate(&mut self, tolerance: f64, max_iters: usize) -> Result<()> {
        self.compute_energy_and_density();

        for i in 0..max_iters {
            let old_energy = self.energy;
            log::info!("Iter: {}  Energy: {}", i, old_energy.re);
            self.compute_coeff_matrix()?;
            self.compute_energy_and_density();
            if (old_energy - self.energy).norm_sqr() < tolerance {
                log::info!("Converged! Energy: {}", old_energy.re);
                log::debug!("Coeffs: {:?}", self.coeff_matrix);
                return Ok(());
            }
        }
        Err(anyhow!(
            "Reached maximum number of SCF cycles with no convergence!"
        ))
    }
}

mod tests {
    use super::*;
    use crate::basis::caching_basis::CachingBasis;
    use crate::basis::contracted_basis::ContractedBasis;
    use crate::basis::gaussian_type_orbital::GTO;
    use crate::functional::lda::lda_functional;
    use crate::functional::lda::lda_potential_functional;

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
    fn test_compute_helium_energy() {
        let basis =
            ContractedBasis::sto_3g(0.0, 0.0, 0.0, "1s").expect("Failed to create basis function!");
        let nucleus = Nucleus {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            charge: 2.0,
        };
        let mut scf = SCF::new(
            vec![nucleus],
            vec![CachingBasis::new(basis)],
            2,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        scf.compute_energy_and_density();
        let actual = scf.energy.re;
        let expected = -2.9034;
        assert!(
            (actual - expected).abs() < 0.2,
            "Incorrect helium atom energy! Expected {} Actual {}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_compute_double_helium() {
        let bond_length: f64 = 5.0;
        let basis1 = ContractedBasis::sto_3g(bond_length / 2.0, 0.0, 0.0, "1s")
            .expect("Failed to create basis function!");
        let basis2 = ContractedBasis::sto_3g(-bond_length / 2.0, 0.0, 0.0, "1s")
            .expect("Failed to create basis function!");
        let nucleus1 = Nucleus {
            x: bond_length / 2.0,
            y: 0.0,
            z: 0.0,
            charge: 2.0,
        };
        let nucleus2 = Nucleus {
            x: -bond_length / 2.0,
            y: 0.0,
            z: 0.0,
            charge: 2.0,
        };
        let mut scf = SCF::new(
            vec![nucleus1, nucleus2],
            vec![CachingBasis::new(basis1), CachingBasis::new(basis2)],
            4,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        // We don't actually run SCF here because the orbitals will degenerate since He-He isn't a
        // real molecule.
        scf.compute_energy_and_density();
        let actual = scf.energy.re;
        let expected = -2.9034 * 2.0;
        assert!(
            (actual - expected).abs() < 0.5,
            "Incorrect double helium energy! Expected {} Actual {}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_scf_molecular_hydrogen() {
        // Experimental H-H bond length is 0.7414A.
        // Source: https://cccbdb.nist.gov/exp2x.asp?casno=1333740
        // Note that all of our calculations are in atomic units though, not Angstroms, so we
        // convert to Bohrs.
        let bond_length: f64 = 1.40104295;
        let basis1 = ContractedBasis::sto_3g(-bond_length / 2.0, 0.0, 0.0, "1s")
            .expect("Failed to create basis function!");
        let basis2 = ContractedBasis::sto_3g(bond_length / 2.0, 0.0, 0.0, "1s")
            .expect("Failed to create basis function!");
        let nucleus1 = Nucleus {
            x: -bond_length / 2.0,
            y: 0.0,
            z: 0.0,
            charge: 1.0,
        };
        let nucleus2 = Nucleus {
            x: bond_length / 2.0,
            y: 0.0,
            z: 0.0,
            charge: 1.0,
        };
        let mut scf = SCF::new(
            vec![nucleus1, nucleus2],
            vec![CachingBasis::new(basis1), CachingBasis::new(basis2)],
            2,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        let converged = scf.iterate(1E-10, 10).is_ok();
        assert!(converged, "Hydrogen molecule SCF failed to converge!");
        let actual = scf.energy.re;
        // Calculated using PySCF
        let expected = -1.121;
        // TODO: This function is really off. I wonder if it's the basis I'm using?
        assert!(
            (actual - expected).abs() < 0.2,
            "Incorrect hydrogen molecule energy! Expected {} Actual {}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_scf_neon() {
        // Adapted from https://www.basissetexchange.org/
        let basis = vec![
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(0.0, 0.0, 0.0, 0.2070156070E+03, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.3770815124E+02, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1020529731E+02, 0, 0, 0, false),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(0.0, 0.0, 0.0, 0.8246315120E+01, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1916266291E+01, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.6232292721E+00, 0, 0, 0, false),
                ],
                vec![
                    Complex::new(-0.9996722919E-01, 0.0),
                    Complex::new(0.3995128261E+00, 0.0),
                    Complex::new(0.7001154689e+00, 0.0),
                ],
            )),
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(0.0, 0.0, 0.0, 0.8246315120E+01, 1, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1916266291E+01, 1, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.6232292721E+00, 1, 0, 0, false),
                ],
                vec![
                    Complex::new(0.1559162750E+00, 0.0),
                    Complex::new(0.6076837186E+00, 0.0),
                    Complex::new(0.3919573931E+00, 0.0),
                ],
            )),
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(0.0, 0.0, 0.0, 0.8246315120E+01, 0, 1, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1916266291E+01, 0, 1, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.6232292721E+00, 0, 1, 0, false),
                ],
                vec![
                    Complex::new(0.1559162750E+00, 0.0),
                    Complex::new(0.6076837186E+00, 0.0),
                    Complex::new(0.3919573931E+00, 0.0),
                ],
            )),
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(0.0, 0.0, 0.0, 0.8246315120E+01, 0, 0, 1, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1916266291E+01, 0, 0, 1, false),
                    GTO::new(0.0, 0.0, 0.0, 0.6232292721E+00, 0, 0, 1, false),
                ],
                vec![
                    Complex::new(0.1559162750E+00, 0.0),
                    Complex::new(0.6076837186E+00, 0.0),
                    Complex::new(0.3919573931E+00, 0.0),
                ],
            )),
        ];
        let nucleus = Nucleus {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            charge: 10.0,
        };
        let mut scf = SCF::new(
            vec![nucleus],
            basis,
            10,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        let converged = scf.iterate(1E-10, 10).is_ok();
        assert!(converged, "Neon SCF failed to converge!");
        let actual = scf.energy.re;
        // Calculated using PySCF
        let expected = -125.3899;
        assert!(
            (actual - expected).abs() < (0.05 * expected.abs()),
            "Incorrect neon energy! Expected {} Actual {}",
            expected,
            actual,
        );
    }
}
