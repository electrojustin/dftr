use std::iter::zip;

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

struct SCF<XC1: Fn(Grid) -> Grid, XC2: Fn(Grid) -> Grid, B: Basis> {
    pub electron_density: Grid,
    pub energy: Complex<f64>,
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
}

impl<XC1: Fn(Grid) -> Grid, XC2: Fn(Grid) -> Grid, B: Basis> SCF<XC1, XC2, B> {
    pub fn new(
        nuclei: Vec<Nucleus>,
        mut basis: Vec<B>,
        num_electrons: usize,
        exchange_correlation_functional: XC1,
        exchange_correlation_potential_functional: XC2,
        grid_config: GridConfig,
    ) -> Self {
        // In order to solve the general eigenvector problem, we make an orthonormal basis. We do
        // this using the Lowdin decomposition of the overlap matrix
        let overlap_matrix = Matrix::from_row_vecs(
            (0..basis.len())
                .map(|i| -> Vector {
                    (0..basis.len())
                        .map(|j| -> Complex<f64> {
                            (basis[i].bra(grid_config.clone()) * basis[j].ket(grid_config.clone()))
                                .integrate()
                        })
                        .collect::<Vec<_>>()
                        .into()
                })
                .collect::<Vec<_>>(),
        );
        assert!(
            overlap_matrix.is_symmetric(1E-10),
            "Overlap matrix not symmetric!"
        );
        let (eigenvals, eigenvecs) = overlap_matrix.clone().eigen(1E-10, basis.len() * 100);
        // Quickly validate overlap matrix eigendecomposition.
        for (val, vec) in zip(eigenvals.iter(), eigenvecs.iter()) {
            let expected = *val * vec.clone();
            let actual = overlap_matrix.clone() * vec.clone();
            assert!(
                expected.compare(&actual, 1E-10),
                "Error in overlap matrix eigen decomposition! Expected: {:?}\nActual: {:?}",
                expected,
                actual,
            );
        }
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

        let default_grid = Grid::new(grid_config.clone());

        Self {
            electron_density: default_grid.clone(),
            energy: Complex::new(0.0, 0.0),
            nuclear_repulsion_energy: nuclear_repulsion(&nuclei),
            exchange_correlation_functional,
            exchange_correlation_potential_functional,
            // Arbitrary guess for the coefficient matrix: normalized diagonal matrix.
            coeff_matrix: Matrix::identity(
                Complex::new(((num_electrons as f64) / 2.0) / (basis.len() as f64), 0.0),
                basis.len(),
            ),
            nuclear_potential: nuclear_potential(&nuclei, grid_config.clone()),
            grid_config,
            basis,
            orthogonalizer,
            inverse_orthogonalizer,
            repulsion_potential: default_grid.clone(),
            exchange_correlation_potential: default_grid,
        }
    }

    fn compute_energy_and_density(&mut self) {
        // Compute molecular orbitals from basis function and coefficients.
        let coeffs = self.coeff_matrix.to_col_vecs();
        let mut bras: Vec<Grid> = Vec::new();
        let mut kinetic_energies: Vec<Grid> = Vec::new();
        let mut kets: Vec<Grid> = Vec::new();
        // In order for the matrices to be square, we need as many basis functions as orbitals.
        for orbital in 0..self.basis.len() {
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

        let kinetic_energy = Complex::new(2.0, 0.0)
            * zip(bras.iter(), kinetic_energies.into_iter()).fold(
                Complex::new(0.0, 0.0),
                |acc, (bra, kinetic_energy)| -> Complex<f64> {
                    acc + (bra.clone() * kinetic_energy).integrate()
                },
            );

        let nuclear_potential_energy = Complex::new(2.0, 0.0)
            * zip(bras.iter(), kets.iter()).fold(
                Complex::new(0.0, 0.0),
                |acc, (bra, ket)| -> Complex<f64> {
                    acc + (bra.clone() * self.nuclear_potential.clone() * ket.clone()).integrate()
                },
            );

        self.repulsion_potential = repulsion_potential_functional(self.electron_density.clone());
        // We still have to divide the repulsion energy by 2 because each pairwise electron
        // interaction will be double counted otherwise.
        let repulsion_potential_energy = Complex::new(0.5, 0.0)
            * zip(bras.into_iter(), kets.into_iter()).fold(
                Complex::new(0.0, 0.0),
                |acc, (bra, ket)| -> Complex<f64> {
                    acc + (bra * self.repulsion_potential.clone() * ket).integrate()
                },
            );

        self.exchange_correlation_potential =
            (self.exchange_correlation_potential_functional)(self.electron_density.clone());
        let exchange_correlation_energy =
            (self.exchange_correlation_functional)(self.electron_density.clone()).integrate();

        println!(
            "{} {} {} {} {}",
            self.nuclear_repulsion_energy,
            kinetic_energy.re,
            nuclear_potential_energy.re,
            repulsion_potential_energy.re,
            exchange_correlation_energy.re
        );

        self.energy = self.nuclear_repulsion_energy
            + kinetic_energy
            + nuclear_potential_energy
            + repulsion_potential_energy
            + exchange_correlation_energy;
    }

    fn fock_matrix(&mut self) -> Matrix {
        Matrix::from_row_vecs(
            (0..self.basis.len())
                .map(|i| -> Vector {
                    (0..self.basis.len())
                        .map(|j| -> Complex<f64> {
                            (self.basis[i].bra(self.grid_config.clone())
                                * (self.basis[j].kinetic_energy(self.grid_config.clone())
                                    + (self.nuclear_potential.clone()
                                        + self.repulsion_potential.clone()
                                        + self.exchange_correlation_potential.clone())
                                        * self.basis[j].ket(self.grid_config.clone())))
                            .integrate()
                        })
                        .collect::<Vec<_>>()
                        .into()
                })
                .collect(),
        )
    }

    // Adapted from https://enccs.github.io/veloxchem-workshop/notebooks/rh-scf/
    // Returns true if orbitals have degenerated, so we should break the SCF loop.
    fn compute_coeff_matrix(&mut self) -> bool {
        let mut ret = false;
        let fock = self.fock_matrix();
        let ortho_fock = self.orthogonalizer.clone() * fock;
        let (eigenvals, mut eigenvecs) = ortho_fock.eigen(1E-10, self.basis.len() * 10);
        // Sometimes orbitals degenerate, so we need to 0 pad the coefficient matrix.
        if eigenvecs.len() < self.basis.len() {
            eigenvecs.append(
                &mut (eigenvecs.len()..self.basis.len())
                    .map(|i| -> Vector { vec![Complex::new(0.0, 0.0); self.basis.len()].into() })
                    .collect(),
            );
            ret = true;
        }
        self.coeff_matrix =
            self.inverse_orthogonalizer.clone() * (Matrix::from_row_vecs(eigenvecs).transpose());
        ret
    }

    // Returns true on convergence.
    fn iterate(&mut self, tolerance: f64, max_iters: usize) -> bool {
        self.compute_energy_and_density();

        for i in 0..max_iters {
            let old_energy = self.energy;
            println!("Iter: {}  Energy: {}", i, old_energy.re);
            if self.compute_coeff_matrix() {
                println!("Orbitals degenerated!");
                return false;
            }
            self.compute_energy_and_density();
            if (old_energy - self.energy).norm_sqr() < tolerance {
                println!(
                    "Converged!  Energy: {}\nCoeffs: {:?}",
                    old_energy.re, self.coeff_matrix
                );
                return true;
            }
        }
        false
    }
}

mod tests {
    use super::*;
    use crate::basis::caching_basis::CachingBasis;
    use crate::basis::sto_ng::STONG;
    use crate::functional::lda::lda_functional;
    use crate::functional::lda::lda_potential_functional;

    const K_GRID_CONFIG: GridConfig = GridConfig {
        start_x: -4.0,
        start_y: -4.0,
        start_z: -4.0,
        end_x: 4.0,
        end_y: 4.0,
        end_z: 4.0,
        width_voxels: 32,
        height_voxels: 32,
        depth_voxels: 32,
    };

    #[test]
    fn test_compute_helium_energy() {
        let basis = STONG::sto_3g(0.0, 0.0, 0.0, "1s").expect("Failed to create basis function!");
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
        let basis1 = STONG::sto_3g(2.0, 0.0, 0.0, "1s").expect("Failed to create basis function!");
        let basis2 = STONG::sto_3g(-2.0, 0.0, 0.0, "1s").expect("Failed to create basis function!");
        let nucleus1 = Nucleus {
            x: 2.0,
            y: 0.0,
            z: 0.0,
            charge: 2.0,
        };
        let nucleus2 = Nucleus {
            x: -2.0,
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
            (actual - expected).abs() < 0.3,
            "Incorrect double helium energy! Expected {} Actual {}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_scf_molecular_hydrogen() {
        // Experimental H-H bond length is 0.7414A.
        // Source: https://cccbdb.nist.gov/exp2x.asp?casno=1333740
        let basis1 =
            STONG::sto_3g(-0.7414 / 2.0, 0.0, 0.0, "1s").expect("Failed to create basis function!");
        let basis2 =
            STONG::sto_3g(0.7414 / 2.0, 0.0, 0.0, "1s").expect("Failed to create basis function!");
        let nucleus1 = Nucleus {
            x: -0.7414 / 2.0,
            y: 0.0,
            z: 0.0,
            charge: 1.0,
        };
        let nucleus2 = Nucleus {
            x: 0.7414 / 2.0,
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
        scf.iterate(1E-10, 10);
        let actual = scf.energy.re;
        let expected = -1.025;
        assert!(
            (actual - expected).abs() < 0.1,
            "Incorrect hydrogen molecule energy! Expected {} Actual {}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_scf_neon() {
        let basis1 = STONG::sto_3g(0.0, 0.0, 0.0, "1s").expect("Failed to create basis function!");
        let basis2 = STONG::sto_3g(0.0, 0.0, 0.0, "2s").expect("Failed to create basis function!");
        let basis3 = STONG::sto_3g(0.0, 0.0, 0.0, "2p1").expect("Failed to create basis function!");
        let basis4 = STONG::sto_3g(0.0, 0.0, 0.0, "2p2").expect("Failed to create basis function!");
        let basis5 = STONG::sto_3g(0.0, 0.0, 0.0, "2p3").expect("Failed to create basis function!");
        let nucleus = Nucleus {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            charge: 10.0,
        };
        let mut scf = SCF::new(
            vec![nucleus],
            vec![
                CachingBasis::new(basis1),
                CachingBasis::new(basis2),
                CachingBasis::new(basis3),
                CachingBasis::new(basis4),
                CachingBasis::new(basis5),
            ],
            10,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        scf.iterate(1E-10, 10);
        let actual = scf.energy.re;
        let expected = -1.025;
        assert!(
            (actual - expected).abs() < 0.1,
            "Incorrect neon energy! Expected {} Actual {}",
            expected,
            actual,
        );
    }
}
