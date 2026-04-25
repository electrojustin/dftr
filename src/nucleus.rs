use num::complex::Complex;

use crate::basis::Basis;
use crate::grid::Grid;
use crate::grid::GridConfig;
use crate::linear::Matrix;
use crate::linear::Vector;

#[derive(Clone, Debug)]
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
                let disp_x = nucleus.x - x;
                let disp_y = nucleus.y - y;
                let disp_z = nucleus.z - z;
                // Cap the distance at 0.01 A to avoid divide by 0 numerical instability.
                let distance = (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z)
                    .sqrt()
                    .max(0.01);
                Complex::new(-nucleus.charge / distance, 0.0)
            })
            .fold(Complex::new(0.0, 0.0), |acc, e| -> Complex<f64> { acc + e })
    });
    grid
}

// Classic coulomb nuclear repulsion energy.
pub fn nuclear_repulsion(nuclei: &Vec<Nucleus>) -> Complex<f64> {
    (0..nuclei.len())
        .map(|i| -> Complex<f64> {
            ((i + 1)..nuclei.len())
                .map(|j| -> Complex<f64> {
                    let disp_x = nuclei[i].x - nuclei[j].x;
                    let disp_y = nuclei[i].y - nuclei[j].y;
                    let disp_z = nuclei[i].z - nuclei[j].z;
                    Complex::new(
                        nuclei[i].charge * nuclei[j].charge
                            / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).sqrt(),
                        0.0,
                    )
                })
                .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x })
        })
        .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x })
}

pub fn nuclear_gradients(nuclei: &Vec<Nucleus>, electron_density: Grid) -> Vector {
    let mut grads: Vec<Complex<f64>> = Vec::new();
    for i in 0..nuclei.len() {
        let mut grad_x = Complex::new(0.0, 0.0);
        let mut grad_y = Complex::new(0.0, 0.0);
        let mut grad_z = Complex::new(0.0, 0.0);
        for j in 0..nuclei.len() {
            if i == j {
                continue;
            }
            let disp_x = nuclei[i].x - nuclei[j].x;
            let disp_y = nuclei[i].y - nuclei[j].y;
            let disp_z = nuclei[i].z - nuclei[j].z;
            let intermediate = -nuclei[i].charge * nuclei[j].charge
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).powf(3.0 / 2.0);
            grad_x += disp_x * intermediate;
            grad_y += disp_y * intermediate;
            grad_z += disp_z * intermediate;
        }
        grads.push(grad_x);
        grads.push(grad_y);
        grads.push(grad_z);
    }

    for i in 0..nuclei.len() {
        let mut grad_x = electron_density.clone();
        grad_x.map(&|x, y, z, val| -> Complex<f64> {
            let disp_x = nuclei[i].x - x;
            let disp_y = nuclei[i].y - y;
            let disp_z = nuclei[i].z - z;
            nuclei[i].charge * val * disp_x
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).max(0.01).powf(3.0 / 2.0)
        });
        let mut grad_y = electron_density.clone();
        grad_y.map(&|x, y, z, val| -> Complex<f64> {
            let disp_x = nuclei[i].x - x;
            let disp_y = nuclei[i].y - y;
            let disp_z = nuclei[i].z - z;
            nuclei[i].charge * val * disp_y
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).max(0.01).powf(3.0 / 2.0)
        });
        let mut grad_z = electron_density.clone();
        grad_z.map(&|x, y, z, val| -> Complex<f64> {
            let disp_x = nuclei[i].x - x;
            let disp_y = nuclei[i].y - y;
            let disp_z = nuclei[i].z - z;
            nuclei[i].charge * val * disp_z
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).max(0.01).powf(3.0 / 2.0)
        });
        grads[3 * i] += grad_x.integrate();
        grads[3 * i + 1] += grad_y.integrate();
        grads[3 * i + 2] += grad_z.integrate();
    }

    grads.into()
}

// I derived these by hand and don't feel super confident about them...
pub fn nuclear_hessian(nuclei: &Vec<Nucleus>, electron_density: Grid) -> Matrix {
    let mut ret: Vec<Vec<Complex<f64>>> =
        vec![vec![Complex::new(0.0, 0.0); nuclei.len() * 3]; nuclei.len() * 3];
    for i in 0..nuclei.len() {
        for j in i..nuclei.len() {
            if i == j {
                // Calculate d^2E / dx_i^2
                let tmp = (0..nuclei.len())
                    .map(|j| -> Complex<f64> {
                        if i == j {
                            return Complex::new(0.0, 0.0);
                        }
                        let disp_x = nuclei[i].x - nuclei[j].x;
                        let disp_y = nuclei[i].y - nuclei[j].y;
                        let disp_z = nuclei[i].z - nuclei[j].z;
                        let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                        Complex::new(
                            -nuclei[i].charge
                                * nuclei[j].charge
                                * (dist_sqr.powf(3.0 / 2.0)
                                    - 3.0 * (disp_x * disp_x) * dist_sqr.powf(0.5))
                                / dist_sqr.powi(3),
                            0.0,
                        )
                    })
                    .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x });
                let mut density_clone = electron_density.clone();
                density_clone.map(&|x, y, z, val| -> Complex<f64> {
                    let disp_x = nuclei[i].x - x;
                    let disp_y = nuclei[i].y - y;
                    let disp_z = nuclei[i].z - z;
                    let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                    Complex::new(
                        nuclei[i].charge
                            * (dist_sqr.powf(3.0 / 2.0)
                                - 3.0 * (disp_x * disp_x) * dist_sqr.powf(0.5))
                            / dist_sqr.powi(3),
                        0.0,
                    ) * val
                });
                ret[i * 3][j * 3] = tmp + density_clone.integrate();

                // Calculate d^2E / dy_i^2
                let tmp = (0..nuclei.len())
                    .map(|j| -> Complex<f64> {
                        if i == j {
                            return Complex::new(0.0, 0.0);
                        }
                        let disp_x = nuclei[i].x - nuclei[j].x;
                        let disp_y = nuclei[i].y - nuclei[j].y;
                        let disp_z = nuclei[i].z - nuclei[j].z;
                        let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                        Complex::new(
                            -nuclei[i].charge
                                * nuclei[j].charge
                                * (dist_sqr.powf(3.0 / 2.0)
                                    - 3.0 * (disp_y * disp_y) * dist_sqr.powf(0.5))
                                / dist_sqr.powi(3),
                            0.0,
                        )
                    })
                    .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x });
                let mut density_clone = electron_density.clone();
                density_clone.map(&|x, y, z, val| -> Complex<f64> {
                    let disp_x = nuclei[i].x - x;
                    let disp_y = nuclei[i].y - y;
                    let disp_z = nuclei[i].z - z;
                    let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                    Complex::new(
                        nuclei[i].charge
                            * (dist_sqr.powf(3.0 / 2.0)
                                - 3.0 * (disp_y * disp_y) * dist_sqr.powf(0.5))
                            / dist_sqr.powi(3),
                        0.0,
                    ) * val
                });
                ret[i * 3 + 1][j * 3 + 1] = tmp + density_clone.integrate();

                // Calculate d^2E / dz_i^2
                let tmp = (0..nuclei.len())
                    .map(|j| -> Complex<f64> {
                        if i == j {
                            return Complex::new(0.0, 0.0);
                        }
                        let disp_x = nuclei[i].x - nuclei[j].x;
                        let disp_y = nuclei[i].y - nuclei[j].y;
                        let disp_z = nuclei[i].z - nuclei[j].z;
                        let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                        Complex::new(
                            -nuclei[i].charge
                                * nuclei[j].charge
                                * (dist_sqr.powf(3.0 / 2.0)
                                    - 3.0 * (disp_z * disp_z) * dist_sqr.powf(0.5))
                                / dist_sqr.powi(3),
                            0.0,
                        )
                    })
                    .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x });
                let mut density_clone = electron_density.clone();
                density_clone.map(&|x, y, z, val| -> Complex<f64> {
                    let disp_x = nuclei[i].x - x;
                    let disp_y = nuclei[i].y - y;
                    let disp_z = nuclei[i].z - z;
                    let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                    Complex::new(
                        nuclei[i].charge
                            * (dist_sqr.powf(3.0 / 2.0)
                                - 3.0 * (disp_z * disp_z) * dist_sqr.powf(0.5))
                            / dist_sqr.powi(3),
                        0.0,
                    ) * val
                });
                ret[i * 3 + 2][j * 3 + 2] = tmp + density_clone.integrate();

                // Calculate d^2E / dx_i dy_i
                let tmp = (0..nuclei.len())
                    .map(|j| -> Complex<f64> {
                        if i == j {
                            return Complex::new(0.0, 0.0);
                        }
                        let disp_x = nuclei[i].x - nuclei[j].x;
                        let disp_y = nuclei[i].y - nuclei[j].y;
                        let disp_z = nuclei[i].z - nuclei[j].z;
                        let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                        Complex::new(
                            3.0 * nuclei[i].charge * nuclei[j].charge * disp_x * disp_y
                                / dist_sqr.powf(5.0 / 2.0),
                            0.0,
                        )
                    })
                    .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x });
                let mut density_clone = electron_density.clone();
                density_clone.map(&|x, y, z, val| -> Complex<f64> {
                    let disp_x = nuclei[i].x - x;
                    let disp_y = nuclei[i].y - y;
                    let disp_z = nuclei[i].z - z;
                    let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                    3.0 * nuclei[i].charge * val * disp_x * disp_y / dist_sqr.powf(5.0 / 2.0)
                });
                ret[i * 3][j * 3 + 1] = tmp + density_clone.integrate();
                ret[j * 3 + 1][i * 3] = ret[i * 3][j * 3 + 1];

                // Calculate d^2E / dx_i dz_i
                let tmp = (0..nuclei.len())
                    .map(|j| -> Complex<f64> {
                        if i == j {
                            return Complex::new(0.0, 0.0);
                        }
                        let disp_x = nuclei[i].x - nuclei[j].x;
                        let disp_y = nuclei[i].y - nuclei[j].y;
                        let disp_z = nuclei[i].z - nuclei[j].z;
                        let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                        Complex::new(
                            3.0 * nuclei[i].charge * nuclei[j].charge * disp_x * disp_z
                                / dist_sqr.powf(5.0 / 2.0),
                            0.0,
                        )
                    })
                    .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x });
                let mut density_clone = electron_density.clone();
                density_clone.map(&|x, y, z, val| -> Complex<f64> {
                    let disp_x = nuclei[i].x - x;
                    let disp_y = nuclei[i].y - y;
                    let disp_z = nuclei[i].z - z;
                    let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                    3.0 * nuclei[i].charge * val * disp_x * disp_z / dist_sqr.powf(5.0 / 2.0)
                });
                ret[i * 3][j * 3 + 2] = tmp + density_clone.integrate();
                ret[j * 3 + 2][i * 3] = ret[i * 3][j * 3 + 2];

                // Calculate d^2E / dy_i dz_i
                let tmp = (0..nuclei.len())
                    .map(|j| -> Complex<f64> {
                        if i == j {
                            return Complex::new(0.0, 0.0);
                        }
                        let disp_x = nuclei[i].x - nuclei[j].x;
                        let disp_y = nuclei[i].y - nuclei[j].y;
                        let disp_z = nuclei[i].z - nuclei[j].z;
                        let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                        Complex::new(
                            3.0 * nuclei[i].charge * nuclei[j].charge * disp_y * disp_z
                                / dist_sqr.powf(5.0 / 2.0),
                            0.0,
                        )
                    })
                    .fold(Complex::new(0.0, 0.0), |acc, x| -> Complex<f64> { acc + x });
                let mut density_clone = electron_density.clone();
                density_clone.map(&|x, y, z, val| -> Complex<f64> {
                    let disp_x = nuclei[i].x - x;
                    let disp_y = nuclei[i].y - y;
                    let disp_z = nuclei[i].z - z;
                    let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;
                    3.0 * nuclei[i].charge * val * disp_y * disp_z / dist_sqr.powf(5.0 / 2.0)
                });
                ret[i * 3 + 1][j * 3 + 2] = tmp + density_clone.integrate();
                ret[j * 3 + 2][i * 3 + 1] = ret[i * 3 + 1][j * 3 + 2];
            } else {
                let disp_x = nuclei[i].x - nuclei[j].x;
                let disp_y = nuclei[i].y - nuclei[j].y;
                let disp_z = nuclei[i].z - nuclei[j].z;
                let dist_sqr = disp_x * disp_x + disp_y * disp_y + disp_z * disp_z;

                // Calculate d^2E / dx_i dx_j
                ret[i * 3][j * 3] = Complex::new(
                    -nuclei[i].charge
                        * nuclei[j].charge
                        * (3.0 * disp_x * dist_sqr.powf(0.5) - dist_sqr.powf(3.0 / 2.0))
                        / dist_sqr.powi(3),
                    0.0,
                );
                ret[j * 3][i * 3] = ret[i * 3][j * 3];

                // Calculate d^2E / dy_i dy_j
                ret[i * 3 + 1][j * 3 + 1] = Complex::new(
                    -nuclei[i].charge
                        * nuclei[j].charge
                        * (3.0 * disp_y * dist_sqr.powf(0.5) - dist_sqr.powf(3.0 / 2.0))
                        / dist_sqr.powi(3),
                    0.0,
                );
                ret[j * 3 + 1][i * 3 + 1] = ret[i * 3 + 1][j * 3 + 1];

                // Calculate d^2E / dz_i dz_j
                ret[i * 3 + 2][j * 3 + 2] = Complex::new(
                    -nuclei[i].charge
                        * nuclei[j].charge
                        * (3.0 * disp_z * dist_sqr.powf(0.5) - dist_sqr.powf(3.0 / 2.0))
                        / dist_sqr.powi(3),
                    0.0,
                );
                ret[j * 3 + 2][i * 3 + 2] = ret[i * 3 + 2][j * 3 + 2];

                // Calculate d^2E / dx_i dy_j
                ret[i * 3][j * 3 + 1] = Complex::new(
                    -nuclei[i].charge * nuclei[j].charge * 3.0 * disp_x * disp_y
                        / dist_sqr.powf(5.0 / 2.0),
                    0.0,
                );
                ret[j * 3 + 1][i * 3] = ret[i * 3][j * 3 + 1];

                // Calculate d^2E / dx_i dz_j
                ret[i * 3][j * 3 + 2] = Complex::new(
                    -nuclei[i].charge * nuclei[j].charge * 3.0 * disp_x * disp_z
                        / dist_sqr.powf(5.0 / 2.0),
                    0.0,
                );
                ret[j * 3 + 2][i * 3] = ret[i * 3][j * 3 + 2];

                // Calculate d^2E / dy_i dx_j
                ret[i * 3 + 1][j * 3] = ret[i * 3][j * 3 + 1];
                ret[j * 3][i * 3 + 1] = ret[i * 3 + 1][j * 3];

                // Calculate d^2E / dy_i dz_j
                ret[i * 3 + 1][j * 3 + 2] = Complex::new(
                    -nuclei[i].charge * nuclei[j].charge * 3.0 * disp_y * disp_z
                        / dist_sqr.powf(5.0 / 2.0),
                    0.0,
                );
                ret[j * 3 + 2][i * 3 + 1] = ret[i * 3 + 1][j * 3 + 2];

                // Calculate d^2E / dz_i dx_j
                ret[i * 3 + 2][j * 3] = ret[i * 3][j * 3 + 2];
                ret[j * 3][i * 3 + 2] = ret[i * 3 + 2][j * 3];

                // Calcualte d^2E / dz_i dy_j
                ret[i * 3 + 2][j * 3 + 1] = ret[i * 3 + 1][j * 3 + 2];
                ret[j * 3 + 1][i * 3 + 2] = ret[i * 3 + 2][j * 3 + 1];
            }
        }
    }

    ret.try_into().unwrap()
}

mod tests {
    use test_log::test;

    use super::*;
    use crate::basis::caching_basis::CachingBasis;
    use crate::basis::contracted_basis::ContractedBasis;
    use crate::basis::gaussian_type_orbital::GTO;
    use crate::functional::lda::lda_functional;
    use crate::functional::lda::lda_potential_functional;
    use crate::grid::GridConfig;
    use crate::scf::SCF;

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

    // Reference value adapted from https://pubs.acs.org/doi/10.1021/ed5004788
    #[test]
    fn test_hydrogen_nuclear_potential() {
        let mut test_gto = GTO::new(0.0, 0.0, 0.0, 0.25, 0, 0, 0, true);
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
            (integral - -0.798).abs() < 0.01,
            "Incorrect hydrogen nuclear-electron energy! Expected {} Actual {}",
            -0.798,
            integral
        );
    }

    #[test]
    fn test_molecular_hydrogen_gradient() {
        ///////////////////////////////////////////////////////////////////////////////////
        // This section is adapted from the "test_scf_molecular_hydrogen" test in scf.rs //
        ///////////////////////////////////////////////////////////////////////////////////
        // Experimental H-H bond length is 0.7414A.
        // Source: https://cccbdb.nist.gov/exp2x.asp?casno=1333740
        // Note that all of our calculations are in atomic units though, not Angstroms, so we
        // convert to Bohrs.
        let bond_length: f64 = 1.4828;
        // Adapted from https://www.basissetexchange.org/
        let basis = vec![
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(
                        -bond_length / 2.0,
                        0.0,
                        0.0,
                        0.3425250914E+01,
                        0,
                        0,
                        0,
                        false,
                    ),
                    GTO::new(0.0, 0.0, 0.0, 0.6239137298E+00, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1688554040E+00, 0, 0, 0, false),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(
                        bond_length / 2.0,
                        0.0,
                        0.0,
                        0.3425250914E+01,
                        0,
                        0,
                        0,
                        false,
                    ),
                    GTO::new(0.0, 0.0, 0.0, 0.6239137298E+00, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1688554040E+00, 0, 0, 0, false),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
        ];
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
            &vec![nucleus1.clone(), nucleus2.clone()],
            basis,
            2,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        let converged = scf.iterate(1E-10, 10).is_ok();
        assert!(converged, "Hydrogen molecule SCF failed to converge!");

        let grads = nuclear_gradients(&vec![nucleus1, nucleus2], scf.electron_density.clone());
        let expected = 0.0;
        assert!(
            (grads.0[0].re - expected).abs() < 0.01,
            "Molecular hydrogen X gradient expected to be 0! Actual: {:?}",
            grads
        );
        assert!(
            (grads.0[1].re - expected).abs() < 0.01,
            "Molecular hydrogen Y gradient expected to be 0! Actual: {:?}",
            grads
        );
        assert!(
            (grads.0[2].re - expected).abs() < 0.01,
            "Molecular hydrogen Z gradient expected to be 0! Actual: {:?}",
            grads
        );
    }

    #[test]
    fn test_molecular_hydrogen_hessian() {
        ///////////////////////////////////////////////////////////////////////////////////
        // This section is adapted from the "test_scf_molecular_hydrogen" test in scf.rs //
        ///////////////////////////////////////////////////////////////////////////////////
        // Experimental H-H bond length is 0.7414A.
        // Source: https://cccbdb.nist.gov/exp2x.asp?casno=1333740
        // Note that all of our calculations are in atomic units though, not Angstroms, so we
        // convert to Bohrs.
        let bond_length: f64 = 1.4828;
        // Adapted from https://www.basissetexchange.org/
        let basis = vec![
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(
                        -bond_length / 2.0,
                        0.0,
                        0.0,
                        0.3425250914E+01,
                        0,
                        0,
                        0,
                        false,
                    ),
                    GTO::new(0.0, 0.0, 0.0, 0.6239137298E+00, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1688554040E+00, 0, 0, 0, false),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(
                        bond_length / 2.0,
                        0.0,
                        0.0,
                        0.3425250914E+01,
                        0,
                        0,
                        0,
                        false,
                    ),
                    GTO::new(0.0, 0.0, 0.0, 0.6239137298E+00, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1688554040E+00, 0, 0, 0, false),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
        ];
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
            &vec![nucleus1.clone(), nucleus2.clone()],
            basis,
            2,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        let converged = scf.iterate(1E-10, 10).is_ok();
        assert!(converged, "Hydrogen molecule SCF failed to converge!");

        let fake_bond_length = 0.8;
        let mut nucleus1 = Nucleus {
            x: -fake_bond_length / 2.0,
            y: 0.0,
            z: 0.0,
            charge: 1.0,
        };
        let mut nucleus2 = Nucleus {
            x: fake_bond_length / 2.0,
            y: 0.0,
            z: 0.0,
            charge: 1.0,
        };
        let grads = nuclear_gradients(
            &vec![nucleus1.clone(), nucleus2.clone()],
            scf.electron_density.clone(),
        );
        let hessian = nuclear_hessian(
            &vec![nucleus1.clone(), nucleus2.clone()],
            scf.electron_density.clone(),
        );
        let step = hessian.clone().inverse(1E-20) * grads.clone();
        nucleus1.x += step.0[0].re;
        nucleus1.y += step.0[1].re;
        nucleus1.z += step.0[2].re;
        nucleus2.x += step.0[3].re;
        nucleus2.y += step.0[4].re;
        nucleus2.z += step.0[5].re;
        let corrected_grads = nuclear_gradients(
            &vec![nucleus1.clone(), nucleus2.clone()],
            scf.electron_density.clone(),
        );
        assert!(
            corrected_grads.l2().re <= 0.1,
            "Error computing hessian!\nHessian: {:?}\nGrads: {:?}\nStep: {:?}\nCorrected Grads: {:?}",
            hessian,
            grads,
            step,
            corrected_grads,
        );
    }
}
