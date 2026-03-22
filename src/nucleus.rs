use num::complex::Complex;

use crate::basis::Basis;
use crate::grid::Grid;
use crate::grid::GridConfig;

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
                // Cap the dispance at 0.1 A to avoid divide by 0 numerical instability.
                let dispance = (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z)
                    .sqrt()
                    .max(0.1);
                Complex::new(-nucleus.charge / dispance, 0.0)
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

pub fn nuclear_gradients(
    nuclei: &Vec<Nucleus>,
    electron_density: Grid,
) -> Vec<(Complex<f64>, Complex<f64>, Complex<f64>)> {
    let mut grads: Vec<(Complex<f64>, Complex<f64>, Complex<f64>)> = Vec::new();
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
            let intermediate = nuclei[i].charge * nuclei[j].charge
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).powf(3.0 / 2.0);
            grad_x += disp_x * intermediate;
            grad_y += disp_y * intermediate;
            grad_z += disp_z * intermediate;
        }
        grads.push((grad_x, grad_y, grad_z));
    }
    println!("{:?}", grads);

    for i in 0..nuclei.len() {
        let mut grad_x = electron_density.clone();
        grad_x.map(&|x, y, z, val| -> Complex<f64> {
            let disp_x = nuclei[i].x - x;
            let disp_y = nuclei[i].y - y;
            let disp_z = nuclei[i].z - z;
            -nuclei[i].charge * val * disp_x
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).powf(3.0 / 2.0)
        });
        let mut grad_y = electron_density.clone();
        grad_y.map(&|x, y, z, val| -> Complex<f64> {
            let disp_x = nuclei[i].x - x;
            let disp_y = nuclei[i].y - y;
            let disp_z = nuclei[i].z - z;
            -nuclei[i].charge * val * disp_y
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).powf(3.0 / 2.0)
        });
        let mut grad_z = electron_density.clone();
        grad_z.map(&|x, y, z, val| -> Complex<f64> {
            let disp_x = nuclei[i].x - x;
            let disp_y = nuclei[i].y - y;
            let disp_z = nuclei[i].z - z;
            -nuclei[i].charge * val * disp_z
                / (disp_x * disp_x + disp_y * disp_y + disp_z * disp_z).powf(3.0 / 2.0)
        });
        grads[i] = (
            grads[i].0 + grad_x.integrate(),
            grads[i].1 + grad_y.integrate(),
            grads[i].2 + grad_z.integrate(),
        );
    }

    grads
}

mod tests {
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
        //////////////////////////////////////////////////////////////////////////////////
        // This section is copied from the "test_scf_molecular_hydrogen" test in scf.rs //
        //////////////////////////////////////////////////////////////////////////////////
        // Experimental H-H bond length is 0.7414A.
        // Source: https://cccbdb.nist.gov/exp2x.asp?casno=1333740
        // Note that all of our calculations are in atomic units though, not Angstroms, so we
        // convert to Bohrs.
        let bond_length: f64 = 1.40104295;
        let basis1 = ContractedBasis::sto_3g(-bond_length / 2.0, 0.0, 0.0, "1s")
            .expect("Failed to create basis function!");
        let basis2 = ContractedBasis::sto_3g(bond_length / 2.0, 0.0, 0.0, "1s")
            .expect("Failed to create basis function!");
        // Adapted from https://www.basissetexchange.org/
        /*let basis = vec![
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(-bond_length / 2.0, 0.0, 0.0, 0.3425250914E+01 , 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.6239137298E+00, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1688554040E+00 , 0, 0, 0, false),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
            CachingBasis::new(ContractedBasis::new(
                vec![
                    GTO::new(bond_length / 2.0, 0.0, 0.0, 0.3425250914E+01 , 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.6239137298E+00, 0, 0, 0, false),
                    GTO::new(0.0, 0.0, 0.0, 0.1688554040E+00 , 0, 0, 0, false),
                ],
                vec![
                    Complex::new(0.154329, 0.0),
                    Complex::new(0.535328, 0.0),
                    Complex::new(0.444635, 0.0),
                ],
            )),
        ];*/
        let basis = vec![basis1, basis2];
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
            vec![nucleus1.clone(), nucleus2.clone()],
            basis,
            2,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            K_GRID_CONFIG,
        );
        let converged = scf.iterate(1E-20, 10);
        assert!(converged, "Hydrogen molecule SCF failed to converge!");

        let grads = nuclear_gradients(&vec![nucleus1, nucleus2], scf.electron_density.clone());
        let expected = 0.0;
        assert!(
            (grads[0].0.re - expected).abs() < 0.1,
            "Molecular hydrogen gradient expected to be 0! Actual: {:?}",
            grads
        );
    }
}
