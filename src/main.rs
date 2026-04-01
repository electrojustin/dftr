use std::collections::HashMap;

use anyhow::anyhow;
use anyhow::Result;
use argh::FromArgs;
use env_logger::Env;
use num::complex::Complex;

use crate::elements::get_nuclei_and_basis;
use crate::elements::parse_basis_config_file;
use crate::elements::symbol_to_atomic_number;
use crate::elements::BasisConfig;
use crate::functional::lda::lda_functional;
use crate::functional::lda::lda_potential_functional;
use crate::grid::Grid;
use crate::grid::GridConfig;
use crate::nucleus::nuclear_gradients;
use crate::nucleus::nuclear_hessian;
use crate::pdb::angstrom_to_bohr;
use crate::pdb::parse_pdb;
use crate::pdb::write_pdb;
use crate::scf::SCF;

mod basis;
mod elements;
mod functional;
mod grid;
mod linear;
mod nucleus;
mod pdb;
mod scf;
mod utils;

#[derive(FromArgs)]
/// Rust based Density Functional Theory
struct Args {
    /// input PDB file
    #[argh(option)]
    input: String,

    /// output PDB file
    #[argh(option)]
    output: String,

    /// file containing basis configuration in Gaussian format
    #[argh(option)]
    basis_file: String,

    /// width (Angstrom) of the periodic bounding box
    #[argh(option)]
    width: f64,

    /// height (Angstrom) of the periodic bounding box
    #[argh(option)]
    height: f64,

    /// depth (Angstrom) of the periodic bounding box
    #[argh(option)]
    depth: f64,

    /// number of grid points to use on X axis
    #[argh(option, default = "128")]
    x_grid: usize,

    /// number of grid points to use on Y axis
    #[argh(option, default = "128")]
    y_grid: usize,

    /// number of grid points to use on Z axis
    #[argh(option, default = "128")]
    z_grid: usize,

    /// max number of SCF cycles to try before giving up
    #[argh(option, default = "1000")]
    iter_scf: usize,

    /// tolerance for SCF convergence
    #[argh(option, default = "1E-4")]
    tolerance_scf: f64,

    /// max number of geometry optimization cycles to try before giving up
    #[argh(option, default = "1000")]
    iter_geo: usize,

    /// tolerance for geometry optimization convergence
    #[argh(option, default = "1E-2")]
    tolerance_geo: f64,

    /// damping coefficient for geometry optimization Newton-Raphson
    #[argh(option, default = "0.25")]
    damping_geo: f64,

    /// whether or not to optimize geometry
    #[argh(switch, short = 'g')]
    geometry_optimize: bool,
}

fn geometry_optimize(
    args: &Args,
    element_symbols: &Vec<String>,
    coords: &mut Vec<(f64, f64, f64)>,
    basis_configs: &HashMap<String, BasisConfig>,
) -> Result<()> {
    // Validate atomic symbols and count number of electrons
    let mut num_electrons = 0;
    for i in 0..element_symbols.len() {
        num_electrons += symbol_to_atomic_number(&element_symbols[i])?;
    }
    log::debug!("Num electrons: {}", num_electrons);

    let grid_width = angstrom_to_bohr(args.width);
    let grid_height = angstrom_to_bohr(args.height);
    let grid_depth = angstrom_to_bohr(args.depth);
    let grid_config = GridConfig {
        start_x: -grid_width / 2.0,
        start_y: -grid_height / 2.0,
        start_z: -grid_depth / 2.0,
        end_x: grid_width / 2.0,
        end_y: grid_height / 2.0,
        end_z: grid_depth / 2.0,
        width_voxels: args.x_grid,
        height_voxels: args.y_grid,
        depth_voxels: args.z_grid,
    };

    let mut prev_energy = Complex::new(0.0, 0.0);
    for iter in 0..args.iter_geo {
        // Recenter coordinates
        let mut center_x = 0.0;
        let mut center_y = 0.0;
        let mut center_z = 0.0;
        for coord in coords.iter() {
            center_x += coord.0;
            center_y += coord.1;
            center_z += coord.2;
        }
        center_x /= coords.len() as f64;
        center_y /= coords.len() as f64;
        center_z /= coords.len() as f64;
        for i in 0..coords.len() {
            coords[i] = (
                coords[i].0 - center_x,
                coords[i].1 - center_y,
                coords[i].2 - center_z,
            );
        }

        log::debug!("Coords: {:?}", coords);

        // Setup nuclei and basis
        let (nuclei, basis) = get_nuclei_and_basis(element_symbols, coords, basis_configs)?;

        // Compute molecular orbitals using SCF
        log::debug!("Initializing SCF");
        let mut scf_state = SCF::new(
            &nuclei,
            basis,
            num_electrons,
            |density| -> Grid { lda_functional(density, 1.05 * 2.0 / 3.0) },
            |density| -> Grid { lda_potential_functional(density, 1.05 * 2.0 / 3.0) },
            grid_config.clone(),
        );
        log::debug!("Running SCF");
        scf_state.iterate(args.tolerance_scf, args.iter_scf)?;

        // Break if energy isn't changing
        if (scf_state.energy - prev_energy).norm_sqr() < args.tolerance_geo {
            log::info!("Geometry converged!");
            return Ok(());
        }
        prev_energy = scf_state.energy;

        // Compute gradient of nucleus-electron attraction energy with respect to nuclear coordinates
        let grads = nuclear_gradients(&nuclei, scf_state.electron_density.clone());

        // Break if gradients are close to 0
        if grads.l2().re < args.tolerance_geo {
            log::info!("Geometry converged!");
            return Ok(());
        } else {
            log::info!(
                "Geometry optimization iteration {}. Gradient length: {}",
                iter,
                grads.l2().re
            );
        }

        // Optimize geometry using a Newton-Raphson iteration
        let hessian = nuclear_hessian(&nuclei, scf_state.electron_density.clone());
        let step = Complex::new(args.damping_geo, 0.0) * (hessian.inverse(1E-10) * grads);
        for i in 0..coords.len() {
            coords[i] = (
                coords[i].0 - step.0[i * 3].re,
                coords[i].1 - step.0[i * 3 + 1].re,
                coords[i].2 - step.0[i * 3 + 2].re,
            );
        }
    }

    Err(anyhow!("Geometry failed to converge!"))
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let args: Args = argh::from_env();
    if args.x_grid.count_ones() != 1
        || args.y_grid.count_ones() != 1
        || args.z_grid.count_ones() != 1
    {
        log::warn!("WARNING: Non-power of two grid dimenions may result in very slow electron-electron repulsion calculations");
    }

    let basis_configs = match parse_basis_config_file(&args.basis_file) {
        Ok(basis_configs) => basis_configs,
        Err(e) => {
            log::error!("Error parsing basis config! {:?}", e);
            return;
        }
    };
    let (element_symbols, mut coords) = match parse_pdb(&args.input) {
        Ok((element_symbols, coords)) => (element_symbols, coords),
        Err(e) => {
            log::error!("Error parsing pdb! {:?}", e);
            return;
        }
    };

    if args.geometry_optimize {
        if let Err(e) = geometry_optimize(&args, &element_symbols, &mut coords, &basis_configs) {
            log::error!("Error optimizing geometry! {:?}", e);
            return;
        }
    }

    if let Err(e) = write_pdb(&args.input, &args.output, &coords, &None) {
        log::error!("Error writing output PDB! {:?}", e);
    }
}
