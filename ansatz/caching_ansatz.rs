use std::collections::HashMap;

use num::complex::Complex;

use crate::ansatz::Ansatz;
use crate::grid::Grid;
use crate::grid::GridConfig;

pub struct CachingAnsatz<T: Ansatz> {
    delegate: T,
    ket_cache: HashMap<GridConfig, Grid>,
    bra_cache: HashMap<GridConfig, Grid>,
    kinetic_energy_cache: HashMap<GridConfig, Grid>,
}

impl<T: Ansatz> CachingAnsatz<T> {
    pub fn new(delegate: T) -> Self {
        CachingAnsatz {
            delegate,
            ket_cache: HashMap::new(),
            bra_cache: HashMap::new(),
            kinetic_energy_cache: HashMap::new(),
        }
    }
}

impl<T: Ansatz> Ansatz for CachingAnsatz<T> {
    fn pos(&self, x: f64, y: f64, z: f64) -> Complex<f64> {
        self.delegate.pos(x, y, z)
    }

    fn laplacian(&self, x: f64, y: f64, z: f64) -> Complex<f64> {
        self.delegate.laplacian(x, y, z)
    }

    fn ket(&mut self, grid_config: GridConfig) -> Grid {
        self.ket_cache
            .entry(grid_config.clone())
            .or_insert_with(|| -> Grid { self.delegate.ket(grid_config) })
            .clone()
    }

    fn bra(&mut self, grid_config: GridConfig) -> Grid {
        self.bra_cache
            .entry(grid_config.clone())
            .or_insert_with(|| -> Grid { self.delegate.bra(grid_config) })
            .clone()
    }

    fn kinetic_energy(&mut self, grid_config: GridConfig) -> Grid {
        self.kinetic_energy_cache
            .entry(grid_config.clone())
            .or_insert_with(|| -> Grid { self.delegate.kinetic_energy(grid_config) })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ansatz::gaussian_type_orbital::GTO;
    use crate::grid::Grid;
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

    #[test]
    fn test_caching_ansatz_proxies() {
        let mut gto = GTO::new(0.0, 0.0, 0.0, 0.25, 0, 0, 0);
        let expected_bra = gto.bra(K_GRID_CONFIG);
        let expected_ket = gto.ket(K_GRID_CONFIG);
        let expected_kinetic_energy = gto.kinetic_energy(K_GRID_CONFIG);
        let mut caching_ansatz = CachingAnsatz::new(gto);
        let actual_bra = caching_ansatz.bra(K_GRID_CONFIG);
        let actual_ket = caching_ansatz.ket(K_GRID_CONFIG);
        let actual_kinetic_energy = caching_ansatz.kinetic_energy(K_GRID_CONFIG);
        assert!(
            (actual_bra - expected_bra).integrate().re.abs() < 0.1,
            "Caching bra doesn't match delegate bra!"
        );
        assert!(
            (actual_ket - expected_ket).integrate().re.abs() < 0.1,
            "Caching ket doesn't match delegate ket!"
        );
        assert!(
            (actual_kinetic_energy - expected_kinetic_energy)
                .integrate()
                .re
                .abs()
                < 0.1,
            "Caching kinetic energy doesn't match delegate kinetic energy!"
        );
    }
}
