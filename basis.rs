use num::complex::Complex;

use crate::grid::Grid;
use crate::grid::GridConfig;

pub mod caching_basis;
pub mod gaussian_type_orbital;
pub mod sto_ng;

pub trait Basis {
    // Evaluate the position operator at coordinates (x, y, z).
    fn pos(&self, x: f64, y: f64, z: f64) -> Complex<f64>;

    // Evaluate Laplacian operator at coordinates (x, y, z).
    fn laplacian(&self, x: f64, y: f64, z: f64) -> Complex<f64>;

    // Fill a grid with the position operator.
    fn ket(&mut self, grid_config: GridConfig) -> Grid {
        let mut grid = Grid::new(grid_config);
        grid.fill(&|x, y, z| -> Complex<f64> { self.pos(x, y, z) });
        grid
    }

    // Fill a grid with the complex conjugate of the position operator.
    fn bra(&mut self, grid_config: GridConfig) -> Grid {
        let mut grid = Grid::new(grid_config);
        grid.fill(&|x, y, z| -> Complex<f64> { self.pos(x, y, z).conj() });
        grid
    }

    // Fill a grid with the output of the kinetic energy operator. Needs a bra to dot product with
    // to yield actual kinetic energy grid.
    fn kinetic_energy(&mut self, grid_config: GridConfig) -> Grid {
        let mut grid = Grid::new(grid_config);
        grid.fill(&|x, y, z| -> Complex<f64> { -0.5 * self.laplacian(x, y, z) });
        grid
    }
}
