use num::complex::Complex;

use crate::grid::Grid;

pub mod gaussian_type_orbital;

pub trait Ansatz {
    // Set the contraction coefficient for this Ansatz.
    fn set_coefficient(&mut self, coefficient: f64);

    // Evaluate the position operator at coordinates (x, y, z).
    fn pos(&self, x: f64, y: f64, z: f64) -> Complex<f64>;

    // Evaluate Laplacian operator at coordinates (x, y, z).
    fn laplacian(&self, x: f64, y: f64, z: f64) -> Complex<f64>;

    // Fill a grid with the position operator.
    fn ket(&self, grid: &mut Grid) {
        grid.fill(&|x, y, z| -> Complex<f64> { self.pos(x, y, z) });
    }

    // Fill a grid with the complex conjugate of the position operator.
    fn bra(&self, grid: &mut Grid) {
        grid.fill(&|x, y, z| -> Complex<f64> { self.pos(x, y, z).conj() });
    }

    // Fill a grid with the output of the kinetic energy operator.
    fn kinetic_energy(&self, grid: &mut Grid) {
        grid.fill(&|x, y, z| -> Complex<f64> { -0.5 * self.laplacian(x, y, z) });
    }
}
