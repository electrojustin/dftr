use crate::grid::Grid;

mod ansatz;
mod grid;
mod nucleus;
mod utils;

fn main() {
    let mut test = Grid::new(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 100, 100, 100);
    println!("{:?}", test);
}
