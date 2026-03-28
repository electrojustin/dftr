use std::collections::HashMap;

use anyhow::anyhow;
use anyhow::Result;

pub fn symbol_to_atomic_number(symbol: &str) -> Result<usize> {
    match symbol {
        "H" => Ok(1),
        "He" => Ok(2),
        "Li" => Ok(3),
        "Be" => Ok(4),
        "B" => Ok(5),
        "C" => Ok(6),
        "N" => Ok(7),
        "O" => Ok(8),
        "F" => Ok(9),
        "Ne" => Ok(10),
        "Na" => Ok(11),
        "Mg" => Ok(12),
        "Al" => Ok(13),
        "Si" => Ok(14),
        "P" => Ok(15),
        "S" => Ok(16),
        "Cl" => Ok(17),
        "Ar" => Ok(18),
        "K" => Ok(19),
        "Ca" => Ok(20),
        "Sc" => Ok(21),
        "Ti" => Ok(22),
        "V" => Ok(23),
        "Cr" => Ok(24),
        "Mn" => Ok(25),
        "Fe" => Ok(26),
        "Co" => Ok(27),
        "Ni" => Ok(28),
        "Cu" => Ok(29),
        "Zn" => Ok(30),
        "Ga" => Ok(31),
        "Ge" => Ok(32),
        "As" => Ok(33),
        "Se" => Ok(34),
        "Br" => Ok(35),
        "Kr" => Ok(36),
        "Rb" => Ok(37),
        "Sr" => Ok(38),
        "Y" => Ok(39),
        "Zr" => Ok(40),
        "Nb" => Ok(41),
        "Mo" => Ok(42),
        "Tc" => Ok(43),
        "Ru" => Ok(44),
        "Rh" => Ok(45),
        "Pd" => Ok(46),
        "Ag" => Ok(47),
        "Cd" => Ok(48),
        "In" => Ok(49),
        "Sn" => Ok(50),
        "Sb" => Ok(51),
        "Te" => Ok(52),
        "I" => Ok(53),
        "Xe" => Ok(54),
        "Cs" => Ok(55),
        "Ba" => Ok(56),
        "La" => Ok(57),
        "Ce" => Ok(58),
        "Pr" => Ok(59),
        "Nd" => Ok(60),
        "Pm" => Ok(61),
        "Sm" => Ok(62),
        "Eu" => Ok(63),
        "Gd" => Ok(64),
        "Tb" => Ok(65),
        "Dy" => Ok(66),
        "Ho" => Ok(67),
        "Er" => Ok(68),
        "Tm" => Ok(69),
        "Yb" => Ok(70),
        "Lu" => Ok(71),
        "Hf" => Ok(72),
        "Ta" => Ok(73),
        "W" => Ok(74),
        "Re" => Ok(75),
        "Os" => Ok(76),
        "Ir" => Ok(77),
        "Pt" => Ok(78),
        "Au" => Ok(79),
        "Hg" => Ok(80),
        "Tl" => Ok(81),
        "Pb" => Ok(82),
        "Bi" => Ok(83),
        "Po" => Ok(84),
        "At" => Ok(85),
        "Rn" => Ok(86),
        "Fr" => Ok(87),
        "Ra" => Ok(88),
        "Ac" => Ok(89),
        "Th" => Ok(90),
        "Pa" => Ok(91),
        "U" => Ok(92),
        "Np" => Ok(93),
        "Pu" => Ok(94),
        "Am" => Ok(95),
        "Cm" => Ok(96),
        "Bk" => Ok(97),
        "Cf" => Ok(98),
        "Es" => Ok(99),
        "Fm" => Ok(100),
        "Md" => Ok(101),
        "No" => Ok(102),
        "Lr" => Ok(103),
        "Rf" => Ok(104),
        "Db" => Ok(105),
        "Sg" => Ok(106),
        "Bh" => Ok(107),
        "Hs" => Ok(108),
        "Mt" => Ok(109),
        "Ds" => Ok(110),
        "Rg" => Ok(111),
        "Cn" => Ok(112),
        "Nh" => Ok(113),
        "Fl" => Ok(114),
        "Mc" => Ok(115),
        "Lv" => Ok(116),
        "Ts" => Ok(117),
        "Og" => Ok(118),
        _ => Err(anyhow!("Unknown element {}", symbol)),
    }
}

fn combinatorial_helper(max_val: usize, depth: usize) -> Vec<Vec<usize>> {
    assert!(
        depth != 0,
        "Error in combinatorial helper! Depth cannot be 0"
    );
    if depth == 1 {
        vec![vec![max_val]]
    } else {
        let mut ret: Vec<Vec<usize>> = vec![];
        for i in 0..(max_val + 1) {
            let mut next_vals = combinatorial_helper(max_val - i, depth - 1);
            for val in next_vals.iter_mut() {
                val.push(i);
            }
            ret.append(&mut next_vals);
        }
        ret
    }
}

pub fn shell_to_gto_ijk(shell: &str) -> Result<Vec<(usize, usize, usize)>> {
    let combos = match shell {
        "s" => combinatorial_helper(0, 3),
        "p" => combinatorial_helper(1, 3),
        "d" => combinatorial_helper(2, 3),
        "f" => combinatorial_helper(3, 3),
        "g" => combinatorial_helper(4, 3),
        _ => return Err(anyhow!("Shell unsupported: {}", shell)),
    };

    Ok(combos
        .into_iter()
        .map(|x| -> (usize, usize, usize) { (x[0], x[1], x[2]) })
        .collect())
}

mod tests {
    use super::*;

    #[test]
    fn test_shell_to_gto_ijk() {
        let ijks = shell_to_gto_ijk("d").expect("Could not create IJKs for D shell!");
        assert!(
            ijks.len() == 6,
            "Error in shells_to_gto_ijk! Invalid number of combos {}",
            ijks.len()
        );
        assert!(
            ijks.contains(&(0, 0, 2)),
            "Error in shells_to_gto_ijk! No combo (0, 0, 2)"
        );
        assert!(
            ijks.contains(&(0, 2, 0)),
            "Error in shells_to_gto_ijk! No combo (0, 2, 0)"
        );
        assert!(
            ijks.contains(&(2, 0, 0)),
            "Error in shells_to_gto_ijk! No combo (2, 0, 0)"
        );
        assert!(
            ijks.contains(&(0, 1, 1)),
            "Error in shells_to_gto_ijk! No combo (0, 1, 1)"
        );
        assert!(
            ijks.contains(&(1, 0, 1)),
            "Error in shells_to_gto_ijk! No combo (1, 0, 1)"
        );
        assert!(
            ijks.contains(&(1, 1, 0)),
            "Error in shells_to_gto_ijk! No combo (1, 1, 0)"
        );
    }
}
