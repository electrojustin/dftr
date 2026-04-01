use std::fs::read_to_string;
use std::fs::File;
use std::io::Write;
use std::str::FromStr;

use anyhow::anyhow;
use anyhow::Result;

pub fn angstrom_to_bohr(angstrom: f64) -> f64 {
    angstrom * 1.88973
}

pub fn bohr_to_angstrom(bohr: f64) -> f64 {
    bohr / 1.88973
}

pub fn parse_pdb(filename: &str) -> Result<(Vec<String>, Vec<(f64, f64, f64)>)> {
    let mut element_symbols: Vec<String> = Vec::new();
    let mut coords: Vec<(f64, f64, f64)> = Vec::new();
    for line in read_to_string(filename)?.split('\n') {
        if line.len() < 6 {
            continue;
        }
        let line_chars: Vec<char> = line.chars().collect();
        let record_type = line_chars[0..6]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
        if record_type != "ATOM" && record_type != "HETATM" {
            continue;
        }
        element_symbols.push(
            line_chars[76..78]
                .iter()
                .collect::<String>()
                .trim()
                .to_string(),
        );
        let x_coord = f64::from_str(&line_chars[30..38].iter().collect::<String>().trim())?;
        let y_coord = f64::from_str(&line_chars[38..46].iter().collect::<String>().trim())?;
        let z_coord = f64::from_str(&line_chars[46..54].iter().collect::<String>().trim())?;
        coords.push((
            angstrom_to_bohr(x_coord),
            angstrom_to_bohr(y_coord),
            angstrom_to_bohr(z_coord),
        ));
    }
    Ok((element_symbols, coords))
}

pub fn write_pdb(
    template_filename: &str,
    output_filename: &str,
    new_coords: &Vec<(f64, f64, f64)>,
    charges: &Option<Vec<f64>>,
) -> Result<()> {
    if let Some(charges) = charges.as_ref() {
        if new_coords.len() != charges.len() {
            return Err(anyhow!("New coords and charges are not same length!"));
        }
    }

    let mut out_file = File::create(output_filename)?;
    let mut i = 0;
    for line in read_to_string(template_filename)?.split('\n') {
        let mut line_chars: Vec<char> = line.chars().collect();
        if line_chars.len() < 66 {
            out_file.write(format!("{}\n", line).as_bytes());
            continue;
        }
        let record_type = line_chars[0..6]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
        if record_type != "ATOM" && record_type != "HETATM" {
            out_file.write(format!("{}\n", line).as_bytes());
            continue;
        }
        let coords = format!(
            "{:>8.3}{:>8.3}{:>8.3}",
            bohr_to_angstrom(new_coords[i].0),
            bohr_to_angstrom(new_coords[i].1),
            bohr_to_angstrom(new_coords[i].2)
        );
        line_chars[30..54].copy_from_slice(coords.chars().collect::<Vec<char>>().as_slice());
        // Partial charges are traditionally put in the B-factor field.
        if let Some(charges) = charges.as_ref() {
            let charge = format!("{:>8.3}", charges[i]);
            line_chars[60..66].copy_from_slice(charge.chars().collect::<Vec<char>>().as_slice());
        }
        i += 1;
        out_file.write(format!("{}\n", line_chars.iter().collect::<String>()).as_bytes());
    }
    Ok(())
}

mod tests {
    use test_log::test;

    use super::*;

    #[test]
    fn test_parse_water() {
        let (element_symbols, coords) =
            parse_pdb(&(env!("CARGO_MANIFEST_DIR").to_string() + "/test/water.pdb"))
                .expect("Error parsing PDB!");
        let is_correct = element_symbols.len() == 3
            && element_symbols[0] == "O"
            && element_symbols[1] == "H"
            && element_symbols[2] == "H";
        assert!(
            is_correct,
            "Error parsing water PDB! Wrong element symbols: {:?}",
            element_symbols
        );
        let is_correct = coords.len() == 3
            && coords[0].0.abs() < 0.1
            && coords[0].1.abs() < 0.1
            && coords[0].2.abs() < 0.1
            && (coords[1].0 - 1.88973).abs() < 0.1
            && coords[1].1.abs() < 0.1
            && coords[1].2.abs() < 0.1
            && coords[2].0.abs() < 0.1
            && (coords[2].1 - 1.88973).abs() < 0.1
            && coords[2].2.abs() < 0.1;
        assert!(
            is_correct,
            "Error parsing water PDB! Wrong coords: {:?}",
            coords
        );
    }
}
