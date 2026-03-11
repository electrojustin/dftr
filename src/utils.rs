use num::Complex;

pub fn factorial(n: i32) -> i32 {
    let mut ret: i32 = 1;
    for i in 2..(n + 1) {
        ret *= i;
    }
    ret
}

// Used for odd number DFTs.
fn slow_dft(
    input: &[Complex<f64>],
    n: usize,
    stride: usize,
    offset: usize,
    inverse: bool,
) -> Vec<Complex<f64>> {
    let mut ret = Vec::with_capacity(n);
    for freq in 0..n {
        let mut acc = Complex::new(0.0, 0.0);
        for i in 0..n {
            if inverse {
                acc += 1.0 / (n as f64)
                    * input[i * stride + offset]
                    * Complex::new(
                        0.0,
                        2.0 * std::f64::consts::PI / (n as f64) * (freq as f64) * (i as f64),
                    )
                    .exp();
            } else {
                acc += input[i * stride + offset]
                    * Complex::new(
                        0.0,
                        -2.0 * std::f64::consts::PI / (n as f64) * (freq as f64) * (i as f64),
                    )
                    .exp();
            }
        }
        ret.push(acc);
    }
    ret
}

// Implementation of Cooley-Tukey FFT
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
fn fft_helper(
    input: &[Complex<f64>],
    n: usize,
    stride: usize,
    offset: usize,
    inverse: bool,
) -> Vec<Complex<f64>> {
    if n == 1 {
        vec![input[offset]]
    } else if n % 2 != 0 {
        slow_dft(input, n, stride, offset, inverse)
    } else {
        let even = fft_helper(input, n / 2, stride * 2, offset, inverse);
        let odd = fft_helper(input, n / 2, stride * 2, offset + stride, inverse);
        let mut ret = vec![Complex::new(0.0, 0.0); n];
        for i in 0..(n / 2) {
            let (p, q) = if inverse {
                (
                    0.5 * even[i],
                    0.5 * odd[i]
                        * Complex::new(0.0, 2.0 * std::f64::consts::PI / (n as f64) * (i as f64))
                            .exp(),
                )
            } else {
                (
                    even[i],
                    odd[i]
                        * Complex::new(0.0, -2.0 * std::f64::consts::PI / (n as f64) * (i as f64))
                            .exp(),
                )
            };
            ret[i] = p + q;
            ret[i + n / 2] = p - q;
        }
        ret
    }
}

pub fn fft(
    input: &mut [Complex<f64>],
    n: usize,
    sampling_interval: f64,
    stride: usize,
    offset: usize,
    shift: usize,
    inverse: bool,
) {
    let ret = slow_dft(input, n, stride, offset, inverse);
    let sampling_interval = if inverse {
        1.0 / sampling_interval
    } else {
        sampling_interval
    };
    for i in 0..shift {
        input[i * stride + offset] = ret[i + n - shift] * sampling_interval;
    }
    for i in shift..n {
        input[i * stride + offset] = ret[i - shift] * sampling_interval;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_arrays(input1: &[Complex<f64>], input2: &[Complex<f64>], tolerance: f64) -> bool {
        if input1.len() != input2.len() {
            false
        } else {
            for i in 0..input1.len() {
                if (input1[i] - input2[i]).re.abs() >= tolerance
                    || (input1[i] - input2[i]).im.abs() >= tolerance
                {
                    return false;
                }
            }
            true
        }
    }

    #[test]
    fn test_factorial() {
        assert!(factorial(1) == 1, "Expected {} Actual {}", 1, factorial(1));
        assert!(factorial(2) == 2, "Expected {} Actual {}", 2, factorial(2));
        assert!(factorial(3) == 6, "Expected {} Actual {}", 3, factorial(3));
    }

    #[test]
    fn test_fft() {
        // Dirac delta has a power spectrum of all 1s.
        let mut test_input = vec![Complex::new(0.0, 0.0); 16];
        test_input[0] = Complex::new(1.0, 0.0);
        let mut actual_output = slow_dft(test_input.as_slice(), test_input.len(), 1, 0, false);
        let mut expected_output = vec![Complex::new(1.0, 0.0); 16];
        assert!(
            compare_arrays(actual_output.as_slice(), expected_output.as_slice(), 0.01),
            "Slow DFT error!\nExpected {:?}\nActual {:?}",
            expected_output,
            actual_output
        );
        actual_output = test_input.clone();
        fft(actual_output.as_mut(), 16, 1.0, 1, 0, 0, false);
        assert!(
            compare_arrays(actual_output.as_slice(), expected_output.as_slice(), 0.01),
            "FFT error!\nExpected {:?}\nActual {:?}",
            expected_output,
            actual_output
        );
        actual_output = slow_dft(expected_output.as_slice(), test_input.len(), 1, 0, true);
        assert!(
            compare_arrays(actual_output.as_slice(), test_input.as_slice(), 0.01),
            "Slow IDFT error!\nExpected {:?}\nActual {:?}",
            test_input,
            actual_output
        );
        actual_output = expected_output.clone();
        fft(actual_output.as_mut(), 16, 1.0, 1, 0, 0, true);
        assert!(
            compare_arrays(actual_output.as_slice(), test_input.as_slice(), 0.01),
            "IFFT error!\nExpected {:?}\nActual {:?}",
            test_input,
            actual_output
        );

        // DC test case
        test_input = vec![Complex::new(1.0, 0.0); 16];
        actual_output = slow_dft(test_input.as_slice(), test_input.len(), 1, 0, false);
        expected_output = vec![Complex::new(0.0, 0.0); 16];
        expected_output[0] = Complex::new(16.0, 0.0);
        assert!(
            compare_arrays(actual_output.as_slice(), expected_output.as_slice(), 0.01),
            "Slow DFT error!\nExpected {:?}\nActual {:?}",
            expected_output,
            actual_output
        );
        actual_output = test_input.clone();
        fft(actual_output.as_mut(), 16, 1.0, 1, 0, 0, false);
        assert!(
            compare_arrays(actual_output.as_slice(), expected_output.as_slice(), 0.01),
            "FFT error!\nExpected {:?}\nActual {:?}",
            expected_output,
            actual_output
        );
        actual_output = slow_dft(expected_output.as_slice(), test_input.len(), 1, 0, true);
        assert!(
            compare_arrays(actual_output.as_slice(), test_input.as_slice(), 0.01),
            "Slow IDFT error!\nExpected {:?}\nActual {:?}",
            test_input,
            actual_output
        );
        actual_output = expected_output.clone();
        fft(actual_output.as_mut(), 16, 1.0, 1, 0, 0, true);
        assert!(
            compare_arrays(actual_output.as_slice(), test_input.as_slice(), 0.01),
            "IFFT error!\nExpected {:?}\nActual {:?}",
            test_input,
            actual_output
        );

        // 1/2 of the highest frequency cosine we can sample.
        for i in 0..test_input.len() {
            test_input[i] = Complex::new((((i as isize % 2) * -2) + 1) as f64, 0.0);
        }
        actual_output = slow_dft(test_input.as_slice(), test_input.len(), 1, 0, false);
        expected_output = vec![Complex::new(0.0, 0.0); 16];
        expected_output[8] = Complex::new(16.0, 0.0);
        assert!(
            compare_arrays(actual_output.as_slice(), expected_output.as_slice(), 0.01),
            "Slow DFT error!\nExpected {:?}\nActual {:?}",
            expected_output,
            actual_output
        );
        actual_output = test_input.clone();
        fft(actual_output.as_mut(), 16, 1.0, 1, 0, 0, false);
        assert!(
            compare_arrays(actual_output.as_slice(), expected_output.as_slice(), 0.01),
            "FFT error!\nExpected {:?}\nActual {:?}",
            expected_output,
            actual_output
        );
        actual_output = slow_dft(expected_output.as_slice(), test_input.len(), 1, 0, true);
        assert!(
            compare_arrays(actual_output.as_slice(), test_input.as_slice(), 0.01),
            "Slow IDFT error!\nExpected {:?}\nActual {:?}",
            test_input,
            actual_output
        );
        actual_output = expected_output.clone();
        fft(actual_output.as_mut(), 16, 1.0, 1, 0, 0, true);
        assert!(
            compare_arrays(actual_output.as_slice(), test_input.as_slice(), 0.01),
            "IFFT error!\nExpected {:?}\nActual {:?}",
            test_input,
            actual_output
        );
    }

    #[test]
    fn test_conv_fft() {
        let size = 16;
        let mut test_input = vec![Complex::new(0.0, 0.0); size];
        test_input[size / 2] = Complex::new(1.0, 0.0);
        let mut conv = vec![Complex::new(0.0, 0.0); size];
        for x1 in 0..size {
            let mut acc: Complex<f64> = Complex::new(0.0, 0.0);
            for x2 in 0..size {
                acc += test_input[x2] / Complex::new((x1 as f64 - x2 as f64).abs().max(0.01), 0.0);
            }
            conv[x1] = acc;
        }

        let mut fft_conv = vec![Complex::new(0.0, 0.0); size];
        for x in 0..size {
            let dx = x as f64 - ((size / 2) as f64 - 1.0);
            fft_conv[x] = (1.0 / (dx * dx).sqrt().max(0.01)).into();
        }
        fft(test_input.as_mut(), size, 1.0, 1, 0, 0, false);
        fft(fft_conv.as_mut(), size, 1.0, 1, 0, 0, false);
        for x in 0..size {
            fft_conv[x] = fft_conv[x] * test_input[x];
        }
        fft(
            fft_conv.as_mut(),
            size,
            1.0,
            1,
            0,
            size - (size / 2 - 1),
            true,
        );

        assert!(
            compare_arrays(conv.as_slice(), fft_conv.as_slice(), 0.01),
            "FFT Conv error!\nExpected {:?}\nActual {:?}",
            conv,
            fft_conv
        );

        let mut test_input = vec![Complex::new(0.0, 0.0); size * size];
        test_input[size * size / 2 + size / 2] = Complex::new(1.0, 0.0);
        let mut conv = vec![Complex::new(0.0, 0.0); size * size];
        for x1 in 0..size {
            for y1 in 0..size {
                let mut acc: Complex<f64> = Complex::new(0.0, 0.0);
                for x2 in 0..size {
                    for y2 in 0..size {
                        let dx = x1 as f64 - x2 as f64;
                        let dy = y1 as f64 - y2 as f64;
                        acc += test_input[y2 * size + x2]
                            / Complex::new((dx * dx + dy * dy).sqrt().max(0.01), 0.0);
                    }
                }
                conv[y1 * size + x1] = acc;
            }
        }
        let mut fft_conv = vec![Complex::new(0.0, 0.0); size * size];
        for x in 0..size {
            for y in 0..size {
                let dx = x as f64 - ((size / 2) as f64 - 1.0);
                let dy = y as f64 - ((size / 2) as f64 - 1.0);
                fft_conv[y * size + x] =
                    Complex::new(1.0 / (dx * dx + dy * dy).sqrt().max(0.01), 0.0);
            }
        }
        let fft2d = |data: &mut [Complex<f64>], shift, inverse| -> () {
            for y in 0..size {
                fft(data, size, 1.0, 1, y * size, shift, inverse);
            }
            for x in 0..size {
                fft(data, size, 1.0, size, x, shift, inverse);
            }
        };
        fft2d(test_input.as_mut(), 0, false);
        fft2d(fft_conv.as_mut(), 0, false);
        for i in 0..fft_conv.len() {
            fft_conv[i] = fft_conv[i] * test_input[i];
        }
        fft2d(fft_conv.as_mut(), size - (size / 2 - 1), true);
        assert!(
            compare_arrays(conv.as_slice(), fft_conv.as_slice(), 0.01),
            "FFT Conv error!\nExpected {:?}\nActual {:?}",
            conv,
            fft_conv
        );
    }
}
