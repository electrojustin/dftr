pub fn factorial(n: i32) -> i32 {
    let mut ret: i32 = 1;
    for i in 2..(n + 1) {
        ret *= i;
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert!(factorial(1) == 1, "Expected {} Actual {}", 1, factorial(1));
        assert!(factorial(2) == 2, "Expected {} Actual {}", 2, factorial(2));
        assert!(factorial(3) == 6, "Expected {} Actual {}", 3, factorial(3));
    }
}
