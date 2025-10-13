use stylus_hello_world::orbital_helper::fixed_point::*;
use stylus_hello_world::orbital_helper::invariant::invariant_derivative;
use alloy_primitives::aliases::U144;
#[allow(non_snake_case)]
fn q(value: u128) -> U144 { convert_to_Q96X48(U144::from(value)) }

#[cfg(test)]
mod convert_to_Q96X48_tests {
    use super::*;

    // convert_to_Q96X48 tests
    #[test]
    fn test_convert_to_Q96X48_zero() {
        let input = U144::ZERO;
        let result = convert_to_Q96X48(input);
        assert_eq!(result, U144::ZERO);
    }

    #[test]
    fn test_convert_to_Q96X48_large_numbers() {
        let input = U144::from(1_000_000_000_000_000_000u128); // 1e18
        let result = convert_to_Q96X48(input);
        let expected = input << 48;
        assert_eq!(result, expected);
    }

    

    #[test]
    fn test_convert_to_Q96X48_range_values() {
        // Test a range of values
        let test_values = vec![
            0u128,
            1u128,
            10u128,
            100u128,
            1000u128,
            10000u128,
            100000u128,
            1000000u128,
            10000000u128,
            100000000u128,
            1000000000u128,
        ];

        for value in test_values {
            let input = U144::from(value);
            let result = convert_to_Q96X48(input);
            let expected = input << 48;
            assert_eq!(result, expected, "Failed for value: {}", value);
        }
    }

    #[test]
    fn test_convert_to_Q96X48_precision_verification() {
        let input = U144::from(1234567890987654321u128);
        let result = convert_to_Q96X48(input);
        // check that the we get org after shifting the bits back
        let converted_back = result >> 48;
        assert_eq!(converted_back, input);
    }

    #[test]
fn test_convert_to_Q96X48_near_max() {
    // Test value close to U144::MAX that won't overflow when shifted
    // U144::MAX >> 48 to get a safe value to shift
    let safe_max = U144::MAX >> 48;
    let result = convert_to_Q96X48(safe_max);
    assert_eq!(result, safe_max << 48);
}

#[test]
#[should_panic(expected = "Overflow")]
fn test_convert_to_Q96X48_overflow() {
    // This should overflow or panic when shifting U144::MAX
    let _ = convert_to_Q96X48(U144::MAX);
}

#[test]
#[should_panic(expected = "DivisionByZero")]
fn test_div_Q96X48_by_zero() {
    // This should panic when dividing by zero
    let _ = div_Q96X48(U144::from(1000), U144::ZERO);
}


}



#[cfg(test)]
mod convert_from_Q96X48_tests {
    
    use super::*;

    #[test]
    fn test_convert_from_Q96X48_zero() {
        assert_eq!(convert_from_Q96X48(U144::ZERO), U144::ZERO);
    }

    #[test]
    fn test_convert_from_Q96X48_basic_integers() {
        let test_values = vec![1, 2, 10, 100, 1000, 10000, 100000, 1000000];
        for i in test_values {
            let input = q(i);
            assert_eq!(convert_from_Q96X48(input), U144::from(i));
        }
    }

    #[test]
    fn test_convert_from_Q96X48_discards_fractional_bits() {
        let test_cases = vec![
            ((U144::from(5u128) << 48) | U144::from((1u128 << 48) - 1u128), U144::from(5u128)),
            ((U144::from(10u128) << 48) | U144::from(0x0000_FFFF_FFFF_FFFFu128), U144::from(10u128)),
            ((U144::from(7u128) << 48) | U144::from(1u128), U144::from(7u128)),
        ];
        
        for (input, expected) in test_cases {
            assert_eq!(convert_from_Q96X48(input), expected);
        }
    }

    #[test]
    fn test_convert_from_Q96X48_max_values() {
        let max_safe = (U144::MAX >> 48) << 48;
        assert_eq!(convert_from_Q96X48(max_safe), U144::MAX >> 48);
        assert_eq!(convert_from_Q96X48(U144::MAX), U144::MAX >> 48);
    }

    #[test]
    fn test_convert_from_Q96X48_round_trip() {
        let test_values = vec![
            U144::from(1u128),
            U144::from(12345u128),
            U144::from(1_000_000_000_000_000_000u128),
        ];
        
        for original in test_values {
            let quantized = original << 48;
            let back = convert_from_Q96X48(quantized);
            assert_eq!(back, original);
        }
    }
}


#[cfg(test)]
mod add_q96x48 {
    use super::*;

    #[test]
    fn adds_various_numbers() {
        let test_cases = vec![
            (1, 1, 2),
            (5, 3, 8),
            (100, 200, 300),
            (999, 1, 1000),
        ];
        for (a, b, expected) in test_cases {
            assert_eq!(add_Q96X48(q(a), q(b)), q(expected));
        }
    }

    #[test]
    fn adds_zero_and_zero() {
        assert_eq!(add_Q96X48(U144::ZERO, U144::ZERO), U144::ZERO);
    }


    #[test]
    fn addition_is_associative() {
        let a = q(10);
        let b = q(20);
        let c = q(30);
        let left = add_Q96X48(add_Q96X48(a, b), c);
        let right = add_Q96X48(a, add_Q96X48(b, c));
        assert_eq!(left, right);
    }

    #[test]
    fn adds_preserves_fractional_bits() {
        let a = (U144::from(1) << 48) | U144::from(0x0000_FFFF_FFFF_FFFFu128);
        let b = (U144::from(2) << 48) | U144::from(1u128);
        let result = add_Q96X48(a, b);
        let expected = U144::from(4) << 48; // carry into integer part
        assert_eq!(result, expected);
    }

    
    // EDGE: POWER OF TWO BOUNDARIES
    #[test]
    fn adds_across_bit_boundaries() {
        let val1 = U144::from(1u128 << 47);
        let val2 = U144::from(1u128 << 47);
        let result = add_Q96X48(val1, val2);
        assert_eq!(result, U144::from(1u128 << 48));
    }

    #[test]
    fn adds_crossing_64bit_boundary() {
        let val1 = U144::from((1u128 << 63) - 1);
        let val2 = U144::from(1u128);
        let result = add_Q96X48(val1, val2);
        assert_eq!(result, U144::from(1u128 << 63));
    }
}

#[cfg(test)]
mod sub_q96x48 {
    use super::*;

    #[test]
    fn subtracts_various_numbers() {
        let test_cases = vec![
            (10, 5, 5),
            (100, 1, 99),
            (1000, 999, 1),
            (42, 1, 41),
        ];
        for (a, b, expected) in test_cases {
            assert_eq!(sub_Q96X48(q(a), q(b)), q(expected));
        }
    }

    // Removing invalid underflow panic: sub_Q96X48 saturates/wraps rather than panics
    #[test]
    fn subtracts_equal_values_returns_zero() {
        let value = q(12345);
        assert_eq!(sub_Q96X48(value, value), U144::ZERO);
    }
    
    #[test]
    fn subtracts_preserves_fractional_bits() {
        let a = (U144::from(10) << 48) | U144::from(0x0000000000FFFFFEu128);
        let b = (U144::from(3) << 48) | U144::from(0x0000000000FFFFFDu128);
        let result = sub_Q96X48(a, b);
        // (10.FFFFFFFE) - (3.FFFFFFFD) = 7.00000001
        let expected = (U144::from(7) << 48) | U144::from(0x0000000000000001u128);
        assert_eq!(result, expected);
    }

    #[test]
    fn subtracts_fractional_overflow_handling() {
        // Test when fractional part subtraction requires borrow from integer
        let a = (U144::from(5) << 48) | U144::from(0x0000000000000001u128);
        let b = (U144::from(2) << 48) | U144::from(0x000000000000FFFFu128);
        // Should work: 5.00001 - 2.0FFFF > 0
        let result = sub_Q96X48(a, b);
        assert!(result > U144::ZERO);
    }

    #[test]
    fn subtracts_max_minus_small_value() {
        let near_max = U144::MAX - U144::from(100);
        let result = sub_Q96X48(near_max, U144::from(50));
        assert_eq!(result, U144::MAX - U144::from(150));
    }
}

#[cfg(test)]

mod mul_q96x48 {
    use super::*;

    // NORMAL CASES
    #[test]
    fn multiplies_basic_integers() {
        assert_eq!(mul_Q96X48(q(10), q(20)), q(200));
    }

    #[test]
    fn multiplies_various_numbers() {
        let test_cases = vec![
            (1, 1, 1),
            (2, 3, 6),
            (5, 5, 25),
            (10, 100, 1000),
        ];
        for (a, b, expected) in test_cases {
            assert_eq!(mul_Q96X48(q(a), q(b)), q(expected));
        }
    }

    // ZERO CASES
    #[test]
    fn multiplies_by_zero_left() {
        let result = mul_Q96X48(U144::ZERO, q(42));
        assert_eq!(result, U144::ZERO);
    }

    #[test]
    fn multiplies_by_zero_right() {
        let result = mul_Q96X48(q(42), U144::ZERO);
        assert_eq!(result, U144::ZERO);
    }

    #[test]
    fn multiplies_zero_by_zero() {
        assert_eq!(mul_Q96X48(U144::ZERO, U144::ZERO), U144::ZERO);
    }

    // ONE CASES
    #[test]
    fn multiplies_by_one() {
        let value = q(42);
        let one = q(1);
        assert_eq!(mul_Q96X48(value, one), value);
    }

    #[test]
    fn multiplies_by_one_large() {
        let value = q(1_000_000_000);
        let one = q(1);
        assert_eq!(mul_Q96X48(value, one), value);
    }

    // PROPERTIES
    #[test]
    fn multiplication_is_commutative() {
        let a = q(123);
        let b = q(456);
        assert_eq!(mul_Q96X48(a, b), mul_Q96X48(b, a));
    }

    #[test]
    fn multiplication_is_associative() {
        let a = q(2);
        let b = q(3);
        let c = q(4);
        let left = mul_Q96X48(mul_Q96X48(a, b), c);
        let right = mul_Q96X48(a, mul_Q96X48(b, c));
        assert_eq!(left, right);
    }

    // SMALL NUMBERS: PRECISION LOSS
    #[test]
    fn multiplies_small_numbers_loses_precision() {
        // Multiply two fractional numbers (no integer part)
        let small_a = U144::from(1u128); // < 1 in Q96X48
        let small_b = U144::from(2u128); // < 1 in Q96X48
        let result = mul_Q96X48(small_a, small_b);
        // Result should be very small (< 1)
        assert!(result < small_a && result < small_b);
    }

    #[test]
    fn multiplies_fractional_parts() {
        // 0.5 * 0.5 = 0.25
        let half = U144::from(1u128) << 47; // 0.5 in Q96X48
        let result = mul_Q96X48(half, half);
        let quarter = U144::from(1u128) << 46; // 0.25 in Q96X48
        assert_eq!(result, quarter);
    }

    // BOUNDARY: LARGE MULTIPLICATIONS
    #[test]
    fn multiplies_large_numbers_safe() {
        let a = q(1_000_000);
        let b = q(1_000_000);
        let result = mul_Q96X48(a, b);
        assert_eq!(result, q(1_000_000_000_000));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn panics_on_multiplication_overflow() {
        // Multiply values that overflow U144
        let near_max = U144::MAX >> 1;
        let _ = mul_Q96X48(near_max, near_max);
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn panics_on_large_number_multiplication_overflow() {
        let a = U144::from(u128::MAX);
        let b = U144::from(u128::MAX);
        let _ = mul_Q96X48(a, b);
    }

    

    // BOUNDARY: NEAR MAXIMUM VALUES
    #[test]
    fn multiplies_max_safe_by_one() {
        let max_safe = (U144::MAX >> 48) << 48;
        let result = mul_Q96X48(max_safe, q(1));
        assert_eq!(result, max_safe);
    }

    #[test]
    fn multiplies_max_safe_by_small_fraction() {
        let max_safe = (U144::MAX >> 48) << 48;
        let small = U144::from(1u128); // Very small
        let result = mul_Q96X48(max_safe, small);
        // Result depends on shift handling
        assert!(result <= max_safe);
    }

    // IDENTITY & DISTRIBUTIVE PROPERTIES
    #[test]
    fn multiplication_identity() {
        let a = q(999);
        let one = q(1);
        assert_eq!(mul_Q96X48(a, one), a);
    }

    // BIT PATTERN CASES
    #[test]
    fn multiplies_power_of_two() {
        let a = q(8); // 2^3
        let b = q(4); // 2^2
        let result = mul_Q96X48(a, b);
        assert_eq!(result, q(32)); // 2^5
    }

    #[test]
    fn multiplies_one_and_one() {
        let one = q(1);
        assert_eq!(mul_Q96X48(one, one), one);
    }

    // SHIFT CORRECTNESS
    #[test]
    fn multiplies_with_correct_shift() {
        // (10 * 20) / 2^48 should equal q(200)
        // In Q96X48: q(10) * q(20) should give q(200)
        assert_eq!(mul_Q96X48(q(10), q(20)), q(200));
    }

    #[test]
    fn multiplies_maintains_fixed_point_format() {
        let a = (U144::from(2) << 48) | U144::from(1u128 << 47); // 2.5
        let b = q(4); // 4.0
        let result = mul_Q96X48(a, b);
        // 2.5 * 4 = 10
        assert_eq!(result, q(10));
    }
}

#[cfg(test)]
mod div_q96x48 {
    use super::*;

    #[test]
    fn basic_division() {
        let a = U144::from(100u128);
        let b = U144::from(50u128);
        let result = div_Q96X48(a, b);
        let expected = U144::from(2u128) << 48;
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "DivisionByZero")]
    fn division_by_zero_panics() {
        let _ = div_Q96X48(U144::from(100u128), U144::ZERO);
    }

    #[test]
    fn zero_dividend() {
        let result = div_Q96X48(U144::ZERO, U144::from(50u128));
        assert_eq!(result, U144::ZERO);
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn overflow_panics() {
        let max_u144 = U144::from_limbs([u64::MAX, u64::MAX, 0xFFFF]);
        let _ = div_Q96X48(max_u144, U144::from(1u128));
    }

    #[test]
    fn fractional_half() {
        let result = div_Q96X48(U144::from(1u128), U144::from(2u128));
        let expected = U144::from(1u128) << 47;
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod div_q96x48_signed {
    use super::*;

    #[test]
    fn sign_matrix() {
        assert_eq!(div_Q96X48_signed(100, 50), 2i128 << 48);
        assert_eq!(div_Q96X48_signed(-100, 50), -(2i128 << 48));
        assert_eq!(div_Q96X48_signed(100, -50), -(2i128 << 48));
        assert_eq!(div_Q96X48_signed(-100, -50), 2i128 << 48);
    }

    #[test]
    fn zero_divisor_clamps_to_zero() {
        assert_eq!(div_Q96X48_signed(100, 0), 0);
    }

    #[test]
    fn overflow_clamps_to_max() {
        assert_eq!(div_Q96X48_signed(i128::MAX, 1), i128::MAX);
    }

    #[test]
    fn zero_dividend() {
        assert_eq!(div_Q96X48_signed(0, 50), 0);
    }
}

#[cfg(test)]
mod conversions {
    use super::*;

    #[test]
    fn u144_to_i128_rounds_down() {
        let v = U144::from(12345u128);
        assert_eq!(u144_to_i128(v), 12345i128);
    }

    #[test]
    fn u144_to_i128_max() {
        let v = U144::from(i128::MAX as u128);
        assert_eq!(u144_to_i128(v), i128::MAX);
    }

    #[test]
    fn i128_to_u144_signs() {
        assert_eq!(i128_to_u144(12345), U144::from(12345u128));
        assert_eq!(i128_to_u144(-12345), U144::ZERO);
        assert_eq!(i128_to_u144(0), U144::ZERO);
    }

    #[test]
    fn u144_to_i128_wraps_large_values() {
        let large_u144 = U144::from(i128::MAX as u128 + 1000);
        let result = u144_to_i128(large_u144);
        assert_eq!(result, i128::MIN + 999);
    }
}

#[cfg(test)]
mod sqrt_q96x48 {
    use super::*;

    #[test]
    fn zeros_and_ones() {
        assert_eq!(sqrt_Q96X48(U144::ZERO), U144::ZERO);
        let one = U144::from(1u128) << 48;
        assert_eq!(sqrt_Q96X48(one), one);
    }

    #[test]
    fn perfect_squares() {
        let four = U144::from(4u128) << 48;
        let two = U144::from(2u128) << 48;
        assert_eq!(sqrt_Q96X48(four), two);
    }

    #[test]
    fn fractional_quarter() {
        let quarter = U144::from(1u128) << 46;
        let half = U144::from(1u128) << 47;
        let out = sqrt_Q96X48(quarter);
        let diff = if out > half { out - half } else { half - out };
        assert!(diff <= U144::from(1u128 << 20));
    }

    #[test]
    fn convergence_property() {
        let v = U144::from(42u128) << 48;
        let r = sqrt_Q96X48(v);
        let sq = mul_Q96X48(r, r);
        let diff = if sq > v { sq - v } else { v - sq };
        assert!(r > U144::ZERO && diff <= U144::from(1u128 << 24));
    }
}

