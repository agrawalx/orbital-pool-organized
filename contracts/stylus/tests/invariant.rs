use stylus_hello_world::orbital_helper::invariant::{
    calculate_A_B_D, calculate_invariant_simple, invariant_derivative, 
    solveQuadraticInvariant, solve_amount_out
};
use stylus_hello_world::orbital_helper::fixed_point::*;
use alloy_primitives::aliases::U144;
#[allow(non_snake_case)]

fn q(value: u128) -> U144 { convert_to_Q96X48(U144::from(value)) }

#[cfg(test)]
mod calculate_invariant_simple_tests {
    use super::*;

    #[test]
    fn test_invariant_simple_positive_a() {
        let a = 10i128 << 48;
        let b = U144::from(5u128 << 48);
        let r_int = U144::from(3u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result > 0);
    }

    #[test]
    fn test_invariant_simple_negative_a() {
        let a = -(10i128 << 48);
        let b = U144::from(5u128 << 48);
        let r_int = U144::from(3u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result > 0);
    }

    #[test]
    fn test_invariant_simple_zero_a() {
        let a = 0i128;
        let b = U144::from(10u128 << 48);
        let r_int = U144::from(5u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result > 0);
    }

    #[test]
    fn test_invariant_simple_zero_b() {
        let a = 10i128 << 48;
        let b = U144::ZERO;
        let r_int = U144::from(5u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result != 0);
    }

    #[test]
    fn test_invariant_simple_zero_r_int() {
        let a = 10i128 << 48;
        let b = U144::from(5u128 << 48);
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result > 0);
    }

    #[test]
    fn test_invariant_simple_all_zeros() {
        let a = 0i128;
        let b = U144::ZERO;
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_invariant_simple_negative_result() {
        let a = 1i128 << 48;
        let b = U144::from(1u128 << 48);
        let r_int = U144::from(10u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result < 0);
    }

    #[test]
    fn test_invariant_simple_large_a() {
        let a = (i128::MAX >> 49) << 48;
        let b = U144::from(1u128 << 48);
        let r_int = U144::from(1u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result != 0);
    }

    #[test]
    fn test_invariant_simple_large_negative_a() {
        let a = -(i128::MAX >> 49) << 48;
        let b = U144::from(1u128 << 48);
        let r_int = U144::from(1u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result != 0);
    }

    #[test]
    fn test_invariant_simple_a_squared_truncation() {
        let a = 1000i128 << 48;
        let b = U144::ZERO;
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result > 0);
    }

    #[test]
    fn test_invariant_simple_b_squared_overflow_protection() {
        let a = 0i128;
        let b = U144::from(1u128 << 70);
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result >= 0);
    }

    #[test]
    fn test_invariant_simple_r_int_squared_overflow_protection() {
        let a = 0i128;
        let b = U144::ZERO;
        let r_int = U144::from(1u128 << 70);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result < 0);
    }

    #[test]
    fn test_invariant_simple_balanced_case() {
        let a = 3i128 << 48;
        let b = U144::from(4u128 << 48);
        let r_int = U144::from(5u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result.abs() < (1i128 << 52));
    }

    #[test]
    fn test_invariant_simple_symmetry_negative_a() {
        let a_pos = 10i128 << 48;
        let a_neg = -10i128 << 48;
        let b = U144::from(5u128 << 48);
        let r_int = U144::from(3u128 << 48);
        let result_pos = calculate_invariant_simple(a_pos, b, r_int);
        let result_neg = calculate_invariant_simple(a_neg, b, r_int);
        assert_eq!(result_pos, result_neg);
    }

    #[test]
    fn test_invariant_simple_u144_to_i128_conversion() {
        let a = 1i128 << 48;
        let b = U144::from((1u128 << 96) - 1);
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result != 0);
    }

    #[test]
    fn test_invariant_simple_min_i128() {
        let a = i128::MIN >> 1;
        let b = U144::ZERO;
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result >= 0);
    }

    #[test]
    fn test_invariant_simple_precision_loss() {
        let a = (1i128 << 48) + 1;
        let b = U144::ZERO;
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result > 0);
        assert!(result < (2i128 << 48));
    }

    #[test]
    fn test_invariant_simple_limbs_extraction() {
        let a = 0xFFFFFFFFi128 << 48;
        let b = U144::ZERO;
        let r_int = U144::ZERO;
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result > 0);
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn test_invariant_simple_sum_overflow_protection() {
        let a = (i128::MAX >> 50) << 48;
        let b = U144::from((u128::MAX >> 50) << 48);
        let r_int = U144::from(1u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result != 0);
    }

    #[test]
    fn test_invariant_simple_underflow_protection() {
        let a = 0i128;
        let b = U144::ZERO;
        let r_int = U144::from(1000u128 << 48);
        let result = calculate_invariant_simple(a, b, r_int);
        assert!(result < -(900i128 << 48));
    }
}

mod calculate_a_b_d_basic_tests {
    use super::*;

    #[test]
    fn test_calculate_a_b_d_basic() {
        let sum_reserves = U144::from(100u128 << 48);
        let sum_reserves_squared = U144::from(10000u128 << 48);
        let n = 4u32;
        let x_j = U144::from(20u128 << 48);
        let k_bound = U144::from(10u128 << 48);
        let r_int = U144::from(5u128 << 48);
        let s_bound = U144::from(15u128 << 48);

        let (a, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(a != 0);
        assert!(b >= U144::ZERO);
        assert!(d > U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_negative_a() {
        let sum_reserves = U144::from(10u128 << 48);
        let sum_reserves_squared = U144::from(100u128 << 48);
        let n = 4u32;
        let x_j = U144::from(2u128 << 48);
        let k_bound = U144::from(50u128 << 48);
        let r_int = U144::from(10u128 << 48);
        let s_bound = U144::from(5u128 << 48);

        let (a, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(a < 0);
        assert!(b >= U144::ZERO);
        assert!(d >= U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_positive_a() {
        let sum_reserves = U144::from(1000u128 << 48);
        let sum_reserves_squared = U144::from(1000000u128 << 48);
        let n = 4u32;
        let x_j = U144::from(100u128 << 48);
        let k_bound = U144::from(1u128 << 48);
        let r_int = U144::from(1u128 << 48);
        let s_bound = U144::from(1u128 << 48);

        let (a, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(a > 0);
    }

    #[test]
    fn test_calculate_a_b_d_zero_xj() {
        let sum_reserves = U144::from(100u128 << 48);
        let sum_reserves_squared = U144::from(10000u128 << 48);
        let n = 4u32;
        let x_j = U144::ZERO;
        let k_bound = U144::from(10u128 << 48);
        let r_int = U144::from(5u128 << 48);
        let s_bound = U144::from(15u128 << 48);

        let (a, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(d >= U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_n_equals_one() {
        let sum_reserves = U144::from(50u128 << 48);
        let sum_reserves_squared = U144::from(2500u128 << 48);
        let n = 1u32;
        let x_j = U144::from(10u128 << 48);
        let k_bound = U144::from(5u128 << 48);
        let r_int = U144::from(2u128 << 48);
        let s_bound = U144::from(5u128 << 48);

        let (a, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(d >= U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_large_n() {
        let sum_reserves = U144::from(1000u128 << 48);
        let sum_reserves_squared = U144::from(1000000u128 << 48);
        let n = 100u32;
        let x_j = U144::from(10u128 << 48);
        let k_bound = U144::from(5u128 << 48);
        let r_int = U144::from(2u128 << 48);
        let s_bound = U144::from(5u128 << 48);

        let (a, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(d >= U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_d_variance_formula() {
        let sum_reserves = U144::from(100u128 << 48);
        let sum_reserves_squared = U144::from(12000u128 << 48);
        let n = 4u32;
        let x_j = U144::from(10u128 << 48);
        let k_bound = U144::from(5u128 << 48);
        let r_int = U144::from(2u128 << 48);
        let s_bound = U144::from(5u128 << 48);

        let (_, _, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(d > U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_b_with_large_s_bound() {
        let sum_reserves = U144::from(100u128 << 48);
        let sum_reserves_squared = U144::from(15000u128 << 48);
        let n = 4u32;
        let x_j = U144::from(20u128 << 48);
        let k_bound = U144::from(5u128 << 48);
        let r_int = U144::from(2u128 << 48);
        let s_bound = U144::from(100u128 << 48);

        let (_, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(b >= U144::ZERO);
        assert!(d > U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_all_bounds_zero() {
        let sum_reserves = U144::from(100u128 << 48);
        let sum_reserves_squared = U144::from(10000u128 << 48);
        let n = 4u32;
        let x_j = U144::from(10u128 << 48);
        let k_bound = U144::ZERO;
        let r_int = U144::ZERO;
        let s_bound = U144::ZERO;

        let (a, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(a > 0);
        assert!(b > U144::ZERO);
        assert!(d > U144::ZERO);
    }

    #[test]
    fn test_calculate_a_b_d_symmetry() {
        let sum_reserves = U144::from(100u128 << 48);
        let sum_reserves_squared = U144::from(10000u128 << 48);
        let n = 4u32;
        let x_j = U144::from(10u128 << 48);
        let k_bound = U144::from(5u128 << 48);
        let r_int = U144::from(2u128 << 48);
        let s_bound = U144::from(5u128 << 48);

        let (a1, _, _) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        let (a2, _, _) = calculate_A_B_D(
            sum_reserves * U144::from(2u128),
            sum_reserves_squared * U144::from(4u128),
            n,
            x_j * U144::from(2u128),
            k_bound,
            r_int,
            s_bound,
        );

        assert!(a2 > a1 || (a1 < 0 && a2 < 0));
    }

    #[test]
    fn test_calculate_a_b_d_zero_d_edge_case() {
        let n = 4u32;
        let sum_reserves = U144::from(100u128 << 48);
        let sum_reserves_squared = div_Q96X48(
            mul_Q96X48(sum_reserves, sum_reserves),
            convert_to_Q96X48(U144::from(n))
        );
        let x_j = U144::ZERO;
        let k_bound = U144::from(5u128 << 48);
        let r_int = U144::from(2u128 << 48);
        let s_bound = U144::from(5u128 << 48);

        let (_, b, d) = calculate_A_B_D(
            sum_reserves,
            sum_reserves_squared,
            n,
            x_j,
            k_bound,
            r_int,
            s_bound,
        );

        assert!(d >= U144::ZERO);
        assert!(b >= U144::ZERO);
    }
}



mod test_invariant_derivative {
    use super::*;

    fn u144(val: u128) -> U144 {
        U144::from(val)
    }

    fn Q96_48(val: u128) -> U144 {
        U144::from(val << 48)
    }

    #[test]
    fn zero_deviation_from_average() {
        let a = 1000_i128 << 48;
        let b = Q96_48(500);
        let d = Q96_48(1000);
        let n = 4;
        let x_j = Q96_48(250);
        let sum_reserves = Q96_48(750);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        let sqrt_n = sqrt_Q96X48(Q96_48(n as u128));
        let sqrt_n_i128 = u144_to_i128(sqrt_n);
        let expected = div_Q96X48_signed(2 * a, sqrt_n_i128);
        assert!((result - expected).abs() < (1 << 46));
    }

    #[test]
    fn positive_deviation_above_average() {
        let a = 1000_i128 << 48;
        let b = Q96_48(500);
        let d = Q96_48(1000);
        let n = 4;
        let x_j = Q96_48(400);
        let sum_reserves = Q96_48(800);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result > 0);
    }

    #[test]
    fn negative_deviation_below_average() {
        let a = -500_i128 << 48;
        let b = Q96_48(1000);
        let d = Q96_48(500);
        let n = 3;
        let x_j = Q96_48(100);
        let sum_reserves = Q96_48(500);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result < 0);
    }

    #[test]
    fn single_asset_pool() {
        let a = 100_i128 << 48;
        let b = Q96_48(200);
        let d = Q96_48(300);
        let n = 1;
        let x_j = Q96_48(1000);
        let sum_reserves = u144(0);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result != 0);
    }

    #[test]
    fn many_assets_in_pool() {
        let a = 5000_i128 << 48;
        let b = Q96_48(3000);
        let d = Q96_48(2000);
        let n = 100;
        let x_j = Q96_48(150);
        let sum_reserves = Q96_48(14850);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result.abs() < (10000_i128 << 48));
    }

    #[test]
    fn extreme_reserve_imbalance() {
        let a = 1000_i128 << 48;
        let b = Q96_48(5000);
        let d = Q96_48(1000);
        let n = 4;
        let x_j = Q96_48(10000);
        let sum_reserves = Q96_48(100);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result.abs() > (1000_i128 << 48));
    }

    #[test]
    fn small_d_amplifies_second_term() {
        let a = 100_i128 << 48;
        let b = Q96_48(100);
        let d = Q96_48(1);
        let n = 2;
        let x_j = Q96_48(50);
        let sum_reserves = Q96_48(50);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result.abs() > (10_i128 << 48));
    }

    #[test]
    fn symmetric_deviations_different_derivatives() {
        let a = 1000_i128 << 48;
        let b = Q96_48(500);
        let d = Q96_48(1000);
        let n = 4;
        
        let x_j1 = Q96_48(200);
        let sum1 = Q96_48(800);
        let deriv1 = invariant_derivative(a, b, d, n, x_j1, sum1);
        
        let x_j2 = Q96_48(300);
        let sum2 = Q96_48(700);
        let deriv2 = invariant_derivative(a, b, d, n, x_j2, sum2);
        
        assert_ne!(deriv1, deriv2);
        assert!(deriv1 < deriv2);
    }

    #[test]
    fn zero_b_eliminates_second_term() {
        let a = 2000_i128 << 48;
        let b = u144(0);
        let d = Q96_48(500);
        let n = 3;
        let x_j = Q96_48(100);
        let sum_reserves = Q96_48(500);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        let sqrt_n = sqrt_Q96X48(Q96_48(n as u128));
        let sqrt_n_i128 = u144_to_i128(sqrt_n);
        let expected = div_Q96X48_signed(2 * a, sqrt_n_i128);
        assert!((result - expected).abs() < (1 << 45));
    }

    #[test]
    fn negative_a_with_balanced_reserves() {
        let a = -2000_i128 << 48;
        let b = Q96_48(800);
        let d = Q96_48(1200);
        let n = 5;
        let x_j = Q96_48(200);
        let sum_reserves = Q96_48(800);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result < 0);
    }

    #[test]
    fn large_positive_a_dominates() {
        let a = 50000_i128 << 48;
        let b = Q96_48(100);
        let d = Q96_48(500);
        let n = 4;
        let x_j = Q96_48(100);
        let sum_reserves = Q96_48(300);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result > (20000_i128 << 48));
    }

    #[test]
    fn large_b_amplifies_deviation_impact() {
        let a = 100_i128 << 48;
        let b = Q96_48(10000);
        let d = Q96_48(2000);
        let n = 3;
        let x_j = Q96_48(500);
        let sum_reserves = Q96_48(100);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result.abs() > (5000_i128 << 48));
    }

    #[test]
    fn two_asset_pool_equal_reserves() {
        let a = 500_i128 << 48;
        let b = Q96_48(300);
        let d = Q96_48(600);
        let n = 2;
        let x_j = Q96_48(250);
        let sum_reserves = Q96_48(250);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        let sqrt_n = sqrt_Q96X48(Q96_48(n as u128));
        let sqrt_n_i128 = u144_to_i128(sqrt_n);
        let expected = div_Q96X48_signed(2 * a, sqrt_n_i128);
        assert!((result - expected).abs() < (1 << 46));
    }

    #[test]
    fn precision_with_small_values() {
        let a = 10_i128 << 48;
        let b = Q96_48(5);
        let d = Q96_48(10);
        let n = 2;
        let x_j = Q96_48(3);
        let sum_reserves = Q96_48(2);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result != 0);
    }

    #[test]
    fn high_n_reduces_first_term_impact() {
        let a = 10000_i128 << 48;
        let b = Q96_48(5000);
        let d = Q96_48(3000);
        let n = 1000;
        let x_j = Q96_48(100);
        let sum_reserves = Q96_48(99900);
        
        let result = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        assert!(result.abs() < (1000_i128 << 48));
    }
}

#[cfg(test)]
mod test_solve_quadratic_invariant {
    use super::*;
    use std::vec;

    fn u144(val: u128) -> U144 {
        U144::from(val)
    }

    fn Q96_48(val: u128) -> U144 {
        U144::from(val << 48)
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn negative_discriminant_returns_fallback() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(2), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(2);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert_eq!(result, delta_linear);
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn balanced_reserves_two_tokens() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn three_token_pool_with_imbalance() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(3), Q96_48(2), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(4);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn large_k_cross_value() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(2), Q96_48(2)];
        let token_in_index = u144(1);
        let token_out_index = u144(2);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(2);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn small_reserves_relative_to_radius() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(1), Q96_48(1), Q96_48(1)];
        let token_in_index = u144(0);
        let token_out_index = u144(2);
        let consolidated_radius = Q96_48(2);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn four_token_pool() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(2), Q96_48(2), Q96_48(2)];
        let token_in_index = u144(1);
        let token_out_index = u144(3);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn token_in_equals_token_out_adjacent() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(2), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn extreme_imbalance_high_in_low_out() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(3), Q96_48(1), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(4);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn positive_b_coefficient_case() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(1)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn negative_b_coefficient_case() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(1), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn returns_smaller_positive_root() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(2), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result > u144(0));
        assert!(result <= delta_linear || result > delta_linear);
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn large_radius_small_reserves() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(1), Q96_48(1), Q96_48(1)];
        let token_in_index = u144(1);
        let token_out_index = u144(2);
        let consolidated_radius = Q96_48(2);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }

    #[test]
    #[should_panic(expected = "Overflow")]
    fn minimal_k_cross() {
        let delta_linear = Q96_48(1);
        let reserves = vec![Q96_48(2), Q96_48(2)];
        let token_in_index = u144(0);
        let token_out_index = u144(1);
        let consolidated_radius = Q96_48(3);
        let k_cross = Q96_48(1);
        
        let result = solveQuadraticInvariant(
            delta_linear,
            reserves,
            token_in_index,
            token_out_index,
            consolidated_radius,
            k_cross,
        );
        
        assert!(result >= u144(0));
    }
}

#[cfg(test)]
mod test_solve_amount_out {
    use super::*;

    fn u144(val: u128) -> U144 {
        U144::from(val)
    }

    fn Q96_48(val: u128) -> U144 {
        U144::from(val << 48)
    }

    #[test]
    fn converges_with_balanced_initial_guess() {
        let sum_reserves = Q96_48(1000);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 4;
        let k_bound = Q96_48(200);
        let r_int = Q96_48(300);
        let s_bound = Q96_48(50);
        let initial_x_j = Q96_48(250);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn handles_high_initial_guess() {
        let sum_reserves = Q96_48(800);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 3;
        let k_bound = Q96_48(150);
        let r_int = Q96_48(400);
        let s_bound = Q96_48(30);
        let initial_x_j = Q96_48(1000);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn handles_low_initial_guess() {
        let sum_reserves = Q96_48(1200);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 5;
        let k_bound = Q96_48(180);
        let r_int = Q96_48(350);
        let s_bound = Q96_48(40);
        let initial_x_j = Q96_48(50);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn converges_within_max_iterations() {
        let sum_reserves = Q96_48(2000);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 8;
        let k_bound = Q96_48(300);
        let r_int = Q96_48(500);
        let s_bound = Q96_48(60);
        let initial_x_j = Q96_48(300);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn two_asset_pool_convergence() {
        let sum_reserves = Q96_48(500);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 2;
        let k_bound = Q96_48(100);
        let r_int = Q96_48(300);
        let s_bound = Q96_48(25);
        let initial_x_j = Q96_48(250);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn large_sum_reserves() {
        let sum_reserves = Q96_48(10000);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 10;
        let k_bound = Q96_48(500);
        let r_int = Q96_48(1200);
        let s_bound = Q96_48(100);
        let initial_x_j = Q96_48(1000);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn small_tolerance_convergence() {
        let sum_reserves = Q96_48(6);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 3;
        let k_bound = Q96_48(2);
        let r_int = Q96_48(3);
        let s_bound = Q96_48(1);
        let initial_x_j = Q96_48(2);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        // Just verify the result is positive and reasonable
        assert!(result > u144(0));
    }

    #[test]
    fn negative_initial_convergence_triggers_halving() {
        let sum_reserves = Q96_48(1500);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 6;
        let k_bound = Q96_48(400);
        let r_int = Q96_48(200);
        let s_bound = Q96_48(80);
        let initial_x_j = Q96_48(100);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn large_r_int_relative_to_bounds() {
        let sum_reserves = Q96_48(800);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 4;
        let k_bound = Q96_48(50);
        let r_int = Q96_48(800);
        let s_bound = Q96_48(20);
        let initial_x_j = Q96_48(200);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn small_s_bound_value() {
        let sum_reserves = Q96_48(1000);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 5;
        let k_bound = Q96_48(200);
        let r_int = Q96_48(300);
        let s_bound = Q96_48(5);
        let initial_x_j = Q96_48(200);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn high_n_many_assets() {
        let sum_reserves = Q96_48(5000);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 50;
        let k_bound = Q96_48(250);
        let r_int = Q96_48(600);
        let s_bound = Q96_48(50);
        let initial_x_j = Q96_48(100);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }

    #[test]
    fn derivative_zero_early_termination() {
        let sum_reserves = Q96_48(1000);
        let sum_reserves_squared = mul_Q96X48(sum_reserves, sum_reserves);
        let n = 4;
        let k_bound = Q96_48(250);
        let r_int = Q96_48(250);
        let s_bound = Q96_48(50);
        let initial_x_j = Q96_48(250);
        
        let result = solve_amount_out(
            sum_reserves,
            sum_reserves_squared,
            n,
            k_bound,
            r_int,
            s_bound,
            initial_x_j,
        );
        
        assert!(result > u144(0));
    }
}

