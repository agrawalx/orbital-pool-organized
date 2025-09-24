
#![cfg_attr(not(feature = "export-abi"), no_main)]
#![allow(non_snake_case)] // Allows camelCase function names to match Solidity.

#[macro_use]
extern crate alloc;

use alloc::vec::Vec;

/// Import items from the SDK. The prelude contains common traits and macros.
use stylus_sdk::{alloy_primitives::U256, prelude::*, storage::StorageU256};
use alloy_primitives::aliases::U144;

// Define some persistent storage using the Solidity ABI.
// `Counter` will be the entrypoint.
#[storage]
#[entrypoint]
pub struct OrbitalHelper {
    /// A number stored in contract storage.
    _placeholder: StorageU256,
}
// everything is in Q98X48 fixed point format unless otherwise specified
// Q96X48 format means there are 96 bits for the integer part and 48 bits for the fractional part.
/// Declare that `OrbitalHelper` is a contract with the following external methods.
#[public]
impl OrbitalHelper {
    // implement arithemtic operations for Q96X48 format and then use that for all further calculations.
    pub fn convert_to_Q96X48(value: U144) -> U144 {
        value << 48
    }
    pub fn convert_from_Q96X48(value: U144) -> U144 {
        value >> 48
    }
    pub fn add_Q96X48(a: U144, b: U144) -> U144 {
        a + b
    }
    pub fn sub_Q96X48(a: U144, b: U144) -> U144 {
        a - b
    }
    pub fn mul_Q96X48(a: U144, b: U144) -> U144 {
        // (a * b) >> 48
        let product: U256 = U256::from(a) * U256::from(b);
        let shifted: U256 = product >> 48;
        // Check if the result fits in U144 (2^144 - 1)
        
        // Convert U256 to U144 by taking the lower 144 bits
        // U144 is represented internally as [u64; 3], so we take the first 2.25 u64s
        let limbs = shifted.as_limbs();
        let low = limbs[0];et max_u144 = (U256::from(1u128) << 144) - U256::from(1u128)
        let mid = limbs[1];
        let high = limbs[2] & 0xFFFF; // Only take lower 16 bits of the third limb (144 - 128 = 16)
        U144::from_limbs([low, mid, high])
    }

    pub fn div_Q96X48(a: U144, b: U144) -> U144 {
        // (a << 48) / b
        assert!(b != U144::ZERO, "Division by zero");
        let dividend: U256 = U256::from(a) << 48;
        let result: U256 = dividend / U256::from(b);
        // Check if the result fits in U144 (2^144 - 1)
        // Convert U256 to U144 by taking the lower 144 bits
        // U144 is represented internally as [u64; 3], so we take the first 2.25 u64s
        let limbs = result.as_limbs();
        let low = limbs[0];
        let mid = limbs[1];
        let high = limbs[2] & 0xFFFF; // Only take lower 16 bits of the third limb (144 - 128 = 16)
        U144::from_limbs([low, mid, high])
    }

    // Square root function for Q96X48 format using Newton's method
    pub fn sqrt_Q96X48(y: U144) -> U144 {
        if y == U144::ZERO {
            return U144::ZERO;
        }

        // Convert y to U256 and shift by 48 to account for fixed-point precision
        let mut z: U256 = U256::from(y) << 48;

        // Initial guess
        let two = U256::from(2u8);
        let one = U256::from(1u8);
        let mut x = z / two + one;

        // Babylonian method loop
        while x < z {
            z = x;
            x = (z + (U256::from(y) << 48) / z) / two;
        }

        // Convert result back to Q96X48
        let result = z;

        // Convert U256 -> U144
        let limbs = result.as_limbs();
        let low = limbs[0];
        let mid = limbs[1];
        let high = limbs[2] & 0xFFFF; // only lower 16 bits for U144
        U144::from_limbs([low, mid, high])
    }

    // calculate radius from reserves and n
    pub fn calculate_radius(reserve: U144) -> U144 {
        let root5 = U144::from(629397181890196u128);
        let one = Self::convert_to_Q96X48(U144::from(1));
        let denominator = Self::sub_Q96X48(one, Self::div_Q96X48(one, root5));
        return Self::div_Q96X48(reserve, denominator);
    }
    // calculate k from p and r using the formula: k = r√n - r(p+n-1)/√(n(p²+n-1))
    pub fn calculateK(depeg_limit: U144, radius: U144) -> U144 {
        // Note: assuming n = 5 based on the context (golden ratio calculations)
        let n = Self::convert_to_Q96X48(U144::from(5));
        let one = Self::convert_to_Q96X48(U144::from(1));
        
        // Calculate √n
        let sqrt_n = Self::sqrt_Q96X48(n);
        
        // Calculate first term: r√n
        let first_term = Self::mul_Q96X48(radius, sqrt_n);
        
        // Calculate p² (depeg_limit is already in Q96X48 format)
        let p_squared = Self::mul_Q96X48(depeg_limit, depeg_limit);
        
        // Calculate p + n - 1
        let p_plus_n_minus_1 = Self::sub_Q96X48(
            Self::add_Q96X48(depeg_limit, n),
            one
        );
        
        // Calculate p² + n - 1
        let p_squared_plus_n_minus_1 = Self::sub_Q96X48(
            Self::add_Q96X48(p_squared, n),
            one
        );
        
        // Calculate n(p² + n - 1)
        let n_times_expr = Self::mul_Q96X48(n, p_squared_plus_n_minus_1);
        
        // Calculate √(n(p² + n - 1))
        let sqrt_denominator = Self::sqrt_Q96X48(n_times_expr);
        
        // Calculate r(p + n - 1)
        let numerator_second_term = Self::mul_Q96X48(radius, p_plus_n_minus_1);
        
        // Calculate second term: r(p + n - 1) / √(n(p² + n - 1))
        let second_term = Self::div_Q96X48(numerator_second_term, sqrt_denominator);
        
        // Calculate final result: r√n - r(p + n - 1) / √(n(p² + n - 1))
        Self::sub_Q96X48(first_term, second_term)
    }
    // return k and r together
    pub fn getTickParameters(depeg_limit: U144, reserve: U144) -> (U144, U144) {
        let radius = Self::calculate_radius(reserve);
        let k = Self::calculateK(depeg_limit, radius);
        (k, radius)
    }
    pub fn calculateBoundaryTickS(radius: U144, k: U144) -> U144 {
        // Implement the boundary tick calculation logic here
        // s =  sqrt(r² - (k - r√n)²)
        let difference = Self::sub_Q96X48(k, Self::mul_Q96X48(radius, Self::sqrt_Q96X48(U144::from(5))));
        Self::sqrt_Q96X48(Self::sub_Q96X48(Self::mul_Q96X48(radius, radius), Self::mul_Q96X48(difference, difference))) 
    }

    /// Solves the quadratic invariant equation to find the amount needed to cross a tick boundary
    /// Based on the formula:
    /// a = 1
    /// b = A + B = (r - x_out - P) + (-(r - x_in)) = -x_out - P + x_in
    /// c = (A² + B² - C) / 2
    /// where P = k_cross * r - Σx_i and C = r² - Σ(r - x_i)² for i ≠ in, out
    /// All values are expected to be in Q96X48 fixed-point format
    pub fn solveQuadraticInvariant(
        delta_linear: U144,
        reserves: Vec<U144>,
        token_in_index: U144,
        token_out_index: U144,
        consolidated_radius: U144,
        k_cross: U144,
    ) -> U144 {
        let r = consolidated_radius;
        let token_in_idx = token_in_index.as_limbs()[0] as usize;
        let token_out_idx = token_out_index.as_limbs()[0] as usize;
        
        // Calculate P = k_cross * r - Σx_i
        let mut sum_reserves = U144::ZERO;
        for &reserve in &reserves {
            sum_reserves = Self::add_Q96X48(sum_reserves, reserve);
        }
        let k_cross_times_r = Self::mul_Q96X48(k_cross, r);
        let p = Self::sub_Q96X48(k_cross_times_r, sum_reserves);
        
        // Get x_in and x_out
        let x_in = reserves[token_in_idx];
        let x_out = reserves[token_out_idx];
        
        // Calculate C = r² - Σ(r - x_i)² for i ≠ in, out
        let r_squared = Self::mul_Q96X48(r, r);
        let mut sum_squared_differences = U144::ZERO;
        
        for (i, &reserve) in reserves.iter().enumerate() {
            if i != token_in_idx && i != token_out_idx {
                let diff = Self::sub_Q96X48(r, reserve);
                let diff_squared = Self::mul_Q96X48(diff, diff);
                sum_squared_differences = Self::add_Q96X48(sum_squared_differences, diff_squared);
            }
        }
        
        let c_term = Self::sub_Q96X48(r_squared, sum_squared_differences);
        
        // Calculate coefficients for the quadratic equation ax² + bx + c = 0
        // Based on the formula from the attachment:
        // a = 1
        // b = A + B = (r - x_out - P) + (-(r - x_in)) = -x_out - P + x_in
        // c = (A² + B² - C) / 2
        
        let a = Self::convert_to_Q96X48(U144::from(1));
        
        // Calculate A = r - x_out - P
        let r_minus_x_out = Self::sub_Q96X48(r, x_out);
        let a_term = Self::sub_Q96X48(r_minus_x_out, p);
        
        // Calculate B = -(r - x_in) = x_in - r  
        let b_term = Self::sub_Q96X48(x_in, r);
        
        // Calculate b = A + B = (r - x_out - P) + (x_in - r) = x_in - x_out - P
        // We need to be careful about potential underflows
        let mut b = U144::ZERO;
        let mut b_is_positive = true;
        
        // Calculate x_in - x_out first
        if x_in >= x_out {
            let diff = Self::sub_Q96X48(x_in, x_out);
            if diff >= p {
                b = Self::sub_Q96X48(diff, p);
                b_is_positive = true;
            } else {
                b = Self::sub_Q96X48(p, diff);
                b_is_positive = false;
            }
        } else {
            let diff = Self::sub_Q96X48(x_out, x_in);
            b = Self::add_Q96X48(diff, p);
            b_is_positive = false;
        }
        
        // For A and B terms to calculate c:
        // A = r - x_out - P (already calculated as a_term)
        // B = x_in - r (already calculated as b_term)
        
        // c = (A² + B² - C) / 2
        let a_squared = Self::mul_Q96X48(a_term, a_term);
        let b_squared = Self::mul_Q96X48(b_term, b_term);
        let numerator = Self::sub_Q96X48(
            Self::add_Q96X48(a_squared, b_squared),
            c_term
        );
        let two = Self::convert_to_Q96X48(U144::from(2));
        let c = Self::div_Q96X48(numerator, two);
        
        // Solve quadratic equation: ax² + bx + c = 0
        // Using quadratic formula: x = (-b ± √(b² - 4ac)) / 2a
        
        // Calculate discriminant: b² - 4ac
        let b_squared_for_discriminant = Self::mul_Q96X48(b, b);
        let four = Self::convert_to_Q96X48(U144::from(4));
        let four_ac = Self::mul_Q96X48(
            Self::mul_Q96X48(four, a),
            c
        );
        
        // Check if discriminant is positive
        if b_squared_for_discriminant < four_ac {
            // No real solution, return delta_linear as fallback
            return delta_linear;
        }
        
        let discriminant = Self::sub_Q96X48(b_squared_for_discriminant, four_ac);
        let sqrt_discriminant = Self::sqrt_Q96X48(discriminant);
        
        // Calculate roots considering the sign of b
        let two_a = Self::mul_Q96X48(two, a);
        
        let (x1, x2) = if b_is_positive {
            // b is positive, so -b is negative
            // x1 = (-b + √discriminant) / 2a = (√discriminant - b) / 2a
            // x2 = (-b - √discriminant) / 2a = -(b + √discriminant) / 2a
            let x1 = if sqrt_discriminant >= b {
                Self::div_Q96X48(
                    Self::sub_Q96X48(sqrt_discriminant, b),
                    two_a
                )
            } else {
                U144::ZERO // This root would be negative
            };
            
            // x2 would be negative, so we set it to zero
            let x2 = U144::ZERO;
            (x1, x2)
        } else {
            // b is negative (stored as positive value), so -b is positive
            // x1 = (-(-b) + √discriminant) / 2a = (b + √discriminant) / 2a
            // x2 = (-(-b) - √discriminant) / 2a = (b - √discriminant) / 2a
            let x1 = Self::div_Q96X48(
                Self::add_Q96X48(b, sqrt_discriminant),
                two_a
            );
            
            let x2 = if b >= sqrt_discriminant {
                Self::div_Q96X48(
                    Self::sub_Q96X48(b, sqrt_discriminant),
                    two_a
                )
            } else {
                U144::ZERO // This root would be negative
            };
            (x1, x2)
        };
        
        // Calculate x1 - P and x2 - P, then return whichever is positive
        let x1_minus_p = if x1 >= p {
            Self::sub_Q96X48(x1, p)
        } else {
            U144::ZERO // Would be negative
        };
        
        let x2_minus_p = if x2 >= p {
            Self::sub_Q96X48(x2, p)
        } else {
            U144::ZERO // Would be negative
        };
        
        // Return the positive result, prioritizing the smaller positive value
        if x1_minus_p > U144::ZERO && x2_minus_p > U144::ZERO {
            // Return the smaller positive value for boundary crossing
            if x1_minus_p <= x2_minus_p { x1_minus_p } else { x2_minus_p }
        } else if x1_minus_p > U144::ZERO {
            x1_minus_p
        } else if x2_minus_p > U144::ZERO {
            x2_minus_p
        } else {
            // No positive solution, return delta_linear as fallback
            delta_linear
        }
    }

    
    pub fn solve_amount_out(
        &self,
        reserves: Vec<U144>,
        amount_in: U144,
        token_in_index: U256,
        token_out_index: U256,
        k_bound: U144,
        r_int: U144,
        s_bound: U144,
    ) -> U144 {
        let token_in_idx = token_in_index.as_limbs()[0] as usize;
        let token_out_idx = token_out_index.as_limbs()[0] as usize;

        let n = reserves.len();
        if n == 0 {
            return U144::ZERO;
        }

        let initial_invariant = Self::calculate_invariant(&reserves, k_bound, r_int, s_bound);

        let mut temp_reserves = reserves.clone();
        temp_reserves[token_in_idx] = Self::add_Q96X48(temp_reserves[token_in_idx], amount_in);

        let initial_reserve_out = reserves[token_out_idx];
        
        let initial_guess_amount_out = amount_in;
        let mut x = if initial_reserve_out > initial_guess_amount_out {
            Self::sub_Q96X48(initial_reserve_out, initial_guess_amount_out)
        } else {
            U144::ZERO // Avoid underflow
        };

        for _ in 0..10 {
            let new_invariant = {
                let mut current_reserves = temp_reserves.clone();
                current_reserves[token_out_idx] = x;
                Self::calculate_invariant(&current_reserves, k_bound, r_int, s_bound)
            };

            let (fx, fx_is_negative) = if new_invariant >= initial_invariant {
                (Self::sub_Q96X48(new_invariant, initial_invariant), false)
            } else {
                (Self::sub_Q96X48(initial_invariant, new_invariant), true)
            };

            if fx < U144::from(1000) {
                break;
            }

            let h = {
                let h_min = Self::convert_to_Q96X48(U144::from(1));
                let h_dyn = Self::div_Q96X48(amount_in, Self::convert_to_Q96X48(U144::from(1000)));
                if h_dyn > h_min { h_dyn } else { h_min }
            };

            let fx_plus_h_val = {
                let mut current_reserves = temp_reserves.clone();
                current_reserves[token_out_idx] = Self::add_Q96X48(x, h);
                Self::calculate_invariant(&current_reserves, k_bound, r_int, s_bound)
            };
            
            let fx_minus_h_val = {
                let mut current_reserves = temp_reserves.clone();
                if x > h {
                    current_reserves[token_out_idx] = Self::sub_Q96X48(x, h);
                } else {
                    current_reserves[token_out_idx] = U144::ZERO;
                }
                Self::calculate_invariant(&current_reserves, k_bound, r_int, s_bound)
            };

            let (delta_f, _delta_f_is_negative) = if fx_plus_h_val >= fx_minus_h_val {
                (Self::sub_Q96X48(fx_plus_h_val, fx_minus_h_val), false)
            } else {
                (Self::sub_Q96X48(fx_minus_h_val, fx_plus_h_val), true)
            };

            let two_h = Self::add_Q96X48(h, h);
            if two_h == U144::ZERO { break; }

            let derivative = Self::div_Q96X48(delta_f, two_h);
            if derivative == U144::ZERO { break; }

            let update_term = Self::div_Q96X48(fx, derivative);

            if fx_is_negative {
                x = Self::add_Q96X48(x, update_term);
            } else {
                if x > update_term {
                    x = Self::sub_Q96X48(x, update_term);
                } else {
                    x = U144::ZERO;
                }
            }
        }

        let final_reserve_out = if x < initial_reserve_out { x } else { initial_reserve_out };
        let final_reserve_out = if final_reserve_out > U144::ZERO { final_reserve_out } else { U144::ZERO };
        
        if initial_reserve_out > final_reserve_out {
            Self::sub_Q96X48(initial_reserve_out, final_reserve_out)
        } else {
            U144::ZERO
        }
    }
}

// Private helper functions for OrbitalHelper
impl OrbitalHelper {
    // Helper function to calculate variance term from reserves
    fn calculate_variance_term(reserves: &[U144]) -> U144 {
        let n_val = reserves.len();
        if n_val == 0 {
            return U144::ZERO;
        }
        let n = Self::convert_to_Q96X48(U144::from(n_val));
        
        let mut sum_total = U144::ZERO;
        let mut sum_squares = U144::ZERO;
        
        for &reserve in reserves {
            sum_total = Self::add_Q96X48(sum_total, reserve);
            let squared = Self::mul_Q96X48(reserve, reserve);
            sum_squares = Self::add_Q96X48(sum_squares, squared);
        }
        
        // Calculate √(Σx²_total_i - 1/n(Σx_total_i)²)
        let sum_total_squared = Self::mul_Q96X48(sum_total, sum_total);
        let one_over_n_sum_squared = Self::div_Q96X48(sum_total_squared, n);
        let variance_inner = Self::sub_Q96X48(sum_squares, one_over_n_sum_squared);
        Self::sqrt_Q96X48(variance_inner)
    }

    fn calculate_invariant(
        reserves: &[U144],
        k_bound: U144,
        r_int: U144,
        s_bound: U144,
    ) -> U144 {
        // This function calculates the full invariant from the paper:
        // r_int^2 = ((xtotal ⋅ v - k_bound) - r_int*√n)^2 + (||w_total|| - s_bound)^2
        // where:
        // xtotal ⋅ v = Σx_i / √n
        // ||w_total|| = sqrt( Σx_i^2 - (1/n)(Σx_i)^2 )  (variance term)
        // s_bound = sqrt(r_bound^2 - (k_bound - r_bound*√n)^2)

        let n_val = reserves.len();
        if n_val == 0 {
            return U144::ZERO;
        }
        let n = Self::convert_to_Q96X48(U144::from(n_val));
        let sqrt_n = Self::sqrt_Q96X48(n);

        // Calculate first term: ((Σx_i/√n - k_bound) - r_int*√n)²
        let sum_reserves: U144 = reserves.iter().fold(U144::ZERO, |acc, &x| Self::add_Q96X48(acc, x));
        let sum_reserves_div_sqrt_n = Self::div_Q96X48(sum_reserves, sqrt_n);
        let r_int_mul_sqrt_n = Self::mul_Q96X48(r_int, sqrt_n);

        let first_term_inner = Self::sub_Q96X48(
            Self::sub_Q96X48(sum_reserves_div_sqrt_n, k_bound),
            r_int_mul_sqrt_n,
        );
        let first_term_squared = Self::mul_Q96X48(first_term_inner, first_term_inner);

        // Calculate second term: (variance_term - s_bound)²
        let variance_term = Self::calculate_variance_term(reserves);
        let second_term_inner = Self::sub_Q96X48(variance_term, s_bound);
        let second_term_squared = Self::mul_Q96X48(second_term_inner, second_term_inner);

        // Invariant = first_term² + second_term²
        Self::add_Q96X48(first_term_squared, second_term_squared)
    }
}