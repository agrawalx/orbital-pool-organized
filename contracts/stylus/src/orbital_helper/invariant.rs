use alloc::vec::Vec;
use stylus_sdk::alloy_primitives::U256;
use alloy_primitives::aliases::U144;
use crate::orbital_helper::fixed_point::{
    add_Q96X48, convert_to_Q96X48, div_Q96X48, mul_Q96X48, sqrt_Q96X48, sub_Q96X48,
    div_Q96X48_signed, u144_to_i128, i128_to_u144,
};

/// Calculate A, B, D terms based on the corrected Orbital whitepaper formulas
pub fn calculate_A_B_D(
    sum_reserves: U144,
    sum_reserves_squared: U144,
    n: u32,
    x_j: U144,
    k_bound: U144,
    r_int: U144,
    s_bound: U144,
) -> (i128, U144, U144) {
    let sqrt_n = sqrt_Q96X48(convert_to_Q96X48(U144::from(n)));
    
    // A = (S + x_j)/√n - k_bound - r_int*√n
    let s_plus_xj = add_Q96X48(sum_reserves, x_j);
    let s_plus_xj_by_sqrt_n = div_Q96X48(s_plus_xj, sqrt_n);
    let r_int_mul_sqrt_n = mul_Q96X48(r_int, sqrt_n);
    let k_bound_plus_r_int_sqrt_n = add_Q96X48(k_bound, r_int_mul_sqrt_n);
    
    // A can be negative - use signed arithmetic like Python
    // Python: A = sub_Q96X48(S_plus_xj_by_rootN, add_Q96X48(k_bound, mul_Q96X48(r_int, sqrt_n)))
    let a_signed = u144_to_i128(s_plus_xj_by_sqrt_n) - u144_to_i128(k_bound_plus_r_int_sqrt_n);
    
    // D = (Q + x_j²) - (S + x_j)²/n
    let q_plus_xj_squared = add_Q96X48(sum_reserves_squared, mul_Q96X48(x_j, x_j));
    let s_plus_xj_squared = mul_Q96X48(s_plus_xj, s_plus_xj);
    let other_term = div_Q96X48(s_plus_xj_squared, convert_to_Q96X48(U144::from(n)));
    let d = sub_Q96X48(q_plus_xj_squared, other_term);
    
    // B = √D - s_bound
    let sqrt_d = sqrt_Q96X48(d);
    let b = sub_Q96X48(sqrt_d, s_bound);
    
    (a_signed, b, d)
}

/// Calculate invariant using A² + B² - r_int²
/// This matches the Python implementation exactly
pub fn calculate_invariant_simple(a: i128, b: U144, r_int: U144) -> i128 {
    let a_abs: u128 = a.unsigned_abs();

    let a_squared: U256 = (U256::from(a_abs) * U256::from(a_abs)) >> 48;

    let limbs = a_squared.as_limbs();
    let low_u128 = ((limbs[1] as u128) << 64) | (limbs[0] as u128);
    let a_squared_u144 = U144::from(low_u128);

    let b_squared = mul_Q96X48(b, b);
    let r_int_squared = mul_Q96X48(r_int, r_int);

    let sum_squares = u144_to_i128(a_squared_u144) + u144_to_i128(b_squared);
    sum_squares - u144_to_i128(r_int_squared)
}

/// Calculate the derivative of the invariant with respect to x_j
/// This matches the Python implementation exactly
pub fn invariant_derivative(
    a: i128,
    b: U144,
    d: U144,
    n: u32,
    x_j: U144,
    sum_reserves: U144,
) -> i128 {
    let sqrt_n = sqrt_Q96X48(convert_to_Q96X48(U144::from(n)));
    
    // term1 = 2*A / sqrt(n) - use signed division
    let two_a = 2 * a;
    let sqrt_n_i128 = u144_to_i128(sqrt_n);
    let term1 = div_Q96X48_signed(two_a, sqrt_n_i128);
    
    // term2 = (B / sqrt(D)) * 2 * (x_j - (sum_reserves + x_j)/n)
    let term2_a = div_Q96X48(b, sqrt_Q96X48(d));
    
    // Calculate the inner term: x_j - (sum_reserves + x_j)/n
    let avg_term = div_Q96X48(add_Q96X48(sum_reserves, x_j), convert_to_Q96X48(U144::from(n)));
    let diff_term = u144_to_i128(x_j) - u144_to_i128(avg_term); // Can be negative
    
    // term2_b = 2 * diff_term
    let term2_b = 2 * diff_term;
    
    // term2 = term2_a * term2_b with manual fixed-point scaling
    let term2_raw = u144_to_i128(term2_a) * term2_b;
    let term2 = term2_raw >> 48; // Divide by 2^48 to maintain Q96.48 format
    
    // Total derivative
    term1 + term2
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
        sum_reserves = add_Q96X48(sum_reserves, reserve);
    }
    let k_cross_times_r = mul_Q96X48(k_cross, r);
    let p = sub_Q96X48(k_cross_times_r, sum_reserves);

    // Get x_in and x_out
    let x_in = reserves[token_in_idx];
    let x_out = reserves[token_out_idx];

    // Calculate C = r² - Σ(r - x_i)² for i ≠ in, out
    let r_squared = mul_Q96X48(r, r);
    let mut sum_squared_differences = U144::ZERO;

    for (i, &reserve) in reserves.iter().enumerate() {
        if i != token_in_idx && i != token_out_idx {
            let diff = sub_Q96X48(r, reserve);
            let diff_squared = mul_Q96X48(diff, diff);
            sum_squared_differences = add_Q96X48(sum_squared_differences, diff_squared);
        }
    }

    let c_term = sub_Q96X48(r_squared, sum_squared_differences);

    // Calculate coefficients for the quadratic equation ax² + bx + c = 0
    // Based on the formula from the attachment:
    // a = 1
    // b = A + B = (r - x_out - P) + (-(r - x_in)) = -x_out - P + x_in
    // c = (A² + B² - C) / 2

    let a = convert_to_Q96X48(U144::from(1));

    // Calculate A = r - x_out - P
    let r_minus_x_out = sub_Q96X48(r, x_out);
    let a_term = sub_Q96X48(r_minus_x_out, p);

    // Calculate B = -(r - x_in) = x_in - r
    let b_term = sub_Q96X48(x_in, r);

    // Calculate b = A + B = (r - x_out - P) + (x_in - r) = x_in - x_out - P
    // We need to be careful about potential underflows
    let mut b = U144::ZERO;
    let mut b_is_positive = true;

    // Calculate x_in - x_out first
    if x_in >= x_out {
        let diff = sub_Q96X48(x_in, x_out);
        if diff >= p {
            b = sub_Q96X48(diff, p);
            b_is_positive = true;
        } else {
            b = sub_Q96X48(p, diff);
            b_is_positive = false;
        }
    } else {
        let diff = sub_Q96X48(x_out, x_in);
        b = add_Q96X48(diff, p);
        b_is_positive = false;
    }

    // For A and B terms to calculate c:
    // A = r - x_out - P (already calculated as a_term)
    // B = x_in - r (already calculated as b_term)

    // c = (A² + B² - C) / 2
    let a_squared = mul_Q96X48(a_term, a_term);
    let b_squared = mul_Q96X48(b_term, b_term);
    let numerator = sub_Q96X48(
        add_Q96X48(a_squared, b_squared),
        c_term
    );
    let two = convert_to_Q96X48(U144::from(2));
    let c = div_Q96X48(numerator, two);

    // Solve quadratic equation: ax² + bx + c = 0
    // Using quadratic formula: x = (-b ± √(b² - 4ac)) / 2a

    // Calculate discriminant: b² - 4ac
    let b_squared_for_discriminant = mul_Q96X48(b, b);
    let four = convert_to_Q96X48(U144::from(4));
    let four_ac = mul_Q96X48(
        mul_Q96X48(four, a),
        c
    );

    // Check if discriminant is positive
    if b_squared_for_discriminant < four_ac {
        // No real solution, return delta_linear as fallback
        return delta_linear;
    }

    let discriminant = sub_Q96X48(b_squared_for_discriminant, four_ac);
    let sqrt_discriminant = sqrt_Q96X48(discriminant);

    // Calculate roots considering the sign of b
    let two_a = mul_Q96X48(two, a);

    let (x1, x2) = if b_is_positive {
        // b is positive, so -b is negative
        // x1 = (-b + √discriminant) / 2a = (√discriminant - b) / 2a
        // x2 = (-b - √discriminant) / 2a = -(b + √discriminant) / 2a
        let x1 = if sqrt_discriminant >= b {
            div_Q96X48(
                sub_Q96X48(sqrt_discriminant, b),
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
        let x1 = div_Q96X48(
            add_Q96X48(b, sqrt_discriminant),
            two_a
        );

        let x2 = if b >= sqrt_discriminant {
            div_Q96X48(
                sub_Q96X48(b, sqrt_discriminant),
                two_a
            )
        } else {
            U144::ZERO // This root would be negative
        };
        (x1, x2)
    };

    // Calculate x1 - P and x2 - P, then return whichever is positive
    let x1_minus_p = if x1 >= p {
        sub_Q96X48(x1, p)
    } else {
        U144::ZERO // Would be negative
    };

    let x2_minus_p = if x2 >= p {
        sub_Q96X48(x2, p)
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
    sum_reserves: U144,
    sum_reserves_squared: U144,
    n: u32,
    k_bound: U144,
    r_int: U144,
    s_bound: U144,
    initial_x_j: U144,
) -> U144 {
    let mut x_j = initial_x_j;
    
    // Newton's method parameters
    let max_iterations = 10;
    let tolerance = div_Q96X48(convert_to_Q96X48(U144::from(1u8)), convert_to_Q96X48(U144::from(1_000_000_000u128))); // Very small tolerance (convert_to_Q96X48(1)/10000000000)
    
    for _iteration in 0..max_iterations {
        // Calculate A, B, D for current guess
        let (a, b, d) = calculate_A_B_D(sum_reserves, sum_reserves_squared, n, x_j, k_bound, r_int, s_bound);
        
        // Calculate invariant f(x_j)
        let invariant_value = calculate_invariant_simple(a, b, r_int);
        
        // Check for convergence (invariant close to zero)
        if invariant_value.abs() < u144_to_i128(tolerance) {
            return x_j;
        }
        
        // Calculate derivative f'(x_j)
        let derivative_value = invariant_derivative(a, b, d, n, x_j, sum_reserves);
        
        if derivative_value == 0 {
            break; // Avoid division by zero
        }
        
        // Newton's method update: x_j = x_j - f(x_j) / f'(x_j)
        let delta = div_Q96X48_signed(invariant_value, derivative_value);
        let x_j_new_signed = u144_to_i128(x_j) - delta;
        
        // Ensure x_j_new is positive
        let x_j_new = if x_j_new_signed <= 0 {
            div_Q96X48(x_j, convert_to_Q96X48(U144::from(2))) // Half the current value
        } else {
            i128_to_u144(x_j_new_signed)
        };
        
        // Check for convergence in x_j
        let change = if x_j_new > x_j {
            x_j_new - x_j
        } else {
            x_j - x_j_new
        };
        
        if change < tolerance {
            return x_j_new;
        }
        
        // Update x_j for next iteration
        x_j = x_j_new;
    }
    
    x_j
}
