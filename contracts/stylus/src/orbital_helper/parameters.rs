use alloy_primitives::aliases::U144;
use crate::orbital_helper::fixed_point::{
    add_Q96X48, convert_to_Q96X48, div_Q96X48, mul_Q96X48, sqrt_Q96X48, sub_Q96X48,
};

// calculate radius from reserves and n
pub fn calculate_radius(reserve: U144) -> U144 {
    let root5 = U144::from(629397181890196u128);
    let one = convert_to_Q96X48(U144::from(1));
    let denominator = sub_Q96X48(one, div_Q96X48(one, root5));
    return div_Q96X48(reserve, denominator);
}

// calculate k from p and r using the formula: k = r√n - r(p+n-1)/√(n(p²+n-1))
pub fn calculateK(depeg_limit: U144, radius: U144) -> U144 {
    // Note: assuming n = 5 based on the context (golden ratio calculations)
    let n = convert_to_Q96X48(U144::from(5));
    let one = convert_to_Q96X48(U144::from(1));

    // Calculate √n
    let sqrt_n = sqrt_Q96X48(n);

    // Calculate first term: r√n
    let first_term = mul_Q96X48(radius, sqrt_n);

    // Calculate p² (depeg_limit is already in Q96X48 format)
    let p_squared = mul_Q96X48(depeg_limit, depeg_limit);

    // Calculate p + n - 1
    let p_plus_n_minus_1 = sub_Q96X48(
        add_Q96X48(depeg_limit, n),
        one
    );

    // Calculate p² + n - 1
    let p_squared_plus_n_minus_1 = sub_Q96X48(
        add_Q96X48(p_squared, n),
        one
    );

    // Calculate n(p² + n - 1)
    let n_times_expr = mul_Q96X48(n, p_squared_plus_n_minus_1);

    // Calculate √(n(p² + n - 1))
    let sqrt_denominator = sqrt_Q96X48(n_times_expr);

    // Calculate r(p + n - 1)
    let numerator_second_term = mul_Q96X48(radius, p_plus_n_minus_1);

    // Calculate second term: r(p + n - 1) / √(n(p² + n - 1))
    let second_term = div_Q96X48(numerator_second_term, sqrt_denominator);

    // Calculate final result: r√n - r(p + n - 1) / √(n(p² + n - 1))
    sub_Q96X48(first_term, second_term)
}

// return k and r together
pub fn getTickParameters(depeg_limit: U144, reserve: U144) -> (U144, U144) {
    let radius = calculate_radius(reserve);
    let k = calculateK(depeg_limit, radius);
    (k, radius)
}

pub fn calculateBoundaryTickS(radius: U144, k: U144) -> U144 {
    // Implement the boundary tick calculation logic here
    // s = sqrt(r² - (k - r√n)²)
    let n_sqrt = sqrt_Q96X48(convert_to_Q96X48(U144::from(5)));
    let difference = sub_Q96X48(k, mul_Q96X48(radius, n_sqrt));
    let radius_squared = mul_Q96X48(radius, radius);
    let difference_squared = mul_Q96X48(difference, difference);
    sqrt_Q96X48(sub_Q96X48(radius_squared, difference_squared))
}
