use stylus_sdk::alloy_primitives::U256;
use alloy_primitives::aliases::U144;

// everything is in Q98X48 fixed point format unless otherwise specified
// Q96X48 format means there are 96 bits for the integer part and 48 bits for the fractional part.
/// Declare that `OrbitalHelper` is a contract with the following external methods.

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
    let max_u144 = (U256::from(1u128) << 144) - U256::from(1u128);
    assert!(shifted <= max_u144, "Overflow in Q96X48 multiplication");
    // Convert U256 to U144 by taking the lower 144 bits
    // U144 is represented internally as [u64; 3], so we take the first 2.25 u64s
    let limbs = shifted.as_limbs();
    let low = limbs[0];
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
    let max_u144 = (U256::from(1u128) << 144) - U256::from(1u128);
    assert!(result <= max_u144, "Overflow in Q96X48 division");
    // Convert U256 to U144 by taking the lower 144 bits
    // U144 is represented internally as [u64; 3], so we take the first 2.25 u64s
    let limbs = result.as_limbs();
    let low = limbs[0];
    let mid = limbs[1];
    let high = limbs[2] & 0xFFFF; // Only take lower 16 bits of the third limb (144 - 128 = 16)
    U144::from_limbs([low, mid, high])
}

/// Divide two Q96.48 numbers with signed result handling.
/// This function preserves negative results without U144 masking.
/// Takes signed integers and returns signed result.
/// Matches Python: dividend = a << Q; result = dividend // b; return result (no mask)
pub fn div_Q96X48_signed(a: i128, b: i128) -> i128 {
    if b == 0 { return 0; }
    let sign_neg = (a < 0) ^ (b < 0);
    let a_abs: u128 = a.unsigned_abs();
    let b_abs: u128 = b.unsigned_abs().max(1);
    let dividend: U256 = U256::from(a_abs) << 48;
    let divisor: U256 = U256::from(b_abs);
    let q: U256 = dividend / divisor;
    // Clamp to i128 range
    let limbs = q.as_limbs();
    let hi = limbs[2] | limbs[3];
    let mut out: i128 = if hi != 0 || (limbs[1] > i128::MAX as u64 && limbs[2] == 0) {
        i128::MAX
    } else {
        (((limbs[1] as i128) << 64) | (limbs[0] as i128))
    };
    if sign_neg { out = out.saturating_neg(); }
    out
}


/// Helper function to convert U144 to i128 for signed operations
pub fn u144_to_i128(value: U144) -> i128 {
    // Extract the lower 128 bits and interpret as signed
    let limbs = value.as_limbs();
    let low = limbs[0] as i128;
    let mid = (limbs[1] as i128) << 64;
    low | mid
}

/// Helper function to convert i128 back to U144 (for positive results only)
pub fn i128_to_u144(value: i128) -> U144 {
    if value < 0 {
        return U144::ZERO;
    }
    U144::from(value as u128)
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
