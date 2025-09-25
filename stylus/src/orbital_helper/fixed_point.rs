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
