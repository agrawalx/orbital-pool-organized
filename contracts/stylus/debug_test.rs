fn main() {
    // Test case that failed
    let a = i128::MIN >> 1;
    println!("Original a: {}", a);
    println!("a in hex: 0x{:x}", a);
    
    // Step 1: unsigned_abs()
    let a_abs = a.unsigned_abs();
    println!("a_abs: {}", a_abs);
    println!("a_abs in hex: 0x{:x}", a_abs);
    
    // Step 2: Square it
    let a_squared_u256 = (a_abs as u128) * (a_abs as u128);
    println!("a_squared_u256: {}", a_squared_u256);
    println!("a_squared_u256 in hex: 0x{:x}", a_squared_u256);
    
    // Step 3: Right shift by 48
    let a_squared_shifted = a_squared_u256 >> 48;
    println!("a_squared_shifted: {}", a_squared_shifted);
    println!("a_squared_shifted in hex: 0x{:x}", a_squared_shifted);
    
    // Check if this becomes 0
    if a_squared_shifted == 0 {
        println!("ISSUE: a_squared_shifted is 0!");
    }
}
