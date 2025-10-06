from math import sqrt
import numpy as np 
Q = 48 
SCALE = 1 << Q  # 2^48

# Test cases for reserves (lists of Q96.48 numbers)
test_reserves = [
    # all equal reserves
    [1*SCALE, 1*SCALE, 1*SCALE, 1*SCALE, 1*SCALE],

    # increasing sequence
    [1*SCALE, 2*SCALE, 3*SCALE, 4*SCALE, 5*SCALE],

    # larger numbers
    [10*SCALE, 20*SCALE, 30*SCALE, 40*SCALE, 50*SCALE],

    # mixed values
    [5*SCALE, 15*SCALE, 25*SCALE, 35*SCALE, 45*SCALE],

    # edge case: one zero
    [0, 2*SCALE, 4*SCALE, 6*SCALE, 8*SCALE],

    # edge case: all zeros
    [0, 0, 0, 0, 0],
]

def to_U144(value: int) -> int:
    """
    Converts an integer (potentially > 144 bits) to a U144 by taking the lower 144 bits.
    Equivalent to U256 -> U144 conversion in Rust/Stylus.
    """
    return value & ((1 << 144) - 1)

def convert_to_Q96X48(value: int) -> int:
    """Converts an integer to Q96.48 format by left-shifting."""
    return value << Q

def convert_from_Q96X48(value: int) -> int:
    """Converts a Q96.48 value back to an integer by right-shifting."""
    return value >> Q

def add_Q96X48(a: int, b: int) -> int:
    """Adds two Q96.48 numbers."""
    return a + b

def sub_Q96X48(a: int, b: int) -> int:
    """Subtracts two Q96.48 numbers."""
    return a - b

def mul_Q96X48(a: int, b: int) -> int:
    product = a * b
    shifted = product >> Q
    return to_U144(shifted)

def div_Q96X48(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("Division by zero")
    dividend = a << Q
    result = dividend // b
    return to_U144(result)

def div_Q96X48_signed(a: int, b: int) -> int:
    """Division for Q96.48 numbers that properly handles negative results"""
    if b == 0:
        raise ValueError("Division by zero")
    dividend = a << Q
    result = dividend // b
    # Don't apply U144 mask to preserve sign for negative results
    return result
def sqrt_preSclae(value: int) -> float:
    return sqrt(value)
def sqrt_Q96X48(value: int) -> int:
    #this takes a Q96.48 number and returns a Q96.48 number
    if value < 0:
        raise ValueError("Cannot compute square root of a negative number")
    # Scale the value to maintain precision
    scaled_value = value << Q
    root = int(sqrt(scaled_value))
    return to_U144(root)
def sqrt_q96x48(y: int) -> int:
    """
    Square root function for Q96x48 fixed-point numbers using Babylonian/Newton's method.
    
    Args:
        y (int): input in Q96x48 format (scaled by 2^48)
    
    Returns:
        int: sqrt(y) also in Q96x48 format
    """
    if y == 0:
        return 0

    SCALE = 1 << 48  # Q48 scaling factor

    # Convert y into U256 equivalent with extra shift for precision
    z = y << 48  # same as U256::from(y) << 48 in Rust

    two = 2
    one = 1

    # Initial guess
    x = z // two + one

    # Babylonian iteration loop
    while x < z:
        z = x
        x = (z + ((y << 48) // z)) // two

    # Result is in z, already in Q96x48 format
    return z

def calculate_radius(reserves: list[int]) -> int:
    n = len(reserves)
    if n <= 1:
        return 0 # Or handle as an error, radius is not well-defined

    sum_reserves = sum(reserves)
    sum_squares = sum(r * r for r in reserves) >> Q
    
    # This is derived from (n-1)r^2 - 2*sum(x_i)*r + sum(x_i^2) = 0
    # The solution is r = (sum(x_i) + sqrt(sum(x_i)^2 - (n-1)sum(x_i^2))) / (n-1)
    # We choose the '+' in the quadratic formula to get the larger, correct radius.
    
    n_minus_1 = convert_to_Q96X48(n - 1)
    
    sum_reserves_sq = mul_Q96X48(sum_reserves, sum_reserves)
    
    term_to_mul = mul_Q96X48(n_minus_1, sum_squares)

    if sum_reserves_sq < term_to_mul:
        # This can happen with rounding errors for very similar reserves.
        # The term under the square root should be close to zero.
        discriminant = 0
    else:
        discriminant = sub_Q96X48(sum_reserves_sq, term_to_mul)

    sqrt_discriminant = sqrt_q96x48(discriminant)
    
    numerator = add_Q96X48(sum_reserves, sqrt_discriminant)
    denominator = n_minus_1
    
    radius = div_Q96X48(numerator, denominator)
    return radius

def calculate_k(p,n,radius):
    # kdepeg (pdepeg ) = r n - r(pdepeg +n-1) / sqrt(n(p^2_depeg +n-1))
    # p is also scaled 
    # r * sqrt(n)
    term1 = mul_Q96X48(radius, sqrt_q96x48(convert_to_Q96X48(n)))

    # r * (p + n - 1)
    numerator2 = mul_Q96X48(radius, add_Q96X48(p, convert_to_Q96X48(n - 1)))

    # p^2
    p_squared = mul_Q96X48(p, p)
    # n * (p^2 + n - 1)
    denominator2_inner = mul_Q96X48(convert_to_Q96X48(n), add_Q96X48(p_squared, convert_to_Q96X48(n - 1)))
    # sqrt(n * (p^2 + n - 1))
    denominator2 = sqrt_q96x48(denominator2_inner)

    if denominator2 == 0:
        term2 = 0
    else:
        term2 = div_Q96X48(numerator2, denominator2)

    return sub_Q96X48(term1, term2)

import math

def k_depeg(p, r, n):
    """
    Compute k_depeg(p) = r*sqrt(n) - (r*(p+n-1)) / sqrt(n*(p**2+n-1))
    
    Parameters:
        p (float): input parameter
        r (float): scaling factor
        n (float): number parameter
        
    Returns:
        float: value of k_depeg(p)
    """
    numerator = r * (p + n - 1)
    denominator = math.sqrt(n * (p**2 + n - 1))
    return r * math.sqrt(n) - numerator / denominator


def calculate_invariant(A:int ,B:int, R_int: int) -> int:
    print(add_Q96X48(mul_Q96X48(A,A), mul_Q96X48(B, B)), mul_Q96X48(R_int,R_int))
    invariant = sub_Q96X48(add_Q96X48(mul_Q96X48(A,A), mul_Q96X48(B, B)), mul_Q96X48(R_int, R_int))
    return invariant

def invariant_derivative(A:int,B:int, D:int, n:int, x_j: int, sum_reserves: int) -> int:
    # Fix: Use proper signed arithmetic for Q96.48 calculations
    
    # term1 = 2*A / sqrt(n) 
    sqrt_n = sqrt_q96x48(convert_to_Q96X48(n))
    # Use signed division to handle negative A correctly
    term1 = div_Q96X48_signed(2*A, sqrt_n)
    
    # term2 = (B / sqrt(D)) * 2 * (x_j - (sum_reserves + x_j)/n)
    term2_a = div_Q96X48(B, sqrt_q96x48(D))  # B and sqrt(D) are always positive
    
    # Calculate the inner term: x_j - (sum_reserves + x_j)/n
    avg_term = div_Q96X48(add_Q96X48(sum_reserves, x_j), convert_to_Q96X48(n))
    diff_term = x_j - avg_term  # This can be negative - use regular subtraction
    
    # term2_b = 2 * diff_term (using integer multiplication to preserve sign)
    term2_b = 2 * diff_term
    
    # term2 = term2_a * term2_b (use regular multiplication since term2_a is always positive)
    # We need to handle the fixed-point scaling manually
    term2_raw = term2_a * term2_b
    term2 = term2_raw >> Q  # Divide by 2^Q to maintain Q96.48 format
    
    # Total derivative
    derivative = term1 + term2
    return derivative 

def calculate_A_B_D(sum_reserves:int, sum_reserves_squared:int ,n:int, x_j:int, k_bound: int, r_int: int, s_bound: int) -> list[int, int, int]:
    sqrt_n = sqrt_q96x48(convert_to_Q96X48(n))
    S_plus_xj_by_rootN= div_Q96X48(add_Q96X48(sum_reserves, x_j),sqrt_n)
    print(mul_Q96X48(r_int, sqrt_n))
    A = sub_Q96X48(S_plus_xj_by_rootN, add_Q96X48(k_bound, mul_Q96X48(r_int, sqrt_n)))
    Q_plus_xj_squared = add_Q96X48(sum_reserves_squared, mul_Q96X48(x_j, x_j))
    other_term = div_Q96X48(mul_Q96X48(add_Q96X48(sum_reserves,x_j),add_Q96X48(sum_reserves,x_j)), convert_to_Q96X48(n))
    D = sub_Q96X48(Q_plus_xj_squared, other_term)
    B = sub_Q96X48(sqrt_q96x48(D),s_bound)
    return  [A, B, D]

def solve_amount_out(sum_reserves: int, sum_reserves_squared: int, n: int, k_bound: int, r_int: int, s_bound: int, x_j: int) -> int:
    """
    Solve for amount_out using Newton's method.
    
    Args:
        sum_reserves: Sum of current reserves
        sum_reserves_squared: Sum of squares of current reserves  
        n: Number of assets in the pool
        k_bound: K boundary parameter
        r_int: Radius parameter
        s_bound: S boundary parameter
    
    Returns:
        Amount out from the swap
    """
    print(f"Starting solve_amount_out", {x_j})
    x_j = x_j
    # Newton's method parameters
    max_iterations = 10
    tolerance = convert_to_Q96X48(1)/100000  # 1 unit tolerance in Q96.48 format
    
    for iteration in range(max_iterations):
        # Calculate A, B, D for current guess
        [A, B, D] = calculate_A_B_D(sum_reserves, sum_reserves_squared, n, x_j, k_bound, r_int, s_bound)
        # Calculate invariant f(x_j)
        invariant_value = calculate_invariant(A, B, r_int)
        
        # Calculate derivative f'(x_j)
        derivative_value = invariant_derivative(A, B, D, n, x_j, sum_reserves)
        # Check for convergence (invariant close to zero)
        if abs(invariant_value) < tolerance:
            return x_j
        
        # Newton's method update: x_j = x_j - f(x_j) / f'(x_j)
        # Use signed division since derivative can be negative
        delta = div_Q96X48_signed(invariant_value, derivative_value)
        x_j_new = sub_Q96X48(x_j, delta)
        print(f"Iteration {iteration}: x_j = {convert_from_Q96X48(x_j)}, f(x_j) = {convert_from_Q96X48(invariant_value)}, f'(x_j) = {convert_from_Q96X48(derivative_value)}, delta = {convert_from_Q96X48(delta)}, x_j_new = {convert_from_Q96X48(x_j_new)}")
        # Ensure x_j_new is positive
        if x_j_new <= 0:
            x_j_new = div_Q96X48(x_j, convert_to_Q96X48(2))  # Half the current value
        
        # Check for convergence in x_j
        if abs(x_j_new - x_j) < tolerance:
            return x_j_new
            
        # Update x_j for next iteration
        x_j = x_j_new
    # If we didn't converge, return the last computed value
    return x_j
    


if __name__ == "__main__":
    reserve1 = [1000*SCALE, 1000*SCALE, 1000*SCALE, 1000*SCALE, 1000*SCALE]
    reserve2 = [1005*SCALE, 1000*SCALE, 1000*SCALE, 1000*SCALE]
    k_bound = 0 
    r_int1 = calculate_radius(reserve1)
    
    # Calculate sum of reserves and sum of squares
    sum_reserves = sum(reserve2)
    sum_reserves_squared = sum(r * r for r in reserve2) >> Q  # Sum of squares in Q96.48 format
    
    x_in = 5 * SCALE  # Amount being swapped in
    n = len(reserve1)  # Number of assets
    s_bound = 0
    amount_out = solve_amount_out(sum_reserves, sum_reserves_squared, n, k_bound, r_int1, s_bound, 995*SCALE)
    print(f"Amount out: {amount_out/SCALE}")

