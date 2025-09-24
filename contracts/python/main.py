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


def calculate_variance_term(reserves: list[int]) -> int:
    """
    Equivalent to the Rust calculate_variance_term.
    Computes sqrt( Σ x_i^2  -  (1/n)(Σ x_i)^2 ) in Q96.48 format.
    """
    # n = len(reserves) (converted to Q96.48)
    n = convert_to_Q96X48(len(reserves))

    sum_total = 0
    sum_squares = 0

    for reserve in reserves:
        sum_total = add_Q96X48(sum_total, reserve)
        squared = mul_Q96X48(reserve, reserve)
        sum_squares = add_Q96X48(sum_squares, squared)

    # (Σx_i)^2
    sum_total_squared = mul_Q96X48(sum_total, sum_total)

    # (1/n)(Σx_i)^2
    one_over_n_sum_squared = div_Q96X48(sum_total_squared, n)

    # variance_inner = Σx_i² - (1/n)(Σx_i)²
    variance_inner = sub_Q96X48(sum_squares, one_over_n_sum_squared)

    # sqrt(variance_inner)
    return sqrt_q96x48(variance_inner)

def calculate_variance_term_unscaled(reserves: list[float]) -> float:
    """
    Unscaled analog of calculate_variance_term: computes
    sqrt( Σ x_i^2  -  (1/n)(Σ x_i)^2 ) using float math and returns an unscaled float.
    """
    if not reserves:
        return 0.0
    n = float(len(reserves))
    sum_total = float(sum(reserves))
    sum_squares = float(sum(x * x for x in reserves))
    variance_inner = sum_squares - (sum_total * sum_total) / n
    if variance_inner < 0.0:
        variance_inner = 0.0  # guard against FP round-off
    return float(sqrt(variance_inner))

def calculate_invariant(reserves: list[int], k_bound: int, r_int: int, s_bound: int) -> int:
    """
    Calculates the invariant given reserves, k_bound, and r_int.
    Invariant = (Σx_i/√n - k_bound - r_int*√n)² + second_term²
    where variance_term = sqrt( Σ x_i^2  -  (1/n)(Σ x_i)^2 )
    """
    n = len(reserves)
    if n == 0:
        return 0  # or handle as an error

    sqrt_n = sqrt_q96x48(convert_to_Q96X48(n))

    sum_reserves = sum(reserves)
    sum_reserves_div_sqrt_n = div_Q96X48(sum_reserves, sqrt_n)

    r_int_mul_sqrt_n = mul_Q96X48(r_int, sqrt_n)

    first_term_inner = sub_Q96X48(sub_Q96X48(sum_reserves_div_sqrt_n, convert_to_Q96X48(k_bound)), r_int_mul_sqrt_n)
    first_term = mul_Q96X48(first_term_inner, first_term_inner)

    variance_term = calculate_variance_term(reserves)
    second_term = variance_term - s_bound
    second_term_squared = mul_Q96X48(second_term, second_term)

    invariant = add_Q96X48(first_term, second_term_squared)
    return invariant

def solve_amount_out(reserves: list[int], amount_in: int, token_in_index: int, token_out_index: int, k_bound: int, r_int: int, s_bound: int) -> int:
    """
    Solves for the amount out given an amount in using Newton's method.
    This function assumes that the invariant is maintained after the trade.
    """
    n = len(reserves)
    if n == 0:
        return 0

    initial_invariant = calculate_invariant(reserves, k_bound, r_int, s_bound)

    # Create a temporary reserves list with the input amount added
    temp_reserves = list(reserves)
    temp_reserves[token_in_index] = add_Q96X48(temp_reserves[token_in_index], amount_in)
    
    initial_reserve_out = reserves[token_out_index]

    def f(x_out_reserve: int) -> int:
        """
        Calculates the difference between the new invariant and the initial one.
        x_out_reserve is the final reserve of the output token.
        """
        current_reserves = list(temp_reserves)
        current_reserves[token_out_index] = x_out_reserve
        new_invariant = calculate_invariant(current_reserves, k_bound, r_int, s_bound)
        print("difference", initial_invariant, new_invariant, (new_invariant-initial_invariant))
        return sub_Q96X48(new_invariant, initial_invariant)

    # Initial guess for the final reserve of the output token.
    # A good guess for the amount_out is amount_in for a stablecoin swap.
    initial_guess_amount_out = amount_in
    x = sub_Q96X48(initial_reserve_out, initial_guess_amount_out)

    # Newton's method iteration
    for _ in range(10):  # 10 iterations are usually enough for convergence
        fx = f(x)
        
        # If fx is close to zero, we have found the root
        if abs(fx) < 1000: # A small tolerance
            break

        # Numerical derivative f'(x) = (f(x+h) - f(x-h)) / 2h
        # h should be a small Q96.48 value.
        h = max(amount_in // 1000, convert_to_Q96X48(1)) # Use a fraction of amount_in, with a minimum.
        
        fx_plus_h = f(x + h)
        fx_minus_h = f(x - h)
        
        delta_f = sub_Q96X48(fx_plus_h, fx_minus_h)
        two_h = add_Q96X48(h, h)

        if two_h == 0:
            break
            
        derivative = div_Q96X48(delta_f, two_h)
        print("------------------------------------", derivative)
        
        if derivative == 0:
            # Avoid division by zero; can happen if the function is flat
            break

        # Newton's update rule: x_new = x - f(x) / f'(x)
        # f(x) is Q96.48, f'(x) is also Q96.48. The division gives a Q96.48 result.
        print("-------------------------", fx, derivative)
        update_term = div_Q96X48(fx, derivative)
        x = sub_Q96X48(x, update_term)
        print("-------------------------", x, update_term)

    # Ensure the final reserve is not negative or greater than the initial reserve
    final_reserve_out = max(0, min(x, initial_reserve_out))
    print(final_reserve_out, initial_reserve_out)
    amount_out = sub_Q96X48(initial_reserve_out, final_reserve_out)

    return amount_out
if __name__ == "__main__":
    reserve1 = [1000*SCALE, 1000*SCALE, 1000*SCALE, 1000*SCALE, 1000*SCALE]
    k_bound = 0 
    r_int1 = calculate_radius(reserve1) / 2**48
    print(calculate_radius(reserve1))
    print(reserve1)
    print(5*SCALE)
    amount_out = solve_amount_out(reserve1, 5*SCALE, 0, 1, k_bound, int(r_int1*SCALE), 0)
    print(amount_out/SCALE)