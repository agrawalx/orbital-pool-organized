from math import sqrt

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
        print("Cannot calculate radius: insufficient number of reserves")
        return 0 # Or handle as an error, radius is not well-defined

    sum_reserves = sum(reserves)
    sum_squares = sum(r * r for r in reserves) >> Q
    
    print(f"\nCalculating radius:")
    print(f"Number of tokens (n): {n}")
    print(f"Sum of reserves: {convert_from_Q96X48(sum_reserves)}")
    print(f"Sum of squares: {convert_from_Q96X48(sum_squares)}")
    
    # This is derived from (n-1)r^2 - 2*sum(x_i)*r + sum(x_i^2) = 0
    # The solution is r = (sum(x_i) + sqrt(sum(x_i)^2 - (n-1)sum(x_i^2))) / (n-1)
    # We choose the '+' in the quadratic formula to get the larger, correct radius.
    
    n_minus_1 = convert_to_Q96X48(n - 1)
    sum_reserves_sq = mul_Q96X48(sum_reserves, sum_reserves)
    term_to_mul = mul_Q96X48(n_minus_1, sum_squares)
    print(f"sum_reserves^2: {convert_from_Q96X48(sum_reserves_sq)}")
    print(f"term_to_mul ( (n-1)*sum_squares ): {convert_from_Q96X48(term_to_mul)}")

    if sum_reserves_sq < term_to_mul:
        # This can happen with rounding errors for very similar reserves.
        # The term under the square root should be close to zero.
        discriminant = 0
        print("Discriminant set to 0 due to rounding (reserves very similar)")
    else:
        discriminant = sub_Q96X48(sum_reserves_sq, term_to_mul)
        print(f"Discriminant: {convert_from_Q96X48(discriminant)}")

    sqrt_discriminant = sqrt_q96x48(discriminant)
    numerator = add_Q96X48(sum_reserves, sqrt_discriminant)
    denominator = n_minus_1
    
    radius = div_Q96X48(numerator, denominator)
    print(f"Calculated radius: {convert_from_Q96X48(radius)}")
    return radius

def calculate_k(p,n,radius):
    print(f"\nCalculating k value:")
    print(f"Input parameters:")
    print(f"p (scaled): {convert_from_Q96X48(p)}")
    print(f"n: {n}")
    print(f"radius: {convert_from_Q96X48(radius)}")
    
    # kdepeg (pdepeg ) = r n - r(pdepeg +n-1) / sqrt(n(p^2_depeg +n-1))
    # p is also scaled 
    # r * sqrt(n)
    term1 = mul_Q96X48(radius, sqrt_q96x48(convert_to_Q96X48(n)))
    print(f"Term 1 (r*sqrt(n)): {convert_from_Q96X48(term1)}")

    # r * (p + n - 1)
    numerator2 = mul_Q96X48(radius, add_Q96X48(p, convert_to_Q96X48(n - 1)))
    print(f"Numerator 2 (r*(p+n-1)): {convert_from_Q96X48(numerator2)}")

    # p^2
    p_squared = mul_Q96X48(p, p)
    # n * (p^2 + n - 1)
    denominator2_inner = mul_Q96X48(convert_to_Q96X48(n), add_Q96X48(p_squared, convert_to_Q96X48(n - 1)))
    # sqrt(n * (p^2 + n - 1))
    denominator2 = sqrt_q96x48(denominator2_inner)
    print(f"Denominator: {convert_from_Q96X48(denominator2)}")

    if denominator2 == 0:
        term2 = 0
        print("Warning: Denominator is zero, term2 set to 0")
    else:
        term2 = div_Q96X48(numerator2, denominator2)
        print(f"Term 2: {convert_from_Q96X48(term2)}")

    result = sub_Q96X48(term1, term2)
    print(f"Final k value: {convert_from_Q96X48(result)}")
    return result


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
    print(f"\nCalculating invariant:")
    print(f"A: {convert_from_Q96X48(A)}")
    print(f"B: {convert_from_Q96X48(B)}")
    print(f"R: {convert_from_Q96X48(R_int)}")
    
    A_squared = mul_Q96X48(A,A)
    B_squared = mul_Q96X48(B,B)
    R_squared = mul_Q96X48(R_int,R_int)
    
    print(f"A² + B²: {convert_from_Q96X48(add_Q96X48(A_squared, B_squared))}")
    print(f"R²: {convert_from_Q96X48(R_squared)}")
    
    invariant = sub_Q96X48(add_Q96X48(A_squared, B_squared), R_squared)
    print(f"Invariant value: {convert_from_Q96X48(invariant)}")
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

def decode_i128_from_32byte(hex_input: str) -> int:
    """
    Decode a 32-byte hex string as a signed 128-bit integer.
    
    Args:
        hex_input (str): Hex string with or without '0x' prefix (must be 64 hex chars / 32 bytes)
    
    Returns:
        int: Signed 128-bit integer
    """
    # Remove 0x prefix if present
    hex_input = hex_input.lower().replace("0x", "")
    
    if len(hex_input) != 64:
        raise ValueError("Input must be exactly 32 bytes (64 hex chars)")
    
    # Take the lower 16 bytes (last 32 hex chars)
    low16_hex = hex_input[-32:]
    
    # Convert to integer
    value = int(low16_hex, 16)
    
    # Apply 2's complement for signed 128-bit
    if value >= 2**127:
        value -= 2**128
    
    return value

def calculate_A_B_D(sum_reserves:int, sum_reserves_squared:int ,n:int, x_j:int, k_bound: int, r_int: int, s_bound: int) -> list[int, int, int]:
    print("\n--- calculate_A_B_D ---")
    print(f"sum_reserves: {convert_from_Q96X48(sum_reserves)}")
    print(f"sum_reserves_squared: {convert_from_Q96X48(sum_reserves_squared)}")
    print(f"n: {n}")
    print(f"x_j: {convert_from_Q96X48(x_j)}")
    print(f"k_bound: {convert_from_Q96X48(k_bound)}")
    print(f"r_int: {convert_from_Q96X48(r_int)}")
    print(f"s_bound: {convert_from_Q96X48(s_bound)}")

    sqrt_n = sqrt_q96x48(convert_to_Q96X48(n))
    print(f"sqrt_n: {convert_from_Q96X48(sqrt_n)}")

    S_plus_xj = add_Q96X48(sum_reserves, x_j)
    S_plus_xj_by_rootN = div_Q96X48(S_plus_xj, sqrt_n)
    print(f"S_plus_xj: {convert_from_Q96X48(S_plus_xj)}")
    print(f"S_plus_xj_by_rootN: {convert_from_Q96X48(S_plus_xj_by_rootN)}")

    r_times_sqrt_n = mul_Q96X48(r_int, sqrt_n)
    print(f"r_int * sqrt_n: {convert_from_Q96X48(r_times_sqrt_n)}")

    A = sub_Q96X48(S_plus_xj_by_rootN, add_Q96X48(k_bound, r_times_sqrt_n))
    print(f"A: {convert_from_Q96X48(A)}")

    xj_squared = mul_Q96X48(x_j, x_j)
    Q_plus_xj_squared = add_Q96X48(sum_reserves_squared, xj_squared)
    print(f"xj_squared: {convert_from_Q96X48(xj_squared)}")
    print(f"Q_plus_xj_squared: {convert_from_Q96X48(Q_plus_xj_squared)}")

    S_plus_xj_squared = mul_Q96X48(S_plus_xj, S_plus_xj)
    print(f"S_plus_xj_squared: {convert_from_Q96X48(S_plus_xj_squared)}")

    other_term = div_Q96X48(S_plus_xj_squared, convert_to_Q96X48(n))
    print(f"other_term: {convert_from_Q96X48(other_term)}")

    D = sub_Q96X48(Q_plus_xj_squared, other_term)
    print(f"D: {convert_from_Q96X48(D)}")

    sqrt_D = sqrt_q96x48(D)
    print(f"sqrt(D): {convert_from_Q96X48(sqrt_D)}")

    B = sub_Q96X48(sqrt_D, s_bound)
    print(f"B: {convert_from_Q96X48(B)}")

    print("--- End calculate_A_B_D ---\n")
    return [A, B, D]

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
    print(f"Starting solve_amount_out with initial x_j (Q96.48): {x_j}")
    initial_x_j = x_j  # Store initial value
    
    # Newton's method parameters
    max_iterations = 10
    tolerance = convert_to_Q96X48(1)/10000000000  # 1 unit tolerance in Q96.48 format
    print(f"Tolerance (Q96.48): {tolerance}")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Current x_j (Q96.48): {x_j}")
        
        # Calculate A, B, D for current guess
        [A, B, D] = calculate_A_B_D(sum_reserves, sum_reserves_squared, n, x_j, k_bound, r_int, s_bound)
        print(f"Calculated A (Q96.48): {A}")
        print(f"Calculated B (Q96.48): {B}")
        print(f"Calculated D (Q96.48): {D}")
        
        # Calculate invariant f(x_j)
        invariant_value = calculate_invariant(A, B, r_int)
        print(f"Invariant value f(x_j) (Q96.48): {invariant_value}")

        # Check if invariant is exactly zero
        if convert_from_Q96X48(invariant_value) == 0:
            print(f"Found exact solution! Invariant is zero at x_j (Q96.48): {x_j}")
            print(f"Returning final x_j (Q96.48): {x_j}")
            return x_j
        
        # Calculate derivative f'(x_j)
        derivative_value = invariant_derivative(A, B, D, n, x_j, sum_reserves)
        print(f"Derivative f'(x_j) (Q96.48): {derivative_value}")
        
        # Check for convergence (invariant close to zero)
        if abs(invariant_value) < tolerance:
            print(f"Converged due to small invariant value! Final x_j (Q96.48): {x_j}")
            return x_j
        
        # Newton's method update: x_j = x_j - f(x_j) / f'(x_j)
        # Use signed division since derivative can be negative
        if derivative_value == 0:
            print("Warning: Derivative is zero, halving step size")
            delta = div_Q96X48(x_j - initial_x_j, convert_to_Q96X48(2))
        else:
            delta = div_Q96X48_signed(invariant_value, derivative_value)
        
        print(f"Calculated delta (Q96.48): {delta}")
        
        x_j_new = sub_Q96X48(x_j, delta)
        print(f"New x_j (Q96.48): {x_j_new}")
        
        # Ensure x_j_new is positive
        if x_j_new <= 0:
            x_j_new = div_Q96X48(x_j, convert_to_Q96X48(2))  # Half the current value
            print(f"x_j_new was negative, halved to (Q96.48): {x_j_new}")
        
        # Check for convergence in x_j
        if abs(x_j_new - x_j) < tolerance:
            print(f"Converged due to small update! Final x_j (Q96.48): {x_j_new}")
            return x_j_new
            
        # Update x_j for next iteration
        x_j = x_j_new
        
    print(f"Warning: Did not converge after {max_iterations} iterations")
    print(f"Returning final x_j (Q96.48): {x_j}")
    return x_j
    
def decompose_reserves(reserves: list[int], n: int) -> tuple[int, list[int]]:
    """
    Decompose reserves into components parallel and orthogonal to equal price vector.
    Returns (alpha, w) where:
    - alpha is projection onto equal price vector
    - w is component orthogonal to equal price vector
    
    Args:
        reserves: List of reserves in Q96.48 format
        n: Number of tokens
    Returns:
        Tuple of (alpha, w) where alpha and w are in Q96.48 format
    """
    # Calculate alpha = (1/n)∑x_i
    alpha = div_Q96X48(sum(reserves), convert_to_Q96X48(n))
    print(f"Alpha: {convert_from_Q96X48(alpha)}")
    # Calculate w = x - alpha*1 where 1 is vector of ones
    w = [sub_Q96X48(x, alpha) for x in reserves]
    print(f"Orthogonal components w: {[convert_from_Q96X48(x) for x in w]}")
    return alpha, w

def is_on_tick_boundary(reserves: list[int], n: int, k: int, tolerance: int = None) -> bool:
    """
    Check if reserves are on a tick boundary defined by k.
    
    Args:
        reserves: List of reserves in Q96.48 format
        n: Number of tokens
        k: Tick boundary parameter in Q96.48 format
        tolerance: Optional tolerance in Q96.48 format
    Returns:
        True if reserves are on boundary
    """
    if tolerance is None:
        tolerance = convert_to_Q96X48(1) // (1 << 20) 
        
    alpha, _ = decompose_reserves(reserves, n)
    print(f"Checking tick boundary: alpha={convert_from_Q96X48(alpha)}, k={convert_from_Q96X48(k)}, tolerance={convert_from_Q96X48(tolerance)}")
    return abs(sub_Q96X48(alpha, k)) < tolerance

def find_crossover_trade(
    reserves: list[int],
    n: int,
    i: int,
    j: int,
    d_i: int,
    k_target: int,
    r: int
) -> tuple[int, int]:
    """
    Find trade amounts that take reserves to crossover point.
    
    Args:
        reserves: Current reserves in Q96.48 format
        n: Number of tokens
        i: Index of input token
        j: Index of output token
        d_i: Total amount of token i to trade in (Q96.48)
        k_target: Target k value at crossover (Q96.48)
        r: AMM radius (Q96.48)
    Returns:
        Tuple of (d_i, d_j) trade amounts in Q96.48 format
    """
    print(f"\nFinding crossover trade:")
    print(f"Current reserves: {[convert_from_Q96X48(x) for x in reserves]}")
    print(f"Target k value: {convert_from_Q96X48(k_target)}")
    print(f"Max input amount: {convert_from_Q96X48(d_i)}")
    
    # Get current projection
    alpha, w = decompose_reserves(reserves, n)
    print(f"Current alpha: {convert_from_Q96X48(alpha)}")
    print(f"Orthogonal components: {[convert_from_Q96X48(x) for x in w]}")
    
    # Using the crossover formula from paper:
    # d_j = n(alpha - k_target) + d_i
    
    alpha_diff = sub_Q96X48(alpha, k_target)
    print(f"Alpha difference (α - k_target): {convert_from_Q96X48(alpha_diff)}")
    
    n_scaled = convert_to_Q96X48(n)
    d_j_base = mul_Q96X48(n_scaled, alpha_diff)
    d_j = add_Q96X48(d_j_base, d_i)
    
    print(f"Calculated crossover trade:")
    print(f"Input amount (d_i): {convert_from_Q96X48(d_i)}")
    print(f"Output amount (d_j): {convert_from_Q96X48(d_j)}")
    
    # Verify the trade by checking resulting reserves satisfy invariant
    new_reserves = reserves.copy()
    new_reserves[i] = add_Q96X48(new_reserves[i], d_i)
    new_reserves[j] = sub_Q96X48(new_reserves[j], d_j)
    print(f"New reserves after trade: {[convert_from_Q96X48(x) for x in new_reserves]}")
    
    return d_i, d_j

def segment_trade(
    reserves: list[int],
    tick_boundaries: list[int], 
    n: int,
    i: int,
    j: int,
    d_i: int,
    r: int
) -> list[tuple[int, int]]:
    """
    Segment a trade that may cross tick boundaries.
    
    Args:
        reserves: Current reserves in Q96.48 format
        tick_boundaries: List of k values defining tick boundaries in Q96.48
        n: Number of tokens
        i: Index of input token
        j: Index of output token
        d_i: Amount of token i to trade in (Q96.48)
        r: AMM radius (Q96.48)
    Returns:
        List of (d_i, d_j) trade segments in Q96.48
    """
    print(f"\nStarting trade segmentation:")
    print(f"Initial reserves: {[convert_from_Q96X48(x) for x in reserves]}")
    print(f"Trading token {i} for token {j}")
    print(f"Input amount: {convert_from_Q96X48(d_i)}")
    print(f"Tick boundaries: {[convert_from_Q96X48(k) for k in tick_boundaries]}")
    
    segments = []
    remaining_i = d_i
    current_reserves = reserves.copy()
    segment_count = 0
    
    while remaining_i > 0:
        segment_count += 1
        print(f"\nProcessing segment {segment_count}:")
        print(f"Remaining input: {convert_from_Q96X48(remaining_i)}")
        
        # Find which tick boundaries are active
        on_boundaries = [
            k for k in tick_boundaries 
            if is_on_tick_boundary(current_reserves, n, k)
        ]
        
        if on_boundaries:
            print(f"Current position is on boundaries: {[convert_from_Q96X48(k) for k in on_boundaries]}")
        
        if not on_boundaries:
            print("No active boundaries - executing remaining trade")
            d_i_seg = remaining_i
            print(f"Segment input amount: {convert_from_Q96X48(d_i_seg)}")
            # Use solve_amount_out to compute output amount
            sum_reserves = sub_Q96X48(sum(current_reserves), current_reserves[j])
            sum_squares = sub_Q96X48(sum(mul_Q96X48(x, x) for x in current_reserves), mul_Q96X48(current_reserves[j], current_reserves[j]))
            print(f"Sum reserves: ", convert_from_Q96X48(sum_reserves))
            print(f"Sum squares: ", convert_from_Q96X48(sum_squares))
            new_xj = solve_amount_out(
                sum_reserves,
                sum_squares,
                n,
                0,  # k_bound
                r,
                0,  # s_bound
                sub_Q96X48(current_reserves[i], d_i_seg)
            )
            
            d_j_seg = sub_Q96X48(current_reserves[j], new_xj)
            print(f"Final segment - Input: {convert_from_Q96X48(d_i_seg)}, Output: {convert_from_Q96X48(d_j_seg)}")
            segments.append((d_i_seg, d_j_seg))
            break
            
        # Find next boundary crossing
        alpha, _ = decompose_reserves(current_reserves, n)
        k_next = min(k for k in tick_boundaries if k > alpha)
        print(f"Current alpha: {convert_from_Q96X48(alpha)}")
        print(f"Next boundary: {convert_from_Q96X48(k_next)}")
        
        # Find trade to crossing point
        d_i_seg, d_j_seg = find_crossover_trade(
            current_reserves,
            n,
            i,
            j,
            remaining_i,
            k_next,
            r
        )
        
        print(f"Crossing segment - Input: {convert_from_Q96X48(d_i_seg)}, Output: {convert_from_Q96X48(d_j_seg)}")
        segments.append((d_i_seg, d_j_seg))
        
        # Update for next iteration
        remaining_i = sub_Q96X48(remaining_i, d_i_seg)
        current_reserves[i] = add_Q96X48(current_reserves[i], d_i_seg)
        current_reserves[j] = sub_Q96X48(current_reserves[j], d_j_seg)
        print(f"Updated reserves: {[convert_from_Q96X48(x) for x in current_reserves]}")
        
    print(f"\nTrade segmentation complete: {len(segments)} segments")
    return segments

if __name__ == "__main__":
    print("\n=== Orbital AMM Testing Suite ===\n")

    # Test Case 1: Basic mathematical functions with equal reserves
    print("\n=== Test Case 1: Basic Functions with Equal Reserves ===")
    n = 5
    equal_reserves = [1000*SCALE] * 5
    print(f"\nTesting with equal reserves: {[convert_from_Q96X48(x) for x in equal_reserves]}")
    
    # Test radius calculation
    r_equal = calculate_radius(equal_reserves)
    
    # Test k calculation
    p = convert_to_Q96X48(1)  # price ratio of 1
    k_equal = calculate_k(p, n, r_equal)

    # Test Case 2: Imbalanced Reserves
    print("\n=== Test Case 2: Imbalanced Reserves ===")
    imbalanced_reserves = [1000*SCALE, 1200*SCALE, 800*SCALE, 1500*SCALE, 900*SCALE]
    print(f"\nTesting with imbalanced reserves: {[convert_from_Q96X48(x) for x in imbalanced_reserves]}")
    
    r_imbalanced = calculate_radius(imbalanced_reserves)
    k_imbalanced = calculate_k(convert_to_Q96X48(2), n, r_imbalanced)  # test with price ratio 2

    # Test Case 3: Trade Segmentation with Multiple Boundaries
    print("\n=== Test Case 3: Trade Segmentation ===")
    initial_reserves = [1000*SCALE] * 5
    r = calculate_radius(initial_reserves)
    
    # Define tick boundaries (95%, 97%, 99% of original price)
    boundaries = [
        mul_Q96X48(convert_to_Q96X48(950), r) // 1000,
        mul_Q96X48(convert_to_Q96X48(970), r) // 1000,
        mul_Q96X48(convert_to_Q96X48(990), r) // 1000
    ]
    
    print("\nTesting large trade crossing multiple boundaries")
    d_i = convert_to_Q96X48(200)  # Trade 200 of token 0
    segments = segment_trade(initial_reserves, boundaries, n, 0, 1, d_i, r)
    
    print("\nTrade segments summary:")
    total_in = 0
    total_out = 0
    for i, (d_i, d_j) in enumerate(segments):
        total_in += convert_from_Q96X48(d_i)
        total_out += convert_from_Q96X48(d_j)
        print(f"Segment {i+1}:")
        print(f"  Input: {convert_from_Q96X48(d_i)}")
        print(f"  Output: {convert_from_Q96X48(d_j)}")
    print(f"\nTotal trade summary:")
    print(f"Total input: {total_in}")
    print(f"Total output: {total_out}")

    # # Test Case 4: solve_amount_out with various scenarios
    # print("\n=== Test Case 4: Amount Out Calculation ===")
    
    # # Scenario 1: Small trade
    # small_reserves = [1000*SCALE] * 5
    # sum_reserves = sum(small_reserves)
    # sum_squares = sum(mul_Q96X48(x, x) for x in small_reserves) >> Q
    
    # print("\nTesting small trade (5 tokens)")
    # small_amount_out = solve_amount_out(
    #     sum_reserves,
    #     sum_squares,
    #     n,
    #     0,  # k_bound
    #     r,
    #     0,  # s_bound
    #     995*SCALE
    # )
    # print(f"Amount out for small trade: {convert_from_Q96X48(small_amount_out)}")

    # # Scenario 2: Large trade
    # print("\nTesting large trade (100 tokens)")
    # large_amount_out = solve_amount_out(
    #     sum_reserves,
    #     sum_squares,
    #     n,
    #     0,
    #     r,
    #     0,
    #     900*SCALE
    # )
    # print(f"Amount out for large trade: {convert_from_Q96X48(large_amount_out)}")

    # # Test Case 5: Edge Cases
    # print("\n=== Test Case 5: Edge Cases ===")
    
    # # Edge case 1: Very imbalanced reserves
    # print("\nTesting very imbalanced reserves")
    # edge_reserves = [10000*SCALE, 100*SCALE, 100*SCALE, 100*SCALE, 100*SCALE]
    # r_edge = calculate_radius(edge_reserves)
    # k_edge = calculate_k(convert_to_Q96X48(10), n, r_edge)  # test with extreme price ratio
    
    # # Edge case 2: Trade with zero input
    # print("\nTesting trade with zero input")
    # zero_segments = segment_trade(initial_reserves, boundaries, n, 0, 1, 0, r)
    
    # print("\nTest suite completed!")



