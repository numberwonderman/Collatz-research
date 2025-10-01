def generalized_collatz(n: int, a: int, b: int, c: int, max_iterations: int = 1000) -> list[int]:
    """
    Generates a generalized Collatz sequence starting at n.

    f(n) = n/a   if n is even
    f(n) = bn+c  if n is odd

    Args:
        n: The starting integer for the sequence.
        a: The divisor for the even step (must be > 1).
        b: The multiplier for the odd step.
        c: The additive term for the odd step.
        max_iterations: Safety limit to prevent infinite loops.

    Returns:
        A list of integers representing the sequence.
    """
    if n <= 0:
        return []
    
    sequence = [n]
    current = n
    
    for _ in range(max_iterations):
        # The standard Collatz (3n+1) converges to the 4, 2, 1 cycle.
        # This implementation simply stops when 1 is reached.
        # You'll likely want a more robust cycle detection as you noted in the plan.
        if current == 1 and a == 2 and b == 3 and c == 1:
            break
            
        if current % 2 == 0:
            # Even step: n / a
            if a == 0 or current % a != 0:
                # This could be a point to halt or raise an error for non-integer steps,
                # depending on the generalized Collatz definition you want to use.
                # For now, we'll assume valid inputs where n is divisible by a if even.
                break 
            current = current // a
        else:
            # Odd step: b*n + c
            current = b * current + c
            
        if current in sequence:
            # Cycle detected (the simple check for 'current == 1' handles the standard case)
            break 
            
        sequence.append(current)
        
    return sequence

# --- Example Usage (will be moved to tests in the next step) ---
# # result = generalized_collatz(6, 2, 3, 1)
# # print(result) # [6, 3, 10, 5, 16, 8, 4, 2, 1]
