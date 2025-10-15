def generalized_collatz(n: int, a: int, b: int, c: int, max_iterations: int = 1000000) -> list[int]:
    """
    Generates a generalized Collatz sequence using the parity rule (n/2 if even).
    For the odd step, it uses (b*n + c), applying the maximal division by 2 
    (shortcut) only for generalized parameters, but NOT for the standard (3n+1).
    """
    if n <= 0:
        return []
    
    sequence = [n]
    current = n
    
    for _ in range(max_iterations):
        
        # 1. Termination Check: Stop after 1 has been calculated and appended.
        if current == 1:
            break
            
        # 2. Magnitude Safety Check
        if current > 10**50: 
            break 
            
        if current % 2 == 0:
            # EVEN STEP: Standard Collatz is always division by 2
            current = current // 2
        else:
            # ODD STEP: 
            numerator = b * current + c
            
            if a == 2 and b == 3 and c == 1:
                # STANDARD COLLATZ (3n+1): No maximal shortcut.
                current = numerator
            else:
                # GENERALIZED COLLATZ: Use the shortcut (maximal division by 2)
                while numerator % 2 == 0 and numerator > 0:
                    numerator = numerator // 2
                current = numerator

        # 3. Append and Check for New Cycle (must happen after the update)
        if current in sequence:
            # Cycle detected (a new, non-standard cycle)
            if current != sequence[-1]:
                sequence.append(current)
            break 
            
        sequence.append(current)
        
    return sequence