import numpy as np

# =====================================================================
# 1. ROOT FINDING ALGORITHMS WITH CONVERGENCE TRACKING
# =====================================================================

def bisection_method(func, a, b, tol=1e-5, max_iter=100):
    if func(a) * func(b) >= 0:
        raise ValueError("Root is not bracketed by the chosen interval [a, b].")
    
    for i in range(1, max_iter + 1):
        midpoint = (a + b) / 2.0
        f_mid = func(midpoint)
        
        if abs(f_mid) < tol or (b - a) / 2.0 < tol:
            return midpoint, i, abs(f_mid), "Converged"
        
        if f_mid * func(a) < 0:
            b = midpoint
        else:
            a = midpoint
            
    return midpoint, max_iter, abs(func(midpoint)), "Max Iterations Reached"


def newton_raphson_method(func, deriv_func, x0, tol=1e-5, max_iter=100):
    x = x0
    for i in range(1, max_iter + 1):
        fx = func(x)
        dfx = deriv_func(x)
        
        if abs(dfx) < 1e-12:
            return x, i, abs(fx), "Failed: Derivative near zero"
            
        x_next = x - (fx / dfx)
        
        if abs(func(x_next)) < tol or abs(x_next - x) < tol:
            return x_next, i, abs(func(x_next)), "Converged"
        x = x_next
        
    return x, max_iter, abs(func(x)), "Max Iterations Reached"


def secant_method(func, x0, x1, tol=1e-5, max_iter=100):
    for i in range(1, max_iter + 1):
        fx0 = func(x0)
        fx1 = func(x1)
        
        if abs(fx1 - fx0) < 1e-12:
            return x1, i, abs(fx1), "Failed: Division by zero"
            
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        
        if abs(func(x2)) < tol or abs(x2 - x1) < tol:
            return x2, i, abs(func(x2)), "Converged"
            
        x0, x1 = x1, x2
        
    return x1, max_iter, abs(func(x1)), "Max Iterations Reached"


# =====================================================================
# 2. NUMERICAL DIFFERENTIATION MODULE
# =====================================================================

def central_difference(y, h=1.0):
    dy = np.zeros(len(y))
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    dy[0] = (y[1] - y[0]) / h 
    dy[-1] = (y[-1] - y[-2]) / h 
    return dy


# =====================================================================
# 3. NUMERICAL INTEGRATION MODULE
# =====================================================================

def trapezoidal_rule(func_values, dx=1.0):
    return dx * (0.5 * func_values[0] + np.sum(func_values[1:-1]) + 0.5 * func_values[-1])


def simpsons_rule(func_values, dx=1.0):
    n = len(func_values) - 1
    if n % 2 != 0:
        raise ValueError("Simpson's 1/3 rule requires an even number of intervals (odd number of data points).")
    
    total = func_values[0] + func_values[-1]
    total += 4 * np.sum(func_values[1:-1:2])
    total += 2 * np.sum(func_values[2:-1:2])
    
    return (dx / 3.0) * total


# =====================================================================
# 4. LINEAR SYSTEMS ENGINE (GAUSSIAN ELIMINATION)
# =====================================================================

def gaussian_elimination(A, b):
    matrix_A = np.array(A, dtype=float)
    vector_b = np.array(b, dtype=float)
    n = len(vector_b)
    
    for i in range(n):
        pivot_row = i + np.argmax(np.abs(matrix_A[i:, i]))
        if i != pivot_row:
            matrix_A[[i, pivot_row]] = matrix_A[[pivot_row, i]]
            vector_b[[i, pivot_row]] = vector_b[[pivot_row, i]]
            
        if np.abs(matrix_A[i, i]) < 1e-12:
            raise ValueError("The system matrix is singular or near-singular.")
            
        for j in range(i + 1, n):
            factor = matrix_A[j, i] / matrix_A[i, i]
            matrix_A[j, i:] -= factor * matrix_A[i, i:]
            vector_b[j] -= factor * vector_b[i]
            
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (vector_b[i] - np.dot(matrix_A[i, i + 1:], x[i + 1:])) / matrix_A[i, i]
        
    return x