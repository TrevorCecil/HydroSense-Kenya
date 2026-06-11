import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from numerical_methods import bisection_method, newton_raphson_method, secant_method

def test_bisection_method():
    def func(x): return x**2 - 4.0
    root, _, _, status = bisection_method(func, 0, 5, tol=1e-5)
    assert np.isclose(root, 2.0, atol=1e-4)
    assert status == "Converged"

def test_newton_raphson_method():
    def func(x): return x**2 - 4.0
    def dfunc(x): return 2.0 * x
    root, _, _, status = newton_raphson_method(func, dfunc, x0=5, tol=1e-5)
    assert np.isclose(root, 2.0, atol=1e-4)
    assert status == "Converged"

def test_secant_method():
    def func(x): return x**2 - 4.0
    root, _, _, status = secant_method(func, x0=0, x1=5, tol=1e-5)
    assert np.isclose(root, 2.0, atol=1e-4)
    assert status == "Converged"