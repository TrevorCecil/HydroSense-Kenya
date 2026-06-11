import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from numerical_methods import trapezoidal_rule, simpsons_rule

def test_trapezoidal_rule():
    # Linear function y = x from x=0 to x=3 (Area of triangle = 0.5 * 3 * 3 = 4.5)
    y_vals = np.array([0.0, 1.0, 2.0, 3.0])
    area = trapezoidal_rule(y_vals, dx=1.0)
    assert np.isclose(area, 4.5)

def test_simpsons_rule():
    # Parabolic function y = x^2 from x=0 to x=4 (Exact area = 64/3 = 21.333...)
    y_vals = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    area = simpsons_rule(y_vals, dx=1.0)
    assert np.isclose(area, 64.0/3.0)