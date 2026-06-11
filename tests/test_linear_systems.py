import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from numerical_methods import gaussian_elimination

def test_gaussian_elimination_2x2():
    A = [[2.0, 1.0], [1.0, 3.0]]
    b = [5.0, 5.0]
    x = gaussian_elimination(A, b)
    assert np.allclose(x, [2.0, 1.0])

def test_gaussian_elimination_3x3():
    A = [[1.0, 1.0, 1.0], [0.0, 2.0, 5.0], [2.0, 5.0, -1.0]]
    b = [6.0, -4.0, 27.0]
    x = gaussian_elimination(A, b)
    assert np.allclose(x, [5.0, 3.0, -2.0])