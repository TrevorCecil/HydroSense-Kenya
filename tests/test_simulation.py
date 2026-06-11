import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from simulation import soil_moisture_derivative, euler_method

def test_soil_moisture_derivative_no_drainage():
    # S=30, Rain=5, Irrig=0, ET=4, FC=40. Since S < FC, drainage is 0.
    # dS/dt = 5 + 0 - 4 - 0 = 1.0
    dS = soil_moisture_derivative(S=30.0, R=5.0, I=0.0, ET=4.0, field_capacity=40.0, drainage_coeff=0.1)
    assert dS == 1.0

def test_euler_method_step():
    S0 = 30.0
    R = np.array([5.0, 5.0])
    I = np.array([0.0, 0.0])
    ET = np.array([4.0, 4.0])
    FC = 40.0
    D_coeff = 0.1
    
    # After step 1 (h=1), S should be S0 + dS = 30 + 1 = 31.0
    S_simulated = euler_method(S0, R, I, ET, FC, D_coeff, days=2, h=1.0)
    assert S_simulated[1] == 31.0