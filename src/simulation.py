import numpy as np

def soil_moisture_derivative(S, R, I, ET, field_capacity, drainage_coeff):
    drainage = 0.0
    if S > field_capacity:
        drainage = drainage_coeff * (S - field_capacity)
    
    dS_dt = R + I - ET - drainage
    return dS_dt

def euler_method(S0, R_arr, I_arr, ET_arr, field_capacity, drainage_coeff, days=30, h=1.0):
    steps = int(days / h)
    S = np.zeros(steps + 1)
    S[0] = S0
    
    for t in range(steps):
        R = R_arr[t] if t < len(R_arr) else 0
        I = I_arr[t] if t < len(I_arr) else 0
        ET = ET_arr[t] if t < len(ET_arr) else 0
        
        dS = soil_moisture_derivative(S[t], R, I, ET, field_capacity, drainage_coeff)
        S[t+1] = S[t] + h * dS
        
    return S

def runge_kutta_4(S0, R_arr, I_arr, ET_arr, field_capacity, drainage_coeff, days=30, h=1.0):
    steps = int(days / h)
    S = np.zeros(steps + 1)
    S[0] = S0
    
    for t in range(steps):
        R = R_arr[t] if t < len(R_arr) else 0
        I = I_arr[t] if t < len(I_arr) else 0
        ET = ET_arr[t] if t < len(ET_arr) else 0
        
        k1 = h * soil_moisture_derivative(S[t], R, I, ET, field_capacity, drainage_coeff)
        k2 = h * soil_moisture_derivative(S[t] + 0.5*k1, R, I, ET, field_capacity, drainage_coeff)
        k3 = h * soil_moisture_derivative(S[t] + 0.5*k2, R, I, ET, field_capacity, drainage_coeff)
        k4 = h * soil_moisture_derivative(S[t] + k3, R, I, ET, field_capacity, drainage_coeff)
        
        S[t+1] = S[t] + (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return S

def generate_monte_carlo_rainfall(mean_rain, std_rain, days=30, iterations=1000):
    np.random.seed(42)
    scenarios = np.random.normal(loc=mean_rain, scale=std_rain, size=(iterations, days))
    scenarios = np.maximum(0, scenarios)
    return scenarios