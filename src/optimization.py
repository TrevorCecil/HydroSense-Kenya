import numpy as np
from simulation import runge_kutta_4

def optimize_irrigation_schedule(S0, R_arr, ET_arr, min_threshold, target_threshold, field_capacity, drainage_coeff, days=30):
    I_schedule = np.zeros(days)
    S_simulated = np.zeros(days + 1)
    S_simulated[0] = S0
    
    for t in range(days):
        S_current = S_simulated[t]
        R = R_arr[t]
        ET = ET_arr[t]
        
        S_projected = S_current + R - ET
        
        if S_projected < min_threshold:
            required_irrigation = target_threshold - S_projected
            I_schedule[t] = required_irrigation
            S_projected += required_irrigation
            
        drainage = 0.0
        if S_projected > field_capacity:
            drainage = drainage_coeff * (S_projected - field_capacity)
            
        S_simulated[t+1] = S_projected - drainage
        
    return I_schedule, S_simulated
