# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:35:19 2020

@author: raulm_000
"""

import numpy as np
from AtmosphereModel import atmosphereModel

##############################################################################
#                           Funciton Definitions                             #
##############################################################################


def Derivatives ( time, state, Planet ):
    
    ders = np.zeros(state.size,dtype='float64')
    ders[0:3] = gravityAccel(state,Planet)
    print(ders)
    
    return ders

def gravityAccel ( state, Planet ):
    position = state[0:3]
    R = np.linalg.norm(position)
    
    g_accel = (position/R)*(Planet["GM"]/R**2)
    
    return g_accel



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def rk45 ( time, state, Planet, time_step):
    # Butcher Tableu for 4th and 5th order
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2],dtype='float64')
    A = np.array([[0, 0, 0, 0, 0, 0],
                  [1/4, 0, 0, 0, 0, 0],
                  [3/32, 9/32, 0, 0, 0, 0],
                  [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                  [439/216, -8, 3680/513, -845/4104, 0, 0],
                  [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]],dtype='float64')
    b1 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0],dtype='float64')
    b2 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],dtype='float64')
    
    k = np.zeros((c.size,state.size),dtype='float64')
    state_out_4 = state
    state_out_5 = state
    for i in range(c.size):
        if ( i == 0 ):
            k[i] = Derivatives(time,state, Planet)
        else:
            state_sum = state
            k_sum = np.zeros(state.size,dtype='float64')
            
            for j in range(i-1):
                k_sum += A[i][j]*k[j]
            
            state_sum = state + k_sum*time_step
            
            k[i] = Derivatives(time + c[i]*time_step,state_sum,Planet)
            
            state_out_4 = state_out_4 + time_step*b1[i]*k[i]
            state_out_5 += time_step*b2[i]*k[i]
    
    error = state_out_5 - state_out_4
    print(error)
    
    # ToDo: Use error to determine new time step
    time_step_out = time_step
    time_out = time + time_step
        
    return time_out, state_out_4, time_step_out

##############################################################################
#                           Main Function                                    #
##############################################################################

# Define Planet Data

Planet = {}
Planet["Radius"] = 6371800 # m
Planet["Rotation"] = np.array([0,0,1/(24*60*60)])
Planet["GM"] = 3.986004418*10**14

# Initial Conditions

state0 = np.array([100, # X
                   0, # Y
                   0, # Z
                   0, # X_dot
                   0, # Y_dot
                   0, # Z_dot
                   10],dtype='float64') # Mass

# Add radius of the Earth

state0[0] += Planet["Radius"]

time_step = 0.1
time0 = 0

(time_out, state_out, h_out) = rk45(0,state0,Planet,time_step)