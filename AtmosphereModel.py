# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:43:57 2020

@author: raulm_000
"""
import numpy as np

# Model is from NASA in the following website
# https://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html

def atmosphereModel ( altitude ):
    AtmoData = {}
    if ( altitude < 11000 ):
        AtmoData["Temp"] = 15.04 - 0.00649*altitude
        AtmoData["Pres"] = 101.29*((AtmoData["Temp"]+273.1)/288.08)**5.256
        
    elif ( altitude >= 11000 and altitude < 25000 ):
        AtmoData["Temp"] = -56.46
        AtmoData["Pres"] = 22.65*np.exp(1.73 - .000157*altitude)
    else:
        AtmoData["Temp"] = -131.24 + .00299*altitude
        AtmoData["Pres"] = 22.65*np.exp(1.73 - .000157*altitude)
        
    AtmoData["Dens"] = AtmoData["Pres"]/(0.2869*(AtmoData["T"]+273.1))
    
    return AtmoData