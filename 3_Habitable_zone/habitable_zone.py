"""
Program for å finne den beboelige sonen i solsystemet

All kode er egenskrevet

"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

class SolarSys(SolarSystem):
    def __init__(self, seed, data_path=None, has_moons=True, verbose=True):
        SolarSystem.__init__(self, seed, data_path, has_moons, verbose)
        self.T = self.star_temperature  #surface temp of star in K
        self.R = self.star_radius*1000     #star radius in m
        self.L = const.sigma*self.T**4*self.A(self.R)   #Star's luminosity


    def A(self, r):
        return 4*np.pi*r**2

    def F(self, r):
        return self.L/self.A(r)     #flux from the star in a distance r

    def distance(self,T):
        return np.sqrt(self.L/(8*np.pi*const.sigma*T**4))  #returns distance from star in m where surf_temp = T

    def planet_temp(self):
        R_p = np.linalg.norm(self.initial_positions, axis = 0)*const.AU #initial distance of planets in m
        F_p = self.F(R_p)   #flux from the star at planet orbits
        E_p = F_p*self.A(self.radii*1000)/2   #total energy hitting planets

        surf_temp = (F_p/(2*const.sigma))**(1/4)    #temp of planets in K

        panel_area = 1
        E_lander = F_p*panel_area*0.12      #energy hitting the solar panels on each planet

        min_T = 260
        max_T = 390
        max_R = self.distance(min_T)
        min_R = self.distance(max_T)
        index = []
        for i,j in enumerate(R_p):
            if min_R < j < max_R:
                index.append(i)


seed = util.get_seed("haakooto")

# mission = SpaceMission(seed)
system = SolarSys(seed)
system.planet_temp()
