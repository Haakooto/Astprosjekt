import numpy as np
from Engine import Engine
from Rocket import Rocket
import sys, os
import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt

np.random.seed(14441111)
seed = util.get_seed("haakooto")

def escape_veolocity(r, v, m):
	ke = 0.5 * v ** 2
	pe = - G * M / r
	return ke > pe

mission = SpaceMission(seed)
system = SolarSystem(seed)

G = const.G
M = system.masses[0]*const.m_sun
R = system.radii[0]*1000

craft_mass = mission.spacecraft_mass
craft_area = mission.spacecraft_area

# Variables for rocketengine
T = 10000 #temperature in K
L = 1e-6 #lengt of box in m
N = int(1e5) #number of particles

area = np.array([16, 38, 40])
Ne = area / L ** 2 #numer of engineboxes
nozzle = 1

dt = 1e-12
ts = 1000

inp = lambda n: [N, Ne[n], nozzle, T, L, dt, ts]

F1 = 3000
F2 = 40000
F3 = 150000

rocket = Rocket(R, M)
engine1 = Engine(*inp(0))
engine2 = Engine(*inp(1))
engine3 = Engine(*inp(2))
rocket.add_stage(11000, F3, engine3, "terrum")
rocket.add_stage(4000, F2, engine2, "venum")
rocket.add_stage(1100, F1, engine1, "mercum")

rocket.launch()
print(rocket.statusrapport())

