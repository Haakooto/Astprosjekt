import numpy as np
from Engine import Engine
import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

seed = util.get_seed("haakooto")

mission = SpaceMission(seed)
system = SolarSystem(seed)

escape_v = np.sqrt(2 * const.G * system.masses[0] / system.radii[0])

mass = mission.spacecraft_mass
area = mission.spacecraft_area

# Variables for rocketengine
T = 3000 #temperature in K
L = 1e-6 #lengt of box in m
N = int(1e5) #number of particles

Ne = area / L ** 2 #numer of engineboxes

dt = 1e-12
ts = 1000

inp = [N, Ne, T, L, dt, ts]

rocket = Engine(*inp)
rocket.build()
rocket.ignite()
rocket.performance(mass, escape_v)

print(rocket.thrust)

