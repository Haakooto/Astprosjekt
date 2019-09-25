"""
Program for importere ast og lage grunnlag for de fleste program

All kode er egenskrevet
"""
import sys, os
sys.path.append(os.path.abspath("../1_the_Rocket_Engine"))
sys.path.append(os.path.abspath("../2_Planetary_Orbits"))
sys.path.append(os.path.abspath("../3_Habitable_zone"))

from launch import *
from orbits import SolarSys
import numpy as np
import matplotlib.pyplot as plt


import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem


util.check_for_newer_version()

seed = util.get_seed("haakooto")
system = SolarSystem(seed)

Volcano = Rocket(*rocket_build())
Epstein = Engine()

orbits = SolarSys(seed)

Epstein.build(*engine_build())
Volcano.assemble(*asseble(Epstein))

Volcano.launch()

mission = verify(Volcano, Epstein)


phi1, phi2 = util.deg_to_rad(mission.star_direction_angles)
u1hat = np.asarray([np.cos(phi1),np.sin(phi1)])
u2hat = np.asarray([np.cos(phi2),np.sin(phi2)])

lamda = mission.reference_wavelength
lamda0 = np.asarray([lamda]*2)
dlamda = np.asarray(mission.measure_star_doppler_shifts()) - np.asarray(mission.star_doppler_shifts_at_sun)
u = const.c*dlamda/lamda0

if phi1 > phi2:
	v = 1/np.sin(phi1-phi2)*np.asarray([np.sin(phi2)*u[0] - np.sin(phi1)*u[1], - np.cos(phi2)*u[0] + np.cos(phi1)*u[1]])
else:
	v = 1/np.sin(phi2-phi1)*np.asarray([np.sin(phi2)*u[0] - np.sin(phi1)*u[1], - np.cos(phi2)*u[0] + np.cos(phi1)*u[1]])

dt = 1E-3

r = mission.measure_distances()
m = np.argsort(r)[:3]
T = Volcano.time/const.yr
#orbits.long_run(20,dt)
