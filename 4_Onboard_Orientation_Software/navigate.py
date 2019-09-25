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
system = SolarSys(seed)

Volcano = Rocket(*rocket_build())
Epstein = Engine()


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
m = np.argsort(r)
T = Volcano.time/const.yr/dt/system.one_year
system.differential_orbits(20,dt)
spos = np.zeros((2,8))
spos[:,:7] = system.d_pos[:,:,int(T)]

N = 1000
angle = np.linspace(0,2*np.pi,N)
def plot_circles(N_circles):
	for i in range(N_circles):
		circle = np.transpose(np.asarray([spos[:,m[i]]]*N)) + r[m[i]]*np.asarray([np.cos(angle), np.sin(angle)])
		plt.plot(circle[0],circle[1])
	plt.show()

circle1, circle2, circle3 = [np.asarray([spos[:,m[i]]]*N) + np.transpose(r[m[i]]*np.asarray([np.cos(angle), np.sin(angle)])) for i in range(1,4)]
for p in circle1:
	for q in circle2:
		if np.linalg.norm(p-q) < 2.5e-4:
			for s in circle3:
				if np.linalg.norm(p-s) < 5e-4:
					print(s)
					plt.scatter(s[0], s[1])
plot_circles(3)
