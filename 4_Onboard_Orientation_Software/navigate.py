"""
Program for å bestemme angulær orientasjon, hastighet og posisjon til rakett

All kode er egenskrevet
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
import glob

sys.path.append(os.path.abspath("../1_the_Rocket_Engine"))
sys.path.append(os.path.abspath("../2_Planetary_Orbits"))
sys.path.append(os.path.abspath("../3_Habitable_zone"))

import launch
from orbits import SolarSys
import angular_orientation as AO

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

def velocity(mission):
	phi1, phi2 = util.deg_to_rad(mission.star_direction_angles)
	u1hat = np.asarray([np.cos(phi1),np.sin(phi1)])
	u2hat = np.asarray([np.cos(phi2),np.sin(phi2)])

	lamda = mission.reference_wavelength
	lamda0 = np.asarray([lamda]*2)
	dlamda = np.asarray(mission.measure_star_doppler_shifts()) - np.asarray(mission.star_doppler_shifts_at_sun)
	u = const.c*dlamda/lamda0

	if phi1 > phi2:
		v = 1/np.sin(phi2-phi1)*np.asarray([np.sin(phi2)*u[0] - np.sin(phi1)*u[1], - np.cos(phi2)*u[0] + np.cos(phi1)*u[1]])
	else:
		v = 1/np.sin(phi1-phi2)*np.asarray([np.sin(phi2)*u[0] - np.sin(phi1)*u[1], - np.cos(phi2)*u[0] + np.cos(phi1)*u[1]])

	return v * const.yr / const.AU


def position(system, mission):

	time_idx = np.argmin(abs(system.time - mission.time_after_launch)) - 1
	dt = 1E-3

	r = mission.measure_distances()
	m = np.argsort(r)

	spos = np.zeros((2, system.number_of_planets + 1))
	spos[:,:system.number_of_planets] = system.d_pos[:,:,time_idx]


	N = 1000
	angle = np.linspace(0,2*np.pi,N)

	circle1, circle2, circle3 = [np.asarray([spos[:,m[i]]]*N) + np.transpose(r[m[i]]*np.asarray([np.cos(angle), np.sin(angle)])) for i in range(1,4)]
	for p in circle1:
		for q in circle2:
			if np.linalg.norm(p-q) < 1e-3:
				for s in circle3:
					if np.linalg.norm(p-s) < 1e-3:
						pos = s
						return pos


def navigate(system, mission, path):
	references = np.load(f"{path}/ang_ori_refs.npy")
	mission.take_picture(full_sky_image_path=f"{path}/himmelkule.npy")
	image = Image.open("sky_picture.png")


	angle = AO.determine_angle(image, references)
	v = velocity(mission)
	pos = position(system, mission)

	return [pos, v, angle]




if __name__ == "__main__":
	seed = 76117
	path = "./../verification_data"

	mission = SpaceMission(seed, path, False, True)
	system = SolarSys(seed, path, False, True)

	years = 30
	dt_pr_yr = 1e-4

	site = 0
	launch_time = 0

	system.differential_orbits(years, dt_pr_yr)

	Volcano, Epstein = launch.do_launch()
	launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)

	X = navigate(system, mission, path)
	mission.verify_manual_orientation(*X)

	print(f"Position after launch: {X[0]}AU")
	print(f"Velocity after launch: {X[1]}AU/yr")
	print(f"Angle after launch: {X[2]}deg")

	print(mission._position_after_launch)
	print(mission._velocity_after_launch)


