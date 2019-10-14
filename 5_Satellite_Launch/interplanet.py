"""
Program for simularing av bane

All kode er egenskrevet
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
import glob
import scipy.integrate as si

sys.path.append(os.path.abspath("../1_the_Rocket_Engine"))
sys.path.append(os.path.abspath("../2_Planetary_Orbits"))
sys.path.append(os.path.abspath("../3_Habitable_zone"))
sys.path.append(os.path.abspath("../4_Onboard_Orientation_Software"))


import launch
from orbits import SolarSys
from navigate import navigate
from rocket import Rocket

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

class SolarSys(SolarSys):
	def plot_orb_for_inter_jour(self, T0, T1):
		pos = self.d_pos

		t0_idx = np.argmin(abs(self.time - T0))
		t1_idx = np.argmin(abs(self.time - T1))

		pos = pos[:, :, t0_idx : t1_idx]

		for p in range(self.number_of_planets):
			plt.plot(*pos[:, p, :])


class Rocket(Rocket):
	def begin_interplanetary_journey(self, system, mission, destination):
		self.travel_time = mission.time_after_launch
		self.pos = np.transpose([mission._position_after_launch])
		self.vel = np.copy(mission._velocity_after_launch)

		self.dest = destination
		self.k = 10

		self.Masses = np.concatenate(([system.star_mass], system.masses))


	def coast(self, time, dt=1e-5):
		def rddot(r, t):
			tidx = np.argmin(abs(Tlin - t))

			Rx = r[0] - self.ri[0, :, tidx]
			Ry = r[1] - self.ri[1, :, tidx]
			R = np.vstack((Rx, Ry))
			Rnorm = np.linalg.norm(R, axis=0)

			if Rnorm[self.dest + 1] < np.linalg.norm(r) * np.sqrt(self.Masses[self.dest + 1] / (self.k * self.Masses[0])):
				# plt.scatter(*r)
				# print("We are close")
				pass

			a = lambda x: -sum(const.G_sol * self.Masses * x / Rnorm ** 3)

			return np.asarray([a(Rx), a(Ry)])

		def diffeq(t, u):
			r = u[:2]
			drx, dry = u[2:]

			dvx, dvy = rddot(r, t)

			return np.array([drx, dry, dvx, dvy])


		# time is number of L_years to coast
		T0 = self.travel_time
		self.travel_time += system.year_convert_to(time, "E")
		T1 = self.travel_time

		nT = int((T1 - T0) / dt)

		t0_idx = np.argmin(abs(system.time - T0))
		t1_idx = np.argmin(abs(system.time - T1))

		planets_pos = np.zeros((2, len(self.Masses), (t1_idx - t0_idx)))

		planets_pos[:, 1:, :] = system.d_pos[:, :, t0_idx : t1_idx]

		T0 = round(T0, 8)
		T1 = round(T1, 8)

		N = round(nT / planets_pos.shape[-1])
		nT = N * planets_pos.shape[-1]
		Tlin = np.linspace(T0, T1, nT)

		planets_pos = np.repeat(planets_pos, N, axis = 2)
		self.ri = planets_pos

		u0 = np.concatenate((self.pos[:, -1], self.vel))

		u = si.solve_ivp(diffeq, (T0, T1), u0, method="Radau", t_eval=Tlin)
		u = u.y

		pos, vel = np.split(u, 2)

		self.pos = np.concatenate((self.pos, pos), axis=1)
		self.vel = vel[:,-1]

		plt.plot(*self.pos, "r--")

	def boost(self, dv):
		self.vel += np.asarray(dv)
		self.fuel -= self.fuel_use(dv)

		if self.fuel < 0:
			print(f"We have run out of fuel!")

	def fuel_use(self, dv):
		return 0

if __name__ == "__main__":
	seed = 76117
	path = "./../verification_data"

	mission = SpaceMission(seed, path, False, True)
	system = SolarSys(seed, path, False, True)

	years = 20
	dt_pr_yr = 1e-4

	launch_time = 1
	site = launch_time * 2 * np.pi
	site = -np.pi/2

	system.differential_orbits(years, dt_pr_yr)

	Volcano, Epstein = launch.do_launch(Rocket=Rocket, verb=False)
	launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)
	pos, vel, ang = navigate(system, mission, path, doangle=False)

	print("\nstart coastin\n")

	Volcano.begin_interplanetary_journey(system, mission, destination=1)

	Volcano.coast(0.2)
	# Volcano.boost((0.95, -0.6))
	# Volcano.coast(0.115)
	# Volcano.boost((-0.5, -0.4))
	# Volcano.coast(0.4)

	system.plot_orb_for_inter_jour(mission.time_after_launch, Volcano.travel_time)

	plt.axis("equal")
	plt.show()

