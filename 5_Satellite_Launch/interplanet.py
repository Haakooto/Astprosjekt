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
	def plot_orb_for_inter_jour(self, T0, T1, a=False):
		pos = self.d_pos

		t0_idx = np.argmin(abs(self.time - T0))
		t1_idx = np.argmin(abs(self.time - T1))

		pos = pos[:, :, t0_idx : t1_idx]

		self.plot_orbits(pos, init=False, a=a)

	def hohmann_transfer(self, dest):
		r1 = self.semi_major_axes[0]
		r2 = self.semi_major_axes[dest]
		mu = self.star_mass * const.G_sol
		u = self.angles_of_all_times
		u1 = u[:, 0]
		u2 = u[:, dest]
		U = u1 - u2

		dv1 = np.sqrt(mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) -1)
		dv2 = np.sqrt(mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
		tH = np.pi * np.sqrt((r1 + r2) ** 3 / (8 * mu))
		omega2 = np.sqrt(mu / r2 ** 3)
		alpha = np.pi - omega2 * tH
		alpha2 = np.pi * (1 - 1 / (2 * np.sqrt(2)) * np.sqrt((r1 / r2) + 1) ** 3)

		print(alpha, alpha2)
		t0 = np.argmin(abs(U - alpha))
		print(t0)
		T0 = self.time[t0]

		print(f"T0: {self.year_convert_to(T0, 'L')}")
		print(f"tH: {tH}")
		print(f"dv1: {dv1}")
		print(f"dv2: {dv2}")

		return T0, tH, dv1, dv2


class Rocket(Rocket):
	def begin_interplanetary_journey(self, system, mission, destination):
		self.system = system
		self.mission = mission

		self.travel_time = mission.time_after_launch
		self.pos = np.transpose([mission._position_after_launch])
		self.vel = np.copy(mission._velocity_after_launch)

		self.dest = destination
		self.k = 10

		self.Masses = np.concatenate(([system.star_mass], system.masses))

	def commands(self, c_list):
		print("\nStarting interplanetary travel\n")
		coasts = c_list[::2]
		boosts = c_list[1::2]

		if len(coasts) != len(boosts):
			boosts.append((0,0))

		if sum(coasts) > self.system.year_convert_to(self.system.time[-1], "L"):
			print("\nCoasting is exceeding simulated time!")
			print("Terminating")
			sys.exit()

		for c, b in zip(coasts, boosts):
			boosted = self.coast(c)
			if boosted != None:
				if boosted.status:
					self.travel_time = boosted.t_events[0][0]
					print(f"We hit it at {boosted.t_events[0][0]}")
					break
			self.boost(b)


	def coast(self, time, dt=1e-5):
		print("coasting ", time)
		def rddot(r, t):
			tidx = np.argmin(abs(Tlin - t))

			Rx = r[0] - self.ri[0, :, tidx]
			Ry = r[1] - self.ri[1, :, tidx]
			R = np.vstack((Rx, Ry))
			Rnorm = np.linalg.norm(R, axis=0)

			a = lambda x: -sum(const.G_sol * self.Masses * x / Rnorm ** 3)

			return np.asarray([a(Rx), a(Ry)])

		def dominant_gravity(t, u):
			r = u[:2]
			tidx = np.argmin(abs(Tlin - t))

			Rx = r[0] - self.ri[0, :, tidx]
			Ry = r[1] - self.ri[1, :, tidx]
			R = np.vstack((Rx, Ry))
			Rnorm = np.linalg.norm(R, axis=0)

			RRRR = Rnorm[self.dest + 1] - np.linalg.norm(r) * np.sqrt(self.Masses[self.dest + 1] / (self.k * self.Masses[0]))
			return RRRR
		dominant_gravity.terminal = True

		def diffeq(t, u):
			r = u[:2]
			drx, dry = u[2:]

			dvx, dvy = rddot(r, t)

			return np.array([drx, dry, dvx, dvy])

		if time == 0:
			return None

		# time is number of L_years to coast
		T0 = self.travel_time
		self.travel_time += self.system.year_convert_to(time, "E")
		T1 = self.travel_time

		nT = int((T1 - T0) / dt)

		t0_idx = np.argmin(abs(self.system.time - T0))
		t1_idx = np.argmin(abs(self.system.time - T1))

		planets_pos = np.zeros((2, len(self.Masses), (t1_idx - t0_idx)))

		planets_pos[:, 1:, :] = self.system.d_pos[:, :, t0_idx : t1_idx]

		T0 = round(T0, 8)
		T1 = round(T1, 8)

		N = round(nT / planets_pos.shape[-1])
		nT = N * planets_pos.shape[-1]
		Tlin = np.linspace(T0, T1, nT)

		planets_pos = np.repeat(planets_pos, N, axis = 2)
		self.ri = planets_pos

		u0 = np.concatenate((self.pos[:, -1], self.vel))

		U = si.solve_ivp(diffeq, (T0, T1), u0, method="Radau", t_eval=Tlin, events=dominant_gravity)
		u = U.y

		pos, vel = np.split(u, 2)
		plt.scatter(*pos[:, -1], color="r")

		self.pos = np.concatenate((self.pos, pos), axis=1)
		self.vel = vel[:,-1]

		return U


	def boost(self, dv):
		print("boosting ", dv)
		if type(dv) is tuple:
			self.vel += np.asarray(dv)
			self.fuel -= self.fuel_use(dv)
		elif type(dv) is int:
			dv = float(dv)
		elif type(dv) is float:
			new_vel = self.vel * dv
			self.fuel -= self.fuel_use(abs(new_vel - self.vel))
			self.vel = new_vel

		if self.fuel < 0:
			print(f"We have run out of fuel!")

	def fuel_use(self, dv):
		return 0

	def plot_journey(self):
		plt.plot(*self.pos, "r--", label="Rocket path")
		plt.scatter(*self.pos[:, -1], color="r", label="Final pos rocket")


if __name__ == "__main__":
	seed = 76117
	path = "./../verification_data"

	mission = SpaceMission(seed, path, False, True)
	system = SolarSys(seed, path, False, True)

	years = 20
	dt_pr_yr = 1e-4
	destination = 1

	system.differential_orbits(years, dt_pr_yr)

	T0, tH, dv1, dv2 = system.hohmann_transfer(destination)
	# specs = [0.76, -0.4, [0.365, (0, -0.5), 0.15, (-1.75, 0.1), 0.15, (0, 0), 0.2, (0.5, 0), 0.7, 0.95, 0, (0.3, 0), 0.4 ]]
	# specs = [0.76, -0.33, [0.3, 1.1, 0.3]]
	# launch_time = specs[0]
	# site = specs[1]

	specs = [0, dv1, tH, dv2, 0.4]
	launch_time = 0.76
	site = -0.6

	Volcano, Epstein = launch.do_launch(Rocket=Rocket, verb=False)
	launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)
	pos, vel, ang = navigate(system, mission, path, doangle=False)

	Volcano.begin_interplanetary_journey(system, mission, destination=destination)

	Volcano.commands(specs)

	# Volcano.hohmann_transfer()
	# sys.exit()

	system.plot_orb_for_inter_jour(mission.time_after_launch, Volcano.travel_time, a=True)

	Volcano.plot_journey()
	plt.legend(loc=1)
	plt.axis("equal")
	plt.show()

