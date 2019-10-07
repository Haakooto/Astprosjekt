"""
Program for simularing av bane

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
sys.path.append(os.path.abspath("../4_Onboard_Orientation_Software"))


import launch
from orbits import SolarSys
from navigate import navigate

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

class SolarSys(SolarSys):
	def simulated_trajectory(self, pos, vel, T0, T1):
		def rddot(r, t):
			Rx = r[0] - ri[0, :, t]
			Ry = r[1] - ri[1, :, t]
			R = np.vstack((Rx, Ry))
			Rnorm = np.linalg.norm(R, axis=0)

			a = lambda x: - sum(const.G_sol * M * x / Rnorm ** 3)

			A = np.asarray([a(Rx), a(Ry)])
			# print(A)
			return A
			# a = - G * Ms * r / np.linalg.norm(r)**3 - np.sum(G * Mp * (r - rp) / np.linalg.norm(r - rp) **3 )

		# input time in Laconia years
		T1 = T0 + self.year_convert_to(T1, "E")

		T0_idx = np.argmin(abs(self.time - T0))
		T1_idx = np.argmin(abs(self.time - T1))

		T0 = self.time[T0_idx]
		T1 = self.time[T1_idx]

		flight_time = self.time[T0_idx : T1_idx + 1]
		dt = flight_time[1] - flight_time[0]

		ndt = T1_idx - T0_idx + 1
		index_time = np.arange(0, len(flight_time))
		# print(len(flight_time))

		M = np.concatenate(([self.star_mass], self.masses))
		ri = np.zeros((2, len(M), ndt))

		ri[:, 1:] = self.d_pos[:, :, T0_idx : T1_idx + 1]

		fr = np.zeros((2, len(flight_time)))
		fv = vel

		fr[:, 0] = pos


		for t in index_time[:-1]:
			a0 = rddot(fr[:, t], t)
			fr[:, t + 1] = fr[:, t] + fv * dt + 0.5 * a0 * dt ** 2
			a1 = rddot(fr[:, t + 1], t + 1)
			fv += 0.5 * (a0 + a1) * dt


		for p in range(self.number_of_planets):
			plt.plot(*ri[:, p + 1, :])
			plt.scatter(*ri[:, p + 1, -1])
		plt.plot(*fr)
		plt.axis("equal")
		plt.show()

		return fr


if __name__ == "__main__":
	seed = 76117
	path = "./../verification_data"

	mission = SpaceMission(seed, path, False, True)
	system = SolarSys(seed, path, False, True)

	years = 20
	dt_pr_yr = 1e-4

	launch_time = 2.1415
	site = np.pi/2

	system.differential_orbits(years, dt_pr_yr)

	Volcano, Epstein = launch.do_launch()
	launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)

	pos, vel, ang = navigate(system, mission, path, doangle=False)
	plt.scatter(*pos)

	# print(pos)
	pos = system.simulated_trajectory(pos, vel, mission.time_after_launch, 3.718)
	# plt.plot(*pos)
	# system.plot_orbits(d=True)

	# print(mission._position_after_launch)
	# print(mission._velocity_after_launch)

