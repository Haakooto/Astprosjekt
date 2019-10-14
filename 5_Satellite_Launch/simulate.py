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

print("\n"*3, "File is depricated, stop use!!", "\n"*3)

class SolarSys(SolarSys):
	def coast(self, pos, vel, T0, T1):
		def rddot(r, t):
			Rx = r[0] - ri[0, :, t]
			Ry = r[1] - ri[1, :, t]
			R = np.vstack((Rx, Ry))
			Rnorm = np.linalg.norm(R, axis=0)

			l_idx = np.argmin(Rnorm)
			l = Rnorm[l_idx]
			try:
				if l < np.linalg.norm(r) * np.sqrt(M[l_idx + 1] / (10 * M[0])):
					print("\n"*4)
					print(f"unstable orbit reached")
					print("\n"*4)
			except:
				print("\n"*3)
				print("Something something")
				print("\n"*3)

			a = lambda x: - sum(const.G_sol * M * x / Rnorm ** 3)

			A = np.asarray([a(Rx), a(Ry)])
			return A

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
			# plt.scatter(*ri[:, p + 1, -1])
		plt.plot(*fr)
		# plt.axis("equal")
		# plt.show()

		return fr[:, -1], fv


if __name__ == "__main__":
	seed = 76117
	path = "./../verification_data"

	mission = SpaceMission(seed, path, False, True)
	system = SolarSys(seed, path, False, True)

	years = 20
	dt_pr_yr = 1e-4

	launch_time = 0.76
	site = launch_time * 2 * np.pi

	system.differential_orbits(years, dt_pr_yr)

	Volcano, Epstein = launch.do_launch(verb=False)
	launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)
	print("star coastin")
	pos, vel, ang = navigate(system, mission, path, doangle=False)
	plt.scatter(*pos)

	T00 = mission.time_after_launch
	# command = [[T00, (0, 0)], [T00 + 0.3, (1, 0)], [T00 + 1.4, (1, -1)], [T00 + 1.5, (0,0)]]
	command = [[T00, (0.95,-0.6)], [T00+0.115, (-1,-0.4)], [T00+0.13, (0,0)]]
	prev_t = T00
	for t, dv in command:

		pos, vel = system.coast(pos, vel, prev_t , t)
		prev_t = t
		vel += np.asarray(dv)
		print(vel)

	plt.axis("equal")
	plt.show()

