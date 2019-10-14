"""
Program for å finne og plotte planetposisjoner numerisk

All kode er egenskrevet

fil ivp.py er nødvendig i samme mappe for en av metodene
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
		SolarSystem.__init__(self, seed, data_path=None, has_moons=True, verbose=True)
		self.one_year = np.sqrt(self.semi_major_axes[0] ** 3 / self.star_mass)
		self.spin = (
			self.initial_positions[0] * self.initial_velocities[1]
			- self.initial_positions[1] * self.initial_velocities[0]
		)

		self.ordered_planets = np.argsort(
			np.argsort(np.linalg.norm(self.initial_positions, axis=0))
		)
		# self.d_pos = np.zeros((2, 1, 1))

	def year_convert_to(self, T, planet="L"):
		# converts to the specified planet, if L: from earth to Laconia, if E other way
		if planet == "L":
			return T / self.one_year
		elif planet == "E":
			return T * self.one_year
		else:
			raise AttributeError("unknown planet used in year conversion")

	def analytical_orbits(self):
		p = 10000
		N = self.number_of_planets

		f = np.transpose(np.array([np.linspace(0, 2 * np.pi, p)] * N))
		a = np.array([self.semi_major_axes] * p)
		e = np.array([self.eccentricities] * p)
		omega = np.array([self.aphelion_angles] * p) + np.pi

		R = a * (1 - e ** 2) / (1 + e * np.cos(f - omega))
		x = R * np.cos(f)
		y = R * np.sin(f)

		self.a_pos = np.array([x, y])

	def differential_orbits(self, yrs, dt_pr_yr, newpos=False):
		from ivp import Diff_eq as Deq

		if newpos:
			r0 = self.r0
		else:
			r0 = self.initial_positions

		T = self.year_convert_to(yrs, "E") # yrs is in Laconia years, T is earth years
		dt = self.one_year * dt_pr_yr
		self.time = np.linspace(0, T, int(T / dt)+1)

		if T != 0:
			e = self.eccentricities
			a = self.semi_major_axes
			omega = self.aphelion_angles + np.pi
			h = self.spin

			start_angle = np.arctan(r0[1] / r0[0])
			start_angle = np.where(r0[0] >= 0, start_angle, start_angle + np.pi)

			orbits = Deq(a, e, h, omega)
			t, u = orbits.solve(start_angle, T, dt)

			R = a * (1 - e ** 2) / (1 + e * np.cos(u - omega))

			x = R * np.cos(u)
			y = R * np.sin(u)

			self.d_pos = np.transpose([x, y], (0, 2, 1))
			self.angles_of_all_times = u

		else:
			self.time = np.zeros((1), dtype=int)
			self.d_pos = np.array([self.initial_positions.T])

	def iterated_orbits(self, yrs, dt_pr_yr):
		def accelerate(r):
			return self.constant * r * (np.linalg.norm(r, axis=0)) ** (-3)

		T = self.one_year * yrs
		dt = self.one_year * dt_pr_yr
		nt = int(T / dt)

		self.time = np.linspace(0, T, nt)

		pos = np.zeros((2, self.number_of_planets, nt))
		pos[:, :, 0] = self.initial_positions

		self.constant = -const.G_sol * self.star_mass

		vel = self.initial_velocities

		acc_0 = accelerate(pos[:, :, 0])

		for t in range(nt - 1):
			pos[:, :, t + 1] = pos[:, :, t] + vel * dt + 0.5 * acc_0 * dt ** 2

			acc_1 = accelerate(pos[:, :, t + 1])
			vel = vel + 0.5 * (acc_0 + acc_1) * dt

			acc_0 = acc_1

		self.i_pos = pos

	def plot_orbits(self, array=1, a=False, init=True, final=True):
		if type(array) is int:
			array = self.d_pos

		shap = array.shape
		if shap[0] != 2 or shap[1] not in [7, 8, 9] or len(shap) != 3:
			print("Youve done wrong!")
			return 0

		ordered_planets = np.argsort(np.linalg.norm(self.initial_positions, axis=0))
		planet_names = [
			"Vulcan",
			"Laconia",
			"Vogsphere",
			"Ilus",
			"Alderaan",
			"Apetos",
			"Auberon",
			"Zarkon",
			"Tellusia",
			"X",
		]

		if a:
			try:
				self.a_pos
			except:
				self.analytical_orbits()
			plt.plot(*self.a_pos, "y")
			plt.plot(*self.a_pos[:, 0, 0], "y", label="Analytical orbits")

		for p in range(self.number_of_planets):
			lab = f"{planet_names[self.ordered_planets[p]]}"#; nr. {self.ordered_planets[p]}"
			plt.plot(*array[:, p, :], label=f"{lab}")

		if init:
			plt.scatter(*array[:, :, 0], label="start")
		if final:
			plt.scatter(*array[:, :, -1], label="final")

		plt.scatter([0], [0], s=80, color=np.array(self.star_color) / 255, label="Sun")

		plt.grid()
		plt.axis("equal")
		plt.xlabel("x, [AU]", fontsize=20)
		plt.ylabel("y, [AU]", fontsize=20)
		plt.legend(loc=1, fontsize=15)

		if __name__ == "__main__":
			plt.show()

	def animate_orbits(self, inn, ut):
		from matplotlib.animation import FuncAnimation, FFMpegWriter

		fig = plt.figure()
		self.ani_pos = self.d_pos[:, inn:ut+1, :]

		# Configure figure
		plt.axis("equal")
		plt.axis("off")
		xmax = 2 * np.max(abs(self.ani_pos))
		plt.axis((-xmax, xmax, -xmax, xmax))

		# Make an "empty" plot object to be updated throughout the animation
		self.positions, = plt.plot([], [], "o", lw=1)
		# Call FuncAnimation
		self.animation = FuncAnimation(
			fig,
			self._next_frame,
			frames=range(len(self.time)),
			repeat=None,
			interval=1,  # 000 * self.dt,
			blit=True,
			save_count=100,
		)

		plt.show()

	def _next_frame(self, i):
		self.positions.set_data((0, *self.ani_pos[0, :, i]), (0, *self.ani_pos[1, :, i]))
		self.positions.set_label(("p1", "p2", "p3"))

		return (self.positions,)


if __name__ == "__main__":
	# seed = util.get_seed("haakooto")
	seed = 76117
	path = "./../verification_data"

	system = SolarSys(seed, path, False, True)
	# system = SolarSys(18116)
	# system = SolarSys(seed)

	years = 30
	dt = 1e-4

	# print(np.linalg.norm(system.initial_positions, axis=0))

	# system.long_run(years, dt)

	# system.analytical_orbits()
	# system.iterated_orbits(years, dt)
	# system.load_pos(f"pos_{years}yr.npy")

	# system.pos = np.load("pos_100yr.npy")
	# system.pos = system.pos[:,:,::2]

	# system.d_pos = np.load(f"./npys/pos_{years}yr.npy")
	# system.t = np.linspace(0, years * system.one_year, len(system.d_pos[0][0]))

	# time, pos = np.load("planet_trajectories.npy", allow_pickle=True)

	# for p in range(7):
	# 	plt.plot(*pos[:, p, :])

	# import time

	# timer = time.time()
	system.differential_orbits(years, dt)
	# t1 = time.time() - timer
	# print(f"time {years}: {t1}")
	# timer = time.time()

	system.plot_orbits(system.d_pos)
	# system.animate_orbits(0, 7)
	# np.save(f"npys/pos_{years}yr", system.d_pos)

	# system.verify_planet_positions(years * system.one_year, system.d_pos, f"{path}/planet_trajectories_{years}yr.npy")
	# print(f"their time: {(time.time() - timer)}")
	# system.generate_orbit_video(system.t, system.d_pos)
