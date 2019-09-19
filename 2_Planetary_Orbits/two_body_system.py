"""
Program for å simulere 1 2-legemesystem

All kode er egenskrevet

"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

import data_analysis as da


class SolarSys(SolarSystem):
	def __init__(self, seed, data_path=None, has_moons=True, verbose=True):
		SolarSystem.__init__(self, seed, data_path=None, has_moons=True, verbose=True)
		self.one_year = np.sqrt(self.semi_major_axes[0] ** 3 / self.star_mass)
		self.one_years = np.sqrt(self.semi_major_axes ** 3 / self.star_mass)

	def find_largest_attractor(self):
		F = (
			-const.G_sol
			* self.star_mass
			* self.masses
			* np.linalg.norm(self.initial_positions, axis=0) ** -2
		)
		return np.argmin(F)

	def two_body_system(self, yrs, dt_pr_yr):
		def A(r):
			R = r[0] - r[1]
			F = -const.G_sol * R * np.linalg.norm(R) ** -3
			return np.array([F * p_mass, -F * self.star_mass])

		T = self.one_year * yrs
		dt = self.one_year * dt_pr_yr
		n = int(T / dt)

		self.time = np.linspace(0, T, n)

		la_idx = self.find_largest_attractor()
		p_mass = self.masses[la_idx]

		Rcm = p_mass * self.initial_positions[:, la_idx] / (p_mass + self.star_mass)
		Vcm = p_mass * self.initial_velocities[:, la_idx] / (p_mass + self.star_mass)

		R = np.zeros((n, 2, 2))  # [time][object][coord]
		R[0, 0] = -Rcm  # moving sun
		R[0, 1] = self.initial_positions[:, la_idx] - Rcm  # setting planet pos

		V = np.zeros((n, 2, 2))  # [time][object][coord]
		V[0, 0] = -Vcm  #
		V[0, 1] = self.initial_velocities[:, la_idx] - Vcm  # setting planet vel

		for t in range(n - 1):

			a0 = A(R[t])
			R[t + 1] = R[t] + V[t] * dt + 0.5 * a0 * dt ** 2
			a1 = A(R[t + 1])
			V[t + 1] = V[t] + 0.5 * (a0 + a1) * dt

		self.solar_orb = R[:, 0]
		self.planet_orb = R[:, 1]
		self.vel = V
		self.la_idx = la_idx
		self.p_mass = p_mass

	def plot_two_pos(self):
		S = np.transpose(self.solar_orb)
		P = np.transpose(self.planet_orb)

		plt.plot(*S, color=np.array(self.star_color) / 255, label="Sun")
		plt.plot(*P, "c", label="Planet")
		plt.scatter([0], [0], label="Centre of mass")

		plt.grid()
		plt.axis("equal")
		plt.legend(loc=1)
		plt.show()

	def radial_vel(self, i=np.pi / 2):
		xvel = np.sin(i) * self.vel[:, 0, 0]
		noise = np.random.normal(0, 0.2 * xvel.max(), len(xvel))
		V = 1444
		self.vnoise = xvel + noise + V

		plt.plot(self.time, self.vnoise)
		plt.show()

	def energy_conserve(self):
		relvel = np.linalg.norm(self.vel[:, 0] - self.vel[:, 1], axis=1)
		relpos = np.linalg.norm(self.solar_orb - self.planet_orb, axis=1)

		mu_hat = self.star_mass * self.p_mass / (self.star_mass + self.p_mass)

		self.E = (
			0.5 * mu_hat * relvel ** 2
			- const.G_sol * (self.star_mass + self.p_mass) * mu_hat / relpos
		)
		plt.plot(self.time, self.E)
		plt.show()

	def light_curve(self):
		def section_area(h):
			return r ** 2 * np.arccos((r - h) / r) - (r - h) * np.sqrt(2 * r * h - h ** 2)

		R = util.km_to_AU(self.star_radius)
		star_area = np.pi * R ** 2
		r = util.km_to_AU(self.radii[self.la_idx])

		velo = np.linalg.norm(self.initial_velocities[:, self.la_idx])

		partial_time = 2 * r / velo
		crossing_time = 2 * R / velo - partial_time

		time_before = np.linspace(0, 5 * partial_time, 1000)
		time_enter = np.linspace(5 * partial_time, 6 * partial_time, 1000)
		time_cross = np.linspace(6 * partial_time, 6 * partial_time + crossing_time, 1000)
		time_exit = np.linspace(time_cross[-1], time_cross[-1] + partial_time, 1000)
		time_after = np.linspace(time_exit[-1], time_exit[-1] + time_before[-1], 1000)

		before = np.ones(1000)
		after = np.ones_like(before)
		enter = 1 - section_area(2 * r * np.linspace(0, 1, 1000)) / star_area
		exit = enter[::-1]
		cross = np.array([enter[-1]] * 1000)

		self.full_time = np.concatenate((time_before, time_enter, time_cross, time_exit, time_after))
		light_curve = np.concatenate((before, enter, cross, exit, after))

		self.light_curve = light_curve + np.random.normal(0, 0.2, 5000)

		plt.plot(self.full_time, self.light_curve)
		plt.show()



	def assemble_data(self):
		rvs = np.concatenate(([self.time], [self.vnoise]))
		np.save("radial_velocity_curve_single.npy", rvs)

		lc = np.concatenate(([self.full_time], [self.light_curve]))
		np.save("light_curve.npy", lc)

		info = np.array([self.radii[self.la_idx], self.p_mass, self.star_mass])
		np.save("info.npy", info)



def planet_mass(vs, P, ms, i=np.pi/2):
	return ms ** (2 / 3) * vs * (2 * np.pi * const.G_sol) ** (-1 / 3) * P ** (1 / 3) / np.sin(i)


if __name__ == "__main__":
	seed = util.get_seed("haakooto")

	system = SolarSys(seed)

	yrs = 40
	dt = 1e-1

	system.two_body_system(yrs, dt)
	# system.plot_two_pos()
	# system.energy_conserve()
	system.radial_vel(i=2*np.pi/3)
	# system.light_curve()
	# system.assemble_data()

	# np.save("star_data.npy", system.vnoise)
	# np.save("times.npy", system.time)
	times = system.time
	data = system.vnoise
	data -= np.mean(data)
	print("Finished data_gen")

	# quite good values for 2-body radial vel [0.027, 2.6, -0.25]
	init_g = [0.1, 3.14, 1] #initial guess, upper

	# data = np.load("star_data.nyp.npy")
	# times = np.load("times.npy")

	# print(planet_mass(0.01969697, 2.60606061, system.star_mass, 2*np.pi/3))
	# print(system.masses[system.la_idx])

	plt.plot(times, data, label="data")

	lesq = da.least_squares(data, times, 100)
	v, P, t0 = init_g
	v, P, t0 = lesq.find_best(0, v, 0.01, P, -t0, t0) # First try
	print(v, P, t0)
	Bs = lesq.find_best(0.5*v, 1.5*v, 0.5*P, 1.5*P, 0.5*t0, 1.5*t0) # Second try
	print(Bs)
	plt.plot(times, lesq.f(times, *Bs), label="Least squares")

	# reg = non_linear_reg(data, times)
	# Bs = reg.solve(100, *Bs)
	# plt.plot(times, reg.f(times, *Bs), label="Gauss_Newton")

	# print(Bs)
	plt.legend()
	plt.show()
