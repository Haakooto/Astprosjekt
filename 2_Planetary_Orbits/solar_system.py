import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

class SolarSystem(SolarSystem):
	def plot_orb(self):
		p = 1000
		N = self.number_of_planets

		f = np.transpose(np.array([np.linspace(0,2*np.pi,p)]*N))
		a = np.array([self.semi_major_axes]*p)
		e = np.array([self.eccentricities]*p)
		omega = np.array([self.semi_major_axis_angles]*p)
		"""
		b = a*np.sqrt(1-e**2)
		x = a*np.cos(f)*np.cos(omega)-b*np.sin(f)*np.sin(omega)
		y = a*np.cos(f)*np.sin(omega)+b*np.sin(f)*np.cos(omega)
		"""
		expr = a*(1-e**2)/(1+e*np.cos(f-omega))
		x = expr*np.cos(f)
		y = expr*np.sin(f)
		P = self.initial_positions
		plt.scatter(P[0], P[1])
		plt.plot(x,y)

		plt.axis("equal")
		#plt.show()

	def diff_orb(self):
		from ivp import ExponentialDecay as ED

		e = self.eccentricities
		a = self.semi_major_axis_angles
		r = self.initial_positions
		v = self.initial_velocities
		h = r[0]*v[1]-r[1]*v[0]

		start_angle = np.arctan(self.initial_positions[1] / self.initial_positions[0])
		start_angle = np.where(self.initial_positions[0] >= 0, start_angle, start_angle + np.pi)

		T = 2
		dt = 0.00001

		orbits = ED(a, e, h, self.semi_major_axis_angles)
		t, u = orbits.solve(start_angle, T, dt)

		R = a * (1 - e ** 2) / (1 + e * np.cos(u - self.semi_major_axis_angles))

		X = R * np.cos(u)
		Y = R * np.sin(u)

		plt.plot(X, Y)
		plt.show()
		from IPython import embed;
		embed()

	def accelerate(self, r):
		return self.m * r * (np.linalg.norm(r, axis = 0)) ** (-3)

	def simulate(self, T, dt):
		self.T = T
		self.dt = dt
		self.nt = int(T / dt) + 1

		self.time = np.linspace(0, T, self.nt)

		self.pos = np.zeros((2, self.number_of_planets, self.nt))
		self.pos[:, :, 0] = self.initial_positions

		self.m = -const.G_sol * self.star_mass

		self.vel = self.initial_velocities


		self.acc_0 = self.accelerate(self.initial_positions)

		for t in range(self.nt - 1):
			self.pos[:, :, t + 1] = self.pos[:, :, t] \
									+ self.vel * self.dt \
									+ 0.5 * self.acc_0 * self.dt ** 2

			self.acc_1 = self.accelerate(self.pos[:, :, t + 1])
			self.vel = self.vel + 0.5 * (self.acc_0 + self.acc_1) * self.dt

			self.acc_0 = self.acc_1


if __name__ == "__main__":
	seed = util.get_seed("haakooto")

	mission = SpaceMission(seed)
	system = SolarSystem(seed)

	# system.diff_orb()

	system.plot_orb()
	year = 0.093
	dt = year/1000
	system.simulate(20*year, dt)

	print("start plotting")
	X = system.pos
	for i in range(system.number_of_planets):
		plt.plot(X[0,i,:], X[1,i,:])
	#plt.scatter(X[0,:,0], X[1,:,0])
	plt.grid()
	plt.show()

	# print(X[:,:,0] == system.initial_positions)

	#system.verify_planet_positions(40*year, system.pos)
