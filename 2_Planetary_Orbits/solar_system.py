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

		b = a*np.sqrt(1-e**2)
		x = a*np.cos(f)*np.cos(omega)-b*np.sin(f)*np.sin(omega)
		y = a*np.cos(f)*np.sin(omega)+b*np.sin(f)*np.cos(omega)

		plt.plot(x,y)

		plt.axis("equal")
		plt.show()

	def accelerate(self, r):
		G = 4*np.pi**2
		m = G*(self.star_mass+self.masses)
		x = -m*r[0]/(r[0]**2+r[1]**2)**(3/2)
		y = -m*r[1]/(r[0]**2+r[1]**2)**(3/2)
		return np.asarray([x,y])

	def simulate(self, T, dt):
		self.T = T
		self.dt = dt
		self.nt = int(T / dt) + 1

		self.time = np.linspace(0, T, self.nt)

		self.pos = np.zeros((self.nt, 2, self.number_of_planets))
		self.vel = np.zeros_like(self.pos)
		self.acc = np.zeros_like(self.pos)

		self.pos[0] = self.initial_positions
		self.vel[0] = self.initial_velocities
		self.acc[0] = self.accelerate(self.initial_positions)

		for t in range(int(T / dt)):
			self.pos[t + 1] = self.pos[t] + self.vel[t] * self.dt + 0.5 * self.acc[t] * self.dt ** 2
			self.acc[t + 1] = self.accelerate(self.pos[t+1])
			self.vel[t + 1] = self.vel[t] + 0.5 * (self.acc[t] + self.acc[t+1]) * self.dt



if __name__ == "__main__":
	seed = util.get_seed("haakooto")

	mission = SpaceMission(seed)
	system = SolarSystem(seed)

	system.plot_orb()
	# system.simulate(5, 0.00001)
	# x = np.zeros(system.nt)
	# y = np.zeros(system.nt)
	# for i in range(system.number_of_planets):
	# 	for j in range(system.nt):
	# 		x[j] = system.pos[j][0][i]
	# 		y[j] = system.pos[j][1][i]
	# 	plt.plot(x,y)
	# 	plt.axis("equal")
	# plt.show()
