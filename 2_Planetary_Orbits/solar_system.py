import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

class SolarSystem(SolarSystem):
	def plot_orb(self):
		f = np.linspace(0,2*np.pi,1000)
		for i in range(self.number_of_planets):
			b = self.semi_major_axes[i]*np.sqrt(1-self.eccentricities[i]**2)
			x = self.semi_major_axes[i]*np.cos(f)*np.cos(self.semi_major_axis_angles[i]-b*np.sin(f)*np.sin(self.semi_major_axis_angles[i]))
			y = self.semi_major_axes[i]*np.cos(f)*np.sin(self.semi_major_axis_angles[i]+b*np.sin(f)*np.cos(self.semi_major_axis_angles[i]))
			plt.plot(x,y)

		plt.show()

	def simulate(self, T, dt):
		self.T = T
		self.dt = dt
		self.nt = int(T / dt)

		self.time = np.linspace(0, T, self.nt)

		#print(self.initial_positions.shape)
		self.pos = np.zeros((self.nt, 2, self.number_of_planets))
		self.vel = np.zeros_like(self.pos)
		self.acc = np.zeros_like(self.pos)

		self.pos[0] = self.initial_positions
		self.vel[0] = self.initial_velocities
		self.acc[0] = self.accelerate()

		for t in range(int(T / dt)):
			self.pos[t + 1] = self.pos[t] + self.vel[t] * self.dt + 0.5 * self.acc[t] * self.dt ** 2
			self.acc[t + 1] = self.accelerate(t)
			self.vel[t + 1] = self.vel[t] + 0.5 * (self.acc[t] + self.acc[t + 1]) * self.dt

	def accelerate(self, t):
		pass

if __name__ == "__main__":
	seed = util.get_seed("haakooto")

	mission = SpaceMission(seed)
	system = SolarSystem(seed)

	system.plot_orb()
	
	#system.simulate(1, 0.01)
