import numpy as np
import matplotlib.pyplot as plt
import sys, os
import ast2000tools.constants as const

np.random.seed(1444000)

# Class for engine
class Engine():
	def __init__(self, N_part, N_engines, Temp, Length, dt, ts):
		self.N = N_part
		self.Ne =Ne
		self.T = Temp
		self.L = Length
		self.l = Length / 2

		self.dt = dt
		self.ts = ts
		self.time = dt * ts

		self.sigma = np.sqrt(k_B * Temp / mass)

		self.consume = 0
		self.mom = 0
		self.avgV = np.zeros(self.ts - 1)

		self.m = 0

	def build(self):

		self.R = np.random.uniform(0, self.L, (self.N, 3)) - self.l #Array containing position. 2dim, [particle][dimension]
		self.V = np.random.normal(0, self.sigma, (self.N, 3)) #array containing velocity. Like R

		#make variables for calcualated properties

	def ignite(self):

		for t in range(self.ts):

			outside_box = np.where(abs(self.R) > self.l)
			
			absvel = abs(self.V[outside_box])
			
			if t != 0:
				self.avgV[t - 1] = np.mean(absvel)

			self.mom += sum(absvel) * mass
			self.consume += sum(outside_box[1])

			self.V[outside_box] *= -1

			self.m = max(self.m, self.R.max())

			self.R += self.V * self.dt #Updating position of particles. 
			#No internal forces; acceleration in 0

	def performance(self):
		if self.m > self.l * 1.1:
			print("Particles left the box")
		thrust = self.mom / 24 / self.time
		consume = self.consume * mass / 24 / self.time
		exhaustv = np.mean(self.avgV)
		print("Thrust: ", thrust)
		print("mass consumed: ", consume)
		print("exhaust V: ", consume)

		dv = 12819
		mf = 1100
		dm = mf * np.exp(dv / exhaustv) - mf
		print(f"mass consumed: {dm}")


# Setting constants
k_B = const.k_B #boltzmann constant
mass = const.m_H2 #particle mass

# Variables
T = 4000 #temperature in K
L = 1e-6 #lengt of box in m
N = int(1e5) #number of praticles

Ne = 1 #numer of engineboxes

dt = 1e-12
ts = 1000

inp = lambda : [N, Ne, T, L, dt, ts]

rocket = Engine(*inp())
rocket.build()
rocket.ignite()
rocket.performance()
