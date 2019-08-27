import numpy as np
import matplotlib.pyplot as plt
import sys, os
import ast2000tools.constants as const

np.random.seed(1444000)

# Class for engine
class Engine():
	def __init__(self, N_part, N_engines, Temp, Length, dt, ts):
		self.N = N_part
		self.Ne = Ne
		self.T = Temp
		self.L = Length
		self.l = Length / 2

		self.dt = dt
		self.ts = ts

		self.sigma = np.sqrt(k_B * Temp / m)

	def build_rocket(self):

		self.R = np.random.uniform(0, self.L, (self.N, 3)) - self.l #Array containing position. 2dim, [particle][dimension]
		self.V = np.random.normal(0, self.sigma, (self.N, 3)) #array containing velocity. Like R

		#make variables for calcualated properties

	def ignite(self):

		for t in range(self.ts):
			self.R += self.V * self.dt #Updating position of particles. 
			#No internal forces; acceleration in 0

			outside_box = np.where(abs(self.R) > self.l)

			self.V[outside_box] *= -1
			

	def performance(self):
		pass

# Setting constants
k_B = const.k_B #boltzmann constant
m = const.m_H2 #particle mass

# Variables
T = 3000 #temperature in K
L = 1e-6 #lengt of box in m
N = 1e5 #number of praticles

Ne = 1 #numer of engineboxes

dt = 1e-9
ts = 1000

sigma = np.sqrt(k_B * T / m)

inp = lambda : [N, Ne, T, L, dt, ts]

