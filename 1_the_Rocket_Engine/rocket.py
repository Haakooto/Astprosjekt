"""
Program for å simmulere rakettoppskytning

All kode er egenskrevet
"""

import ast2000tools.constants as const
import numpy as np
import sys, os


class Rocket:
	def __init__(self, m0, r0, M, dt, sT=0):
		self.mass = m0  # rocket drymass
		self.r = r0  # rocket position
		self.R = r0  # planet radius
		self.M = M  # planet mass
		self.dt = dt  # timestep

		self.v = 0  # initial velocity
		self.time = sT  # start time

	def assemble(self, engine, fuel_mass, N_engines):
		self.engine = engine
		self.fuel = fuel_mass
		self.Ne = N_engines

	def launch(self):

		self.launch_sequence()
		self.engine.ignite()

		self.thrust = self.engine.thrust * self.Ne
		self.burn_rate = self.engine.consume * self.Ne

		while True:

			# leapfrog integration
			a0 = self.acceleration()
			self.r += self.v * self.dt + 0.5 * a0 * self.dt ** 2
			a1 = self.acceleration()
			self.v += 0.5 * (a0 + a1) * self.dt

			self.fuel -= self.burn_rate * self.dt
			self.time += self.dt

			# 3 possible outcomes of launch:
			if self.escaped():
				self.status = ("Launch successful", 0)
				break
			elif self.fuel < 0:
				self.status = ("Burnout", 1)
				break
			elif self.r < self.R:  # if underground
				self.status = ("RUD in LAO", 1)
				# Rapid Unschedules Disassembly in Low Atmospheric Orbit
				break

		self.statusrapport()

	def escaped(self):
		ke = 0.5 * self.v ** 2
		pe = const.G * self.M / self.r
		return ke > pe

	def acceleration(self):
		return self.thrust / (self.mass + self.fuel) - const.G * self.M / self.r ** 2

	def statusrapport(self):
		print(self.status[0], "\n")
		print(f"Altitude: {self.r - self.R}")
		print(f"Speed: {self.v}")
		print(f"fuel left: {self.fuel}")
		print(f"Time: {self.time}\n")

		if self.status[1] == 1:
			sys.exit()

	def launch_sequence(self):
		pass
