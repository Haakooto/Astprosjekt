import numpy as np
import ast2000tools.constants as const
import sys

class Rocket():
	def __init__(self, r0, M):
		self.stages = []
		self.mass = 0
		self.r = r0
		self.R = r0
		self.v = 0
		self.M = M
		self.time = 0
		self.dt = 0.01

	def add_stage(self, drymass, fuel_mass, engine, name):
		self.stages.append([drymass, fuel_mass, engine, name])
		self.mass += drymass + fuel_mass

	def next_stage(self, n):
		return self.stages[n]

	def launch(self):

		stage = self.stages[0]
		print(f"Stage {stages[3]}")
		engine = stages[2]
		engine.build()
		engine.ignite()
		T = engine.thrust

		mF = stages[1]
		mass_use = 0

		while not self.escaped(self.r, self.v):

			A0 = self.acceleration(self.r, self.mass, T)
			self.r += self.v * self.dt + 0.5 * A0 * self.dt ** 2
			A1 = self.acceleration(self.r, self.mass, T)
			self.v += + 0.5 * (A0 + A1) * self.dt
			mass_use += engine.consume * self.dt
			self.mass -= engine.consume * self.dt

			if mass_use >= mF:
				print("Stage over, starting next stage")
				self.statusrapport()

				self.mass -= stage[0]
				break

			if self.r < self.R:
				self.statusrapport()
				print("RUD in LAO")
				sys.exit()

			self.time += self.dt


	self.statusrapport()
	print("We escaped")

	def escaped(self, r, v):
		ke = 0.5 * v ** 2
		pe = const.G * self.M / r
		return ke > pe

	def acceleration(self, r, m, T):
		return (T - const.G * m * self.M / r ** 2) / m

	def statusrapport(self):
		print(f"Altitude: {self.r - self.R}")
		print(f"Speed: {self.v}")
		print(f"Time: {self.time}")
		print(f"mass: {self.mass}")


