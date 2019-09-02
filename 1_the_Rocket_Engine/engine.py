import numpy as np
import matplotlib.pyplot as plt
import sys, os
import ast2000tools.constants as const
import distributions as dists

np.random.seed(14441111)

# Class for engine
class Engine():
	def build(self, N_part, nozzle_size, Temp, Length, dt, ts):
		# Names should be self explanatory
		self.N = N_part
		self.T = Temp
		self.L = Length
		self.l = Length / 2
		self.nozzle = nozzle_size

		self.dt = dt
		self.ts = ts
		self.time = dt * ts

		# To improve efficiency we use numpys normal distribution instead of own defined
		# Maxwell Boltzman. These are the same for each component when sigma in normal is as given under
		self.sigma = np.sqrt(k_B * Temp / mass)

		self.R = np.random.uniform(0, self.L, (self.N, 3)) - self.l #Array containing position. 2dim, [particle][dimension]
		self.V = np.random.normal(0, self.sigma, (self.N, 3)) #array containing velocities. Like R

		self.consume = 0 # number of particles that left nozzle
		self.mom = 0 # momentum of particles that left nozzle, (in z-direction)
		self.avgV = np.zeros(self.ts - 1) # avg exhaust velocity

		self.m = 0 #maximum particle distance from centre of box.

	def ignite(self):

		for t in range(self.ts):

			outside_box = np.where(abs(self.R) > self.l)
			#particles outside box, [particle][coord outside]

			absvel = abs(self.V[outside_box])
			#abs to account for which side of box

			if t != 0: #at t = 0 all are inside
				self.avgV[t - 1] = np.mean(absvel)
				#average v of particles that leave box in coordinate
				#exhaust velocity

			self.mom += sum(absvel) * mass #momentum P = sum(p_i) = sum(v_i)*m
			self.consume += len(outside_box[0]) #number of particles that left box

			self.m = max(self.m, self.R.max())

			# After calculation, move particles
			self.V[outside_box] *= -1 # Bounce off wall
			self.R += self.V * self.dt #Updating position of particles.
			#No internal forces; acceleration in 0
			# Basically Euler, but since no acceleration no energy is added

		self.thrust = self.mom * self.nozzle / 6 / self.time
		self.consume = self.consume * mass * self.nozzle / 6 / self.time

	def test(self): #Some tests to validate simulation
		if self.m > self.l * 1.1:
			#if timestep is too high, particles will go far outside
			#This tests if particles always are relatively close to the box
			print("Particles left the box, timestep is too high")
		else:
			print("Timestep is good")

		#plot histogram of vx, vy ,vz and V
		for x, y in enumerate(["x", "y", "z"]):
			max_bolz = dists.P_mb(round(self.V[:,x].min()), round(self.V[:,x].max()), False, self.T, mass)
			plt.plot(max_bolz[0], max_bolz[1], label = "analytical velocity distribution")
			hist = plt.hist(self.V[:,x], bins = "auto", density = True, label = "system velocity distribution")

			plt.xlabel(f"velocity in {y} direction")
			plt.ylabel(f"number of particles in bin")
			plt.title(f"Histogram of velocities in {y} direction")
			plt.legend(loc = 1)
			plt.show()

		V = np.linalg.norm(self.V, axis = 1)
		plt.hist(V, bins = "auto", density = True, label = "system velocity distribution")
		max_bolz = dists.P_mb(0, V.max(), True, self.T, mass)
		plt.plot(max_bolz[0], max_bolz[1], label = "analytical velocity distribution")

		plt.xlabel("velocity")
		plt.ylabel("number of particles with velocity")
		plt.title("Histogram of velocities")
		plt.legend(loc = 1)
		plt.show()

	def performance(self, drymass, dv):
		self.exhaustv = np.mean(self.avgV)
		self.dm = drymass * np.exp(dv / self.exhaustv) - drymass


# Setting constants
k_B = const.k_B #boltzmann constant
mass = const.m_H2 #particle mass

if __name__ == "__main__":

	# Variables
	T = 3000 #temperature in K
	L = 1e-6 #lengt of box in m
	N = int(1e5) #number of praticles

	nozzle_size = 0.4 #area of nozzle in L**2

	dt = 1e-12
	ts = 1000

	inp = [N, nozzle_size, T, L, dt, ts]

	rocket = Engine(*inp)
	rocket.build()
	rocket.ignite()
	rocket.test()
	rocket.performance(1100, 12819)
	print("Thrust [N]: ", rocket.thrust)
	print("mass consumed [kg/s]: ", rocket.consume)
	print("exhaust V [m/s]: ", rocket.exhaustv)
	print(f"mass consumed to reach escape velocity: {rocket.dm}")