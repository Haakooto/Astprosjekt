import sys, os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1444000)

class Engine():
	def __init__(self, N_particles, N_engines, Size, Temperature, dt, time_steps):
		self.m = 3.347E-27 #mass of H2
		self.k_B = 1.38E-23
		self.N = N_particles #number of particles in one engine
		self.engines = N_engines #number of engines side by side
		self.S = Size #side length of one cube engine, (0,0,0) is in centre of box
		self.T = Temperature #temperature inside engine

		self.dt = dt #length of timesteps
		self.ts = time_steps #number of timesteps for sim
		self.time = np.linspace(0, self.ts-1, self.ts) #array with all times
		# print(int(self.dt * self.ts))

		self.R = np.zeros((self.ts, self.N, 3)) #array with particle_positions for every timestep. (timestep, particle, dimension)
		self.V = np.zeros_like(self.R) #array with velocities, simmilar to self.R

		self.R[0] = np.random.uniform(0, self.S, (self.N, 3)) - self.S / 2 #place particles centred uniformly in first timestep  in 3 dimensions
		self.V[0] = np.random.normal(0, np.sqrt(self.k_B * self.T / self.m), (self.N, 3))

		self.count = np.zeros(self.ts)
		self.mom = 0

		self.ke = np.zeros(self.ts-1)
		self.avgV = np.sqrt(8 * self.k_B * self.T / np.pi / self.m)
		print("avgV", self.avgV)

		#self.SS = np.zeros(self.ts-1)
	def in_nozzle(self, A, r = np.sqrt(0.25/np.pi), h = 0.01):
		r = self.S * r
		h = self.S * h
		#determines particle inside sylindrical nozzle
		circle = A[:,0] ** 2 + A[:,1] ** 2 <= r ** 2
		height = A[:,2] <= h - self.S / 2

		return np.logical_and(circle, height)

	def simulate(self):

		#p = np.sqrt(8 * self.k_B * self.T / np.pi)
		#print(p)
		#nps = self.N * self.k_B * np.sqrt(3) * self.T / (8 * p * self.S) * self.ts * self.dt
		#print(nps)
		sump = self.N * self.k_B * self.T * self.ts * self.dt * 0.125 / self.S
		print(sump)

		for i in range(1, self.ts): #for every timestep
			self.Euler_Cromer(i - 1)

			#for p in self.V[i]:
			#	self.SS[i-1] += np.linalg.norm(p)
			#self.SS[i-1] /= self.N
			V = 0
			for p in self.V[i]:
				V += np.linalg.norm(p)

			self.ke[i-1] = V ** 2 * (3 * self.N)**(-1)
			#implement nozzle here
			ps = self.in_nozzle(self.R[i])
			#self.count[i] = sum(ps)
			#for p in self.V[i][ps]:
			#print(np.linalg.norm(self.V[i][ps]) )
			self.R[i][ps] = [*np.random.uniform(-self.S/2, self.S/2, (2)), self.S / 2] #move particle to top
			self.V[i][ps] = np.random.normal(0, np.sqrt(self.k_B * self.T / self.m), (3))
			#self.V[i][ps] = function for maxwell_boltzmann dist

			self.V[i] = np.where(abs(self.R[i]) < self.S / 2, self.V[i], -self.V[i]) #if particle at wall, flip speed

			#calculate stuff here
		self.mom *= self.m		

	def Euler_Cromer(self, k):
		self.V[k + 1] = self.V[k]
		self.R[k + 1] = self.R[k] + self.V[k + 1] * self.dt

	def write(self, name): #writes xyz-file for visualization in Ovito
			here = os.path.dirname(os.path.realpath(__file__))
			filepath = os.path.join(here, name) #creates file in specified directory
				
			outfile = open(filepath, "w") 

			for i in range(self.ts): #for every timestep
				outfile.write(str(self.N) + "\n") #write number of particles
				outfile.write("coord x y z \n") #non_read line
				for p in range(self.N): #for every particle
					for x in range(3): #for every coord
						outfile.write(f"{str(round(self.R[i][p][x], 4))} ")
					outfile.write("\n")
			outfile.close()

N = 100000
ne = 1
L = 1e-6
T = 3000
dt = 1e-9
t = 1000

inp = lambda : [N, ne, L, T, dt, t]

System = Engine(*inp())
System.simulate()
#System.write("test.xyz")
# print(sum(System.count))
#print(System.mom)
#print(System.SS.min())
# #array = np.zeros((N,3))
# array = np.random.uniform(-l, l, (1,N,3))
# array[0][0] = [-l, -l, -l]
# array[0][-1] = [l, l, l]
plt.plot(System.time[:-1], System.ke)
plt.show()

# #print(array)
# write(array, "test.xyz")
