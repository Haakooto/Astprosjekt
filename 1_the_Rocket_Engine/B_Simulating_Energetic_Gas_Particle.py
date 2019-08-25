import sys, os
import numpy as np

class Engine():
	def __init__(self, N_particles, N_engines, Size, Temperature, dt, time_steps):
		self.N = N_particles #number of particles in one engine
		self.engines = N_engines #number of engines side by side
		self.S = Size #side length of one cube engine, (0,0,0) is in centre of box
		self.T = Temperature #temperature inside engine

		self.dt = dt #length of timesteps
		self.ts = time_steps #number of timesteps for sim

		self.time = np.linspace(0, int(self.dt * self.ts), self.ts) #array with all times

		self.R = np.zeros((self.ts, self.N, 3)) #array with particle_positions for every timestep. (timestep, particle, dimension)
		self.V = np.zeros_like(self.R) #array with velocities, simmilar to self.R

		self.R[0] = np.random.uniform(0, self.S, (self.N, 3)) - self.S / 2 #place particles centred uniformly in first timestep  in 3 dimensions
		self.V[0] = np.random.uniform(-10,10,(self.N,3)) #temp init velo dist
		#self.V[0] = function for maxwell_boltzman distrubution

	def simulate(self):

		for i in range(self.ts - 1): #for every timestep
			self.Euler_Cromer(i)

			#implement nozzle here

			self.V[i + 1] = np.where(abs(self.R[i + 1]) < self.S / 2, self.V[i], - self.V[i]) #if particle at wall, flip speed

			#calculate stuff here
			
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

N = 1000
ne = 1
L = 6
T = 3000
dt = 0.01
t = 100

inp = lambda : [N, ne, L, T, dt, t]

System = Engine(*inp())
System.simulate()
System.write("test.xyz")


# #array = np.zeros((N,3))
# array = np.random.uniform(-l, l, (1,N,3))
# array[0][0] = [-l, -l, -l]
# array[0][-1] = [l, l, l]

# #print(array)
# write(array, "test.xyz")
