"""
Program for importere ast og lage grunnlag for de fleste program

All kode er egenskrevet
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

def read_first_time():
	lambdas = np.zeros(int(1e+7))
	rel_flux = np.zeros_like(lambdas)
	sigmas = np.zeros_like(lambdas)


	with open("spectrum_seed17_600nm_3000nm.txt", "r") as infile:
		i = 0
		for line in infile:
			l, f = line.split()
			lambdas[i] = float(l.strip())
			rel_flux[i] = float(f.strip())
			i += 1
	with open("sigma_noise.txt", "r") as infile:
		i = 0
		for line in infile:
			l, s = line.split()
			sigmas[i] = float(s.strip())
			i += 1

	np.save("lambdas.npy", lambdas)
	np.save("rel_flux.npy", rel_flux)
	np.save("sigmas.npy", sigmas)
	sys.exit()

def load_data():
	return np.load("lambdas.npy"), np.load("rel_flux.npy"), np.load("sigmas.npy")

class Chi_square:
	def __init__(self, data, resolution):
		self.lambdas, self.fluxes, self.sigmas = data
		self.resolution = resolution # number of points to divvy interval into
		self.data_points = len(self.lambdas)

	def f(self, L, Fmin, S, L0):
		return 1 + (Fmin - 1) * np.exp(- (L - L0) ** 2 / (2 * S ** 2))

	def residual(self, m, betas):
		Fmin = betas[:, :, :, 0]
		S = betas[:, :, :, 1]
		L0 = betas[:, :, :, 2]
		return ((self.fluxes[m] - self.f(self.lambdas[m], Fmin, S, L0)) / self.sigmas[m]) ** 2


	def find_best(self, Fred, Fopp, Sned, Sopp, Lied, Lopp):
		Fmin = np.linspace(Fred, Fopp, self.resolution)
		S = np.linspace(Sned, Sopp, self.resolution)
		L0 = np.linspace(Lied, Lopp, self.resolution)


		permutations = np.transpose(np.asarray(np.meshgrid(Fmin, S, L0)), (1, 2, 3, 0))
		# creates 3d-meshgrid and inverts it such that it is a n*n*n cube
		# where every element is one set of (Fmin, S, L0) which varies spatially

		R = np.zeros(([self.resolution] * 3))
		# 3D residual array

		for m in range(self.data_points):
			R += self.residual(m, permutations)
			# Add to R by applying all permuts to each data_point, not by applying all datapoints to each permut.

		idx = np.unravel_index(np.argmin(R, axis = None), R.shape)
		# finds index of min(R), so we can find the permut that gave least residual
		return permutations[idx]



class Atmosphere:
	def __init__(self, names, masses, lambda_zeros, data):
		self.Ns = np.asarray(names)
		self.N = len(names)
		self.Ms = np.asarray(masses)
		self.LZs = np.asarray(lambda_zeros)

		self.data = np.asarray(data)

	def Sigma(self, L, mass):
		tmin = 100
		tmax = 600

		S = lambda T: L /  const.c * np.sqrt(const.k_B * T / mass)
		return S(tmin), S(tmax)

	def Temp(self, L, S, m):
		return m * (S * const.c / L) ** 2 / const.k_B

	def velocity(self, L, L0):
		return const.c * (L - L0) / L0

	def analyse_data(self, res=50, mxD=3e+4):
		self.est_Fmins = np.zeros(self.N)
		self.est_sigms = np.zeros(self.N)
		self.est_lam0s = np.zeros(self.N)

		self.est_Ts = np.zeros(self.N)

		for mass, lz, i in zip(self.Ms, self.LZs, np.arange(self.N)):
			print(i)
			Dmax = lz / mxD
			L_min = lz - Dmax
			L_max = lz + Dmax

			start = np.argmin(abs(self.data[0] - L_min))
			end = np.argmin(abs(self.data[0] - L_max))

			data_intervalled = self.data[:, start : end]
			molecyl = Chi_square(data_intervalled, res)
			analysis = molecyl.find_best(0.7, 1, *self.Sigma(lz, mass * const.m_p), L_min, L_max)

			self.est_Fmins[i], self.est_sigms[i], self.est_lam0s[i] = analysis

		self.est_Ts = self.Temp(self.est_lam0s, self.est_sigms, self.Ms * const.m_p)
		self.est_vel = self.velocity(self.est_lam0s, self.LZs)

		print(self.LZs - self.est_lam0s)
		print(self.est_Fmins)
		print(self.est_sigms)
		print(self.est_Ts)
		print(self.est_vel)

	def plot(self):
		pass




if __name__ == "__main__":
	seed = 76117

	if sys.argv[-1] == "read_data":
		read_first_time()
	# else:
		# lambdas, rel_flux, sigmas = load_data()

	names = ["O2_1", "O2_2", "O2_3", "H20_1", "H2O_2", "H2O_3", "CO2_1", "CO2_2", "CH4_1", "CH4_2", "CO", "N2O"]
	masses = [32, 32, 32, 18, 18, 18, 44, 44, 16, 16, 28, 44]
	lambda_zeros = [632, 690, 760, 720, 820, 940, 1400, 1600, 1660, 2200, 2340, 2870]

	Atmos = Atmosphere(names, masses, lambda_zeros, load_data())
	Atmos.analyse_data()
	Atmos.plot()

