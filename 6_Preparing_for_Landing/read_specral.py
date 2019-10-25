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
		self.data_points = len(self.lambdas)
		self.resolution = resolution

	def f(self, L, Fmin, S, L0):
		return 1 + (Fmin - 1) * np.exp(- (L - L0) ** 2 / (2 * S ** 2))

	def residual(self, m, Bs):

		Fmin = Bs[:, :, :, 0]
		S = Bs[:, :, :, 1]
		L0 = Bs[:, :, :, 2]
		return ((self.fluxes[m] - self.f(self.lambdas[m], Fmin, S, L0)) / self.sigmas[m]) ** 2


	def find_best(self, Fred, Fopp, Sned, Sopp, Lied, Lopp):
		Fmin = np.linspace(Fred, Fopp, self.resolution)
		S = np.linspace(Sned, Sopp, self.resolution)
		L0 = np.linspace(Lied, Lopp, self.resolution)


		permutations = np.transpose(np.asarray(np.meshgrid(Fmin, S, L0)), (1, 2, 3, 0))

		R = np.zeros(([self.resolution] * 3))

		for m in range(self.data_points):
			R += self.residual(m, permutations)

		idx = np.unravel_index(np.argmin(R, axis = None), R.shape)
		return permutations[idx]


def Sigma(L, mass):
	tmin = 150
	tmax = 450

	S = lambda T: L /  const.c * np.sqrt(const.k_B * T / mass)

	return S(tmin), S(tmax)


if __name__ == "__main__":
	seed = 76117

	if sys.argv[-1] == "read_data":
		read_first_time()
	else:
		lambdas, rel_flux, sigmas = load_data()

	plt.plot(lambdas, rel_flux, "b")

	names = ["O2_1", "O2_2", "O2_3", "H20_1", "H2O_2", "H2O_3", "CO2_1", "CO2_2", "CH4_1", "CH4_2", "CO", "N2O"]
	masses = [32, 32, 32, 18, 18, 18, 44, 44, 16, 16, 28, 44]
	lambda_zeros = [632, 690, 760, 720, 820, 940, 1400, 1600, 1660, 2200, 2340, 2870]

	for name, mass, lambda_zero in zip(names, masses, lambda_zeros):
		dmax = lambda_zero / 3e+4
		lambda_min = lambda_zero - dmax
		lambda_max = lambda_zero + dmax

		start = np.argmin(abs(lambdas - lambda_min))
		end = np.argmin(abs(lambdas - lambda_max))

		data = [lambdas[start : end], rel_flux[start : end], sigmas[start : end]]

		molecyl = Chi_square(data, 50)

		analysis = molecyl.find_best(0.7, 1, *Sigma(lambda_zero, mass * const.m_p), lambda_min, lambda_max)

		L = lambdas[start : end]
		plt.plot(L, molecyl.f(L, *analysis), "k")

	plt.show()

