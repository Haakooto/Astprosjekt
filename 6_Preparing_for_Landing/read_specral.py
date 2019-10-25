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

if __name__ == "__main__":
	seed = 76117

	if sys.argv[-1] == "read_data":
		read_first_time()
	else:
		lambdas, rel_flux, sigmas = load_data()

	plt.plot(lambdas, rel_flux)
	plt.show()


