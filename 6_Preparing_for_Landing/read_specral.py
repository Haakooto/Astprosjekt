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


if __name__ == "__main__":
	seed = 76117

	lambdas = np.zeros(int(1e+7))
	rel_flux = np.zeros_like(lambdas)

	with open("spectrum_seed17_600nm_3000nm.txt", "r") as infile:
		i = 0
		for line in infile:
			l, f = line.split()
			lambdas[i] = float(l.strip())
			rel_flux[i] = float(f.strip())
			i += 1




