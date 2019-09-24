"""
Program for importere ast og lage grunnlag for de fleste program

All kode er egenskrevet
"""

import numpy as np
from numpy import cos, sin
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

util.check_for_newer_version()



def thph_to_xy(theta, phi, th0, ph0):
	kappa_ = 1 + cos(th0) * cos(theta) + sin(th0) * sin(theta) * cos(phi - ph0)
	kappa = 2 / kappa_

	x = kappa * sin(theta) * sin(phi - ph0)
	y = kappa * (sin(th0) * cos(theta) - cos(th0) * sin(theta) * cos(phi - ph0))

	return x, y

def xy_to_thph(x, y, th0, ph0):
	rho = np.sqrt(x ** 2 - y ** 2)
	beta = 2 * np.arctan(rho / 2)

	theta = th0 - np.arcsin(cos(beta) * cos(th0) + y / rho * sin(beta) * sin(th0))
	phi = ph0 + np.arctan(x * sin(beta) / (rho * sin(th0) * cos(beta) - y * cos(th0) * sin(beta)))

	return theta, phi

def measure_dists(time):
	set_pos = np.array([[0.5, 0.5]])

	# planet_pos = np.concatenate((np.random.randint(-5, 4, (2, 7)) + np.random.random((2, 7)), np.zeros((2, 1))), axis=1)
	planet_pos = np.transpose([[1, 1], [0, 1], [1, 0], [-1, -1], [0, -1], [-1, 0], [0, 0]])

	distances = np.linalg.norm(set_pos.T - planet_pos, axis = 0)

	return distances, planet_pos

def position(time):
	distances = measure_dists(time)[0]
	# Xp, Yp = pos[time]
	Xp, Yp = measure_dists(time)[1]
	#distances = np.random.randint(0, 9, (8, 2)) + np.random.random((8, 2))

	grid = np.linspace(-1.5, 1.5, 100)
	X, Y = np.meshgrid(grid, grid)
	tol = 1
	x, y = np.where(abs(X ** 2 + Y ** 2 - distances[-1]) < tol)

	# for p in range(system.number_of_planets):
	for p in range(7):
		x = np.where(abs((x - Xp[p]) ** 2 - (y - Yp[p]) ** 2) < tol, x)
		print(x)

	print(x)




if __name__ == "__main__":
	# seed = util.get_seed(76117)

	position(0)
	# positions = np.concatenate((np.random.randint(-5, 4, (2, 7)) + np.random.random((2, 7)), np.zeros((2, 1)), np.transpose([[np.pi, -np.pi]])), axis=1)
	# plt.scatter(*positions)
	# plt.axis([-5, 5, -5, 5])
	# plt.show()


	# theta, phi = angular_orient(picture)
	# vx, vy = velocity(lambda 1, lambda2)
	# x, y = position(distances, time)
	# x, y = distances(time)
