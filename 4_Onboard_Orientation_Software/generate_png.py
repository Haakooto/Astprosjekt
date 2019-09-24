"""
Program for importere ast og lage grunnlag for de fleste program

All kode er egenskrevet
"""

import numpy as np
from numpy import sin, cos
import sys, os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

# util.check_for_newer_version()

def thph_to_xy(theta, phi, th0=np.pi/2, ph0=0):
	kappa_ = 1 + cos(th0) * cos(theta) + sin(th0) * sin(theta) * cos(phi - ph0)
	kappa = 2 / kappa_

	x = kappa * sin(theta) * sin(phi - ph0)
	y = kappa * (sin(th0) * cos(theta) - cos(th0) * sin(theta) * cos(phi - ph0))

	return x, y

def xy_to_thph(x, y, th0=np.pi/2, ph0=0):
	rho = np.sqrt(x ** 2 + y ** 2)
	beta = 2 * np.arctan(rho / 2)

	theta = th0 - np.arcsin(cos(beta) * cos(th0) + y / rho * sin(beta) * sin(th0))
	phi = ph0 + np.arctan(x * sin(beta) / (rho * sin(th0) * cos(beta) - y * cos(th0) * sin(beta)))

	return np.asarray([theta, phi])

def C_maxmin(a):
	v = 2 * sin(a / 2) / (1 + cos(a / 2))
	return [v, -v]

def generate_png(angles):
	ang = np.reshape(angles, (2, len(angles[0]) * len(angles[0][0])))

	RGB = np.zeros((len(ang[0]), 3), dtype=int)

	for i  in range(len(ang[0])):
		theta = ang[0][i]
		phi = ang[1][i]
		RGB[i] = himmelkule[mission.get_sky_image_pixel(theta, phi)][2:]


	RGB = np.reshape(RGB, (*angles.shape[1:], 3))
	return RGB


if __name__ == "__main__":
	seed = util.get_seed(76117)

	himmelkule = np.load("himmelkule.npy")

	mission = SpaceMission(seed)

	img = Image.open("sample0000.png")
	pixels = np.array(img)
	height, width, colors = pixels.shape

	alph_th = util.deg_to_rad(70)
	alph_phi = util.deg_to_rad(70)

	x = np.linspace(*C_maxmin(alph_phi), width)
	y = np.linspace(*C_maxmin(alph_th), height)

	X, Y = np.meshgrid(-x, y)

	for phii in range(360):
		phiii = util.deg_to_rad(phii)

		angles = xy_to_thph(X, Y, ph0 = phiii)
		pixels_new = generate_png(angles)

		new_img = Image.fromarray(pixels_new.astype("uint8"))
		new_img.save(f"pngs/th_piover2_phi_{phii}.png")


