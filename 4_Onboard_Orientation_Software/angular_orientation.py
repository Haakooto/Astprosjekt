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
import glob

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

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

def take_360_pictures():
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

def take_random_pic():
	height = 480
	width = 640

	alph_th = util.deg_to_rad(70)
	alph_phi = util.deg_to_rad(70)

	x = np.linspace(*C_maxmin(alph_phi), width)
	y = np.linspace(*C_maxmin(alph_th), height)

	X, Y = np.meshgrid(-x, y)

	phi = np.random.random() * 2 * np.pi
	angles = xy_to_thph(X, Y, ph0 = phi)
	image = generate_png(angles)

	return image, phi

def generate_unique_id(image):
	unique_val = np.zeros(12)

	h, w, c = image.shape
	w2 = int(w/2)
	h2 = int(h/2)

	c1 = image[:h2, :w2]
	c2 = image[:h2, w2:]
	c3 = image[h2:, :w2]
	c4 = image[h2:, w2:]

	unique_val[:3] = np.sum(c1, axis=(0, 1))
	unique_val[3:6] = np.sum(c2, axis=(0, 1))
	unique_val[6:9] = np.sum(c3, axis=(0, 1))
	unique_val[9:] = np.sum(c4, axis=(0, 1))

	return unique_val

def generate_set_ids():
	files = glob.glob("pngs/*.png")
	files = np.array([file.rsplit(".", 1) for file in files])
	files = np.array([name[0].rsplit("_", 1) for name in files])
	files = sorted(files, key = lambda x: int(x[1]))
	files = np.array([file[0] + "_" + file[1] + ".png" for file in files])

	unique_vals = np.zeros((len(files), 12))

	for i, file in enumerate(files):
		img = Image.open(file)
		pixels = np.array(img)

		unique_vals[i] = generate_unique_id(pixels)

	return unique_vals

def determine_angle(image, reference_set):
	pixs = np.array(image)
	val = generate_unique_id(pixs)

	residual = np.zeros(len(reference_set))

	for i, pic in enumerate(reference_set):
		residual[i] = sum((val - pic) ** 2)

	return np.argmin(residual)



if __name__ == "__main__":
	seed = util.get_seed(76117)

	himmelkule = np.load("himmelkule.npy")

	mission = SpaceMission(seed)


	# take_360_pictures()
	# np.save("reference.npy", generate_set_ids())

	reference_set = np.load("reference.npy")
	N = 100
	fail = 0

	for _ in range(N):
		img, answer = take_random_pic()
		result = determine_angle(generate_unique_id(img), reference_set)

		succ = abs(util.deg_to_rad(result) - answer) < np.pi/360

		if not succ:
			fail += 1

	print(f"Number of fails in {N} attempts: {fail}")




