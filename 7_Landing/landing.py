"""
Program for simulering av landing

All kode er egenskrevet
"""

import numpy as np
import sys, os
import time as tim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath("../1_the_Rocket_Engine"))
sys.path.append(os.path.abspath("../2_Planetary_Orbits"))
sys.path.append(os.path.abspath("../3_Habitable_zone"))
sys.path.append(os.path.abspath("../4_Onboard_Orientation_Software"))
sys.path.append(os.path.abspath("../5_Satellite_Launch"))
sys.path.append(os.path.abspath("../6_Preparing_for_Landing"))

import launch
from orbits import SolarSys
from navigate import navigate
from journey import Rocket
from atmosphere import density

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
from ast2000tools.shortcuts import SpaceMissionShortcuts as SMS



class Landing:
	def __init__(self, r0, v0, system, mission):
		self.r = r0.reshape((1, 3)).T
		self.v = v0
		self.t = 0

		self.sys = system
		self.mis = mission

		self.R = self.sys.radii[self.mis.dest] * 1000
		self.M = self.sys.masses[self.mis.dest] * const.m_sun
		self.m = self.sys.mission.lander_mass
		self.rho_at_0 = self.sys.atmospheric_densities[self.mis.dest]

		self.parachute_deployed = False
		self.parachute_area = (
			2 * const.G * self.M * self.m / (self.rho_at_0 * (3 * self.R) ** 2)
		)

	def deploy(self):
		self.parachute_deployed = True

	def free_fall(self, T, dt=1e-3):
		def F_d(r, v):
			C_d = 1
			if self.parachute_deployed:
				A = self.parachute_area
			else:
				A = self.sys.mission.lander_area
			omega = 2 * np.pi / (self.sys.rotational_periods[self.mis.dest] * const.day)
			w = omega * np.asarray([-r[1], r[0], 0])
			v_d = v - w
			dens = density(np.linalg.norm(r) - self.R)
			return dens / 2 * C_d * A * np.linalg.norm(v_d) * (-v_d)

		def gravity(r):
			return -r * const.G * self.M * np.linalg.norm(r) ** -3

		def rddot(r, v):
			if np.linalg.norm(r) <= self.R:
				return np.zeros(3)
			else:
				a = gravity(r) + F_d(r, v)
				return a

		def at_surface(t, u):
			r = u[:3]
			R = np.linalg.norm(r)
			return R - self.R

		at_surface.terminal = True

		def rhs(t, u):
			r = u[:3]
			v = u[3:]
			dr = v
			dv = rddot(r, v)
			return np.concatenate((dr, dv))

		nT = int(T / dt)

		t = np.linspace(self.t, self.t + T, nT, endpoint=False)
		r0 = self.r[:, -1]
		v0 = self.v

		u0 = np.concatenate((r0, v0))
		faller = solve_ivp(
			rhs, (t[0], t[-1]), u0, t_eval=t, events=at_surface, atol=1e-6, rtol=1e-6
		)

		falled = faller.y

		r, v = np.split(falled, 2)

		self.r = np.concatenate((self.r, r), axis=1)
		self.v = v[:, -1]
		self.t = t[-1]
		plt.scatter(self.r[0, -1], self.r[1, -1])

		# self.did_we_crash(faller)

	def did_we_crash(self, result):
		succ = result.status
		if succ == 1:
			print(f"We are at ground, with velocity {self.v}")
			h0 = np.linalg.norm(self.r[:, -2])
			h1 = np.linalg.norm(self.r[:, -1])
			print(result.t)
			v_rad = (h1 - h0) / (result.t[-1] - result.t[-2])
			print(f"radial velocity is {v_rad}")

	def boost(self, v):
		self.v += v

	def orient(self):
		print(f"Time: {self.t}")
		print(f"Position: {self.r[:, -1]}")
		print(f"Velocity: {self.v}")
		return self.t, self.r[:, -1], self.v

	def plot(self):
		h = np.linspace(0, 2 * np.pi, 1000)
		x = self.R * np.cos(h)
		y = self.R * np.sin(h)
		plt.plot(x, y, color="g")
		plt.fill(x, y, color="g")

		cut = 100
		x, y, z = self.r[:, ::cut]
		plt.plot(x, y)
		plt.axis("equal")
		plt.show()

		time = np.linspace(0, self.t, len(x))
		height = np.linalg.norm(self.r[:, ::cut], axis=0) - self.R
		plt.plot(time, height)
		plt.show()

def stabilize_orbit(r0, v0, system, dest):
	r = np.linalg.norm(r0)
	t_tang_normed = np.array([-r0[1], r0[0], r0[2]]) / r

	vpm = np.sqrt(const.G * system.masses[dest] * const.m_sun / r) * t_tang_normed

	return vpm - v0

if __name__ == "__main__":
	"""
	Initialize everything from scratch
	make system and mission
	launch and verify
	use shortcut to record destination
	begin landings
	stabilize orbit
	"""
	seed = 76117
	path = "./../verification_data"
	system = SolarSys(seed, path, False, True)
	mission = SpaceMission(seed, path, False, True)
	system.mission = mission

	launch_time = 0.75
	launch_site = 0

	years = launch_time + 1
	dt_pr_yr = 1e-5
	destination = 1

	system.differential_orbits(years, dt_pr_yr)

	Volcano, Epstein = launch.do_launch(Rocket=Rocket, verb=False)
	launch.change_reference(mission, system, Volcano, Epstein, launch_site, launch_time)
	mission.verify_manual_orientation(*navigate(system, mission, path))

	Volcano.begin_interplanetary_journey(
		system, mission, destination=destination, verbose=False
	)

	time = 0.11598196795767118
	shortcut = SMS(mission, [97905])
	shortcut.place_spacecraft_in_unstable_orbit(time, destination)

	lander = mission.begin_landing_sequence(mission)
	t0, r0, v0 = lander.orient()
	boost = stabilize_orbit(r0, v0, system, destination)
	lander.boost(boost)
	t0, r0, v0 = lander.orient()

	landing = Landing(r0, v0, system, Volcano)
	# Dont do anything above this


	lander.adjust_parachute_area(landing.parachute_area)

	lander.boost(-v0*0.1)
	landing.boost(-v0*0.1)

	lander.fall(100)
	landing.free_fall(100)

	lander.orient()
	landing.orient()

	# landing.slow_down(0.9)
	# for _ in range(10):
		# landing.free_fall(1e4)
	# for _ in range(2):
		# landing.free_fall(1e4, 1e-3)
	# landing.deploy()
	# landing.slow_down(0.9)
	# for _ in range(5):
		# landing.free_fall(1e4, 1e-3)
	# print(f"time: {tim.time() - timer}")
	# print(f"time: {tim.time() - alltimer}")

	# print(landing.t)
	landing.plot()
