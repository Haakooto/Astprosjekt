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
from atmosphere import densityv

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem


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

	def free_fall(self, T, dt):
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

	def slow_down(self, v):
		self.v *= v

	def plot(self):
		h = np.linspace(0, 2 * np.pi, 1000)
		x = self.R * np.cos(h)
		y = self.R * np.sin(h)
		plt.plot(x, y)

		x, y, z = self.r[:, ::100]
		plt.plot(x, y)
		plt.axis("equal")
		plt.show()


if __name__ == "__main__":
	alltimer = tim.time()
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

	time = 0.11598196795767118
	r_pos = np.asarray([0.12979, 0.157862])
	r_vel = np.asarray([-6.74248, 6.84742])
	Volcano.begin_interplanetary_journey(
		system, mission, destination=destination, verbose=False
	)
	Volcano.teleport(time, r_pos, r_vel)
	Volcano.boost(Volcano.enter_stable_orbit_boost())

	"""
	To have accurate planet position and velocity at time we use
	methods from SolarSystem found by using dir(SolarSystem),
	and guessed based on name which parameters the methods took.
	We did this to save time by not having to calculate it ourself
	each time we ran code, and accuracy
	"""
	p_pos = system._compute_single_planet_position(time, destination)
	p_vel = system._compute_single_planet_velocity(time, destination)

	r0 = r_pos - p_pos
	r0 = np.array([*r0, 0]) * const.AU
	v0 = Volcano.vel - p_vel
	v0 = np.array([*v0, 0]) * const.AU / const.yr

	landing = Landing(r0, v0, system, Volcano)
	timer = tim.time()

	for _ in range(2):
		landing.free_fall(1e4, 1e-3)
		landing.slow_down(0.9)
	for _ in range(2):
		landing.free_fall(1e4, 1e-3)
	landing.deploy()
	landing.slow_down(0.9)
	for _ in range(5):
		landing.free_fall(1e4, 1e-3)
	print(f"time: {tim.time() - timer}")
	print(f"time: {tim.time() - alltimer}")

	landing.plot()

	# R = system.radii[1]*1000
	# def F_d(A, r, v, C_d=1):
	#     omega = 2*np.pi/(system.rotational_periods[1]*const.day)
	#     w = omega*np.asarray([-r[1], r[0], 0])
	#     v_d = v - w
	#     dens = density(np.linalg.norm(r) - R)
	#     return 1/2*dens*C_d * A*np.linalg.norm(v_d)*(-v_d)

	# r0 = np.asarray([500000 + R, 0, 0])
	# v0 = np.asarray([2000, 0, 0])

	# A = 2*const.G*system.masses[1]*const.m_sun*mission.lander_mass/(system.atmospheric_densities[1]*(3*R)**2)

	# launch_time = 0.75
	# site = 0
	# destination = 1
	# Volcano, Epstein = launch.do_launch(Rocket=Rocket, verb=False)
	# launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)
	# mission.verify_manual_orientation(*navigate(system, mission, path))
	# Volcano.begin_interplanetary_journey(system, mission, destination=destination, k=1)
	# mission.begin_landing_sequence()

