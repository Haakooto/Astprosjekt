"""
Program for simularing av bane

All kode er egenskrevet
"""

import numpy as np
import sys, os
import time as tim
import matplotlib.pyplot as plt
from PIL import Image
import glob
import scipy.integrate as si

sys.path.append(os.path.abspath("../1_the_Rocket_Engine"))
sys.path.append(os.path.abspath("../2_Planetary_Orbits"))
sys.path.append(os.path.abspath("../3_Habitable_zone"))
sys.path.append(os.path.abspath("../4_Onboard_Orientation_Software"))


import launch
from orbits import SolarSys
from navigate import navigate
from rocket import Rocket

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

from ast2000tools.shortcuts import SpaceMissionShortcuts as SMS

font = {"family": "DejaVu Sans", "weight": "normal", "size": 19}

plt.rc("font", **font)

class SolarSys(SolarSys):
	def plot_orb_for_inter_jour(self, T0, T1, a=False):
		pos = self.d_pos[:, :2]

		t0_idx = np.argmin(abs(self.time - T0))
		t1_idx = np.argmin(abs(self.time - T1))
		print(t0_idx, t1_idx)

		pos = pos[:, :, t0_idx : t1_idx]

		self.plot_orbits(pos, init=False, a=a)

class Rocket(Rocket):
	def begin_interplanetary_journey(self, system, mission, destination, k=10, verbose=True):
		"""
		Initiate travel phase
		"""
		if verbose:
			print("\nStarting interplanetary travel\n")

		self.system = system
		self.mission = mission

		self.travel_time = mission.time_after_launch
		self.pos = np.transpose([mission._position_after_launch])
		self.vel = np.copy(mission._velocity_after_launch)

		self.dest = destination
		self.k = k
		self.verb = verbose

		self.Masses = np.concatenate(([system.star_mass], system.masses))
		self.M = system.masses[self.dest]*const.m_sun
		self.R = system.radii[self.dest]*1000

	def teleport(self, t, pos, vel):
		if self.verb:
			print("Teleporting")
		self.travel_time = t
		# self.pos = np.concatenate((self.pos, np.transpose([pos])), axis=1)
		self.pos = np.transpose([pos])
		self.vel = np.copy(vel)

	def orient(self):
		return self.travel_time, self.pos[:, -1], self.vel

		if self.verb:
			print(f"Coasting for {time} years")
		def rddot(t, r):
			"""
			Calculate acceleration of spacecraft at time t and position r
			"""
			tidx = np.argmin(abs(Tlin - t))

			Rx = r[0] - self.ri[0, :, tidx]
			Ry = r[1] - self.ri[1, :, tidx]
			R = np.vstack((Rx, Ry))
			Rnorm = np.linalg.norm(R, axis=0)

			a = lambda x: -sum(const.G_sol * self.Masses * x / Rnorm ** 3)

			return np.asarray([a(Rx), a(Ry)])

		def dominant_gravity(t, u):
			"""
			Method used by solve_ivp to terminate when dominant gravity
			"""
			r = u[:2]
			tidx = np.argmin(abs(Tlin - t))

			Rx = r[0] - self.ri[0, :, tidx]
			Ry = r[1] - self.ri[1, :, tidx]
			R = np.vstack((Rx, Ry))
			Rnorm = np.linalg.norm(R, axis=0)

			return Rnorm[self.dest + 1] - np.linalg.norm(r) * np.sqrt(self.Masses[self.dest + 1] / (self.k * self.Masses[0]))
		dominant_gravity.terminal = True

		def diffeq(t, u):
			"""
			RHS of componentwise dr and dv
			"""
			r = u[:2]
			drx, dry = u[2:]

			dvx, dvy = rddot(t, r)

			return np.array([drx, dry, dvx, dvy])

		if time == 0:
			return None # Do nothing

		if stop:
			event = dominant_gravity
		else:
			event = None

		# time is number of L_years to coast
		T0 = self.travel_time
		self.travel_time += self.system.year_convert_to(time, "E")
		T1 = self.travel_time
		dt = self.system.year_convert_to(dt, "E")
		print(T0, T1)


		nT = round((T1 - T0) / dt)

		t0_idx = np.argmin(abs(self.system.time - T0))
		t1_idx = np.argmin(abs(self.system.time - T1))
		# Find where interval starts and ends

		planets_pos = np.zeros((2, len(self.Masses), (t1_idx - t0_idx)))

		planets_pos[:, 1:, :] = self.system.d_pos[:, :, t0_idx : t1_idx]

		T0 = round(T0, 8)
		T1 = round(T1, 8)

		N = int(round(nT / planets_pos.shape[-1])) # number of new points pr point in planetpos
		nT = N * planets_pos.shape[-1] # make nT and N exactly compatible
		Tlin = np.linspace(T0, T1, nT)

		self.ri = np.repeat(planets_pos, N, axis = 2) # have planetpos in as many points as times
		for i in range(planets_pos.shape[-1] - 1):
			k0 = i * N
			k1 = k0 + N
			self.ri[:, :, k0:k1] = np.transpose(np.linspace(self.ri[:, :, k0], self.ri[:, :, k1], N), (1, 2, 0))


		u0 = np.concatenate((self.pos[:, -1], self.vel))

		U = si.solve_ivp(diffeq, (T0, T1), u0, method="Radau", t_eval=Tlin, events=event, atol=1e-5, rtol=1e-7)
		# solves problem
		if U.status:
			print("We were close!")
		u = U.y

		pos, vel = np.split(u, 2)
		# plt.scatter(*pos[:, -1], color="r")

		self.pos = np.concatenate((self.pos, pos), axis=1)
		self.vel = vel[:,-1]

		return U #dict with a lot of info about problem solution


	def boost(self, dv):
		"""
		Changes velocity, dv can be tuple or float
		"""
		try:
			dv = np.asarray(dv)
		except:
			print("Invalid dv given")
		else:
			try:
				dv = float(dv)
			except:
				if self.verb:
					print(f"Boost by array, dv = {dv} AU/yr")
				self.vel += dv
				self.fuel -= self.fuel_use(dv)
			else:
				new_vel = self.vel * dv
				if self.verb:
					print(f"Boost by float, dv = {new_vel - self.vel} AU/y")
				self.fuel -= self.fuel_use(abs(new_vel - self.vel))
				self.vel = new_vel
		finally:
			plt.scatter(*self.pos[:, -1], color="r")
			if self.fuel < 0:
				print(f"We have run out of fuel!")

	def fuel_use(self, dv):
		return 0

	def leave_orbit_boost(self):
		pos = self.pos[:, -1]

		v = np.sqrt(2 * const.G_sol * self.system.star_mass * (1 / np.linalg.norm(pos)) - 1 / self.system.semi_major_axes[self.dest])
		r = np.linalg.norm(pos)
		r_tang = np.array([-pos[1], pos[0]]) / r

		V_coast = v * r_tang

		return (V_coast - self.vel) * 0.07

	def enter_stable_orbit_boost(self):
		planet_pos = self.system.d_pos[:, self.dest, np.argmin(abs(self.system.time - self.travel_time))]

		R = self.pos[:, -1] - planet_pos

		r = np.linalg.norm(R)
		r_tang = np.array([-R[1], R[0]]) / r

		vpm = np.sqrt(const.G_sol * self.system.masses[self.dest] / r) * r_tang

		planet_pos_norm = np.linalg.norm(planet_pos)
		Rp_tang = np.array([-planet_pos[1], planet_pos[0]])
		a = self.system.semi_major_axes[self.dest]

		Vp = np.sqrt(const.G_sol * self.system.star_mass * (2 / planet_pos_norm - 1 / a))
		Vp *= Rp_tang / planet_pos_norm

		Vfinal = Vp + vpm

		return Vfinal - self.vel

	def plot_journey(self):
		plt.plot(*self.pos, "r--", label="Rocket path")
		plt.scatter(*self.pos[:, -1], color="r", label="Final pos rocket")

	def free_fall(self, r0, v0, T, title, dt=0.5):
		def rddot(r):
			return -r * const.G * self.M * np.linalg.norm(r) ** -3

		def rhs(t, u):
			r = u[:2]
			v = u[2:]
			dr = v
			dv = rddot(r)
			return np.concatenate((dr, dv))

		nT = int(T / dt)

		t = np.linspace(0, T, nT, endpoint=False)

		u0 = np.concatenate((r0, v0))
		faller = si.solve_ivp(
			rhs, (t[0], t[-1]), u0, t_eval=t, atol=1e-6, rtol=1e-6
		)

		falled = faller.y

		r, v = np.split(falled, 2)

		#self.r = np.concatenate((self.r, r), axis=1)
		#self.v = v[:, -1]
		#self.t = t[-1]

		theeta = np.linspace(0,2*np.pi,1000)

		cutoff = int(T/145650*1000)
		plt.plot(r[0,::cutoff], r[1,::cutoff])
		plt.title(title)
		plt.xlabel("Distance (m)")
		plt.ylabel("Distance (m)")
		plt.scatter(r0[0], r0[1], label="Initial position")
		plt.fill(self.R*np.cos(theeta), self.R*np.sin(theeta), label="Vogsphere")
		plt.legend()
		plt.axis("equal")
		plt.show()

		abs_r = np.linalg.norm(r, axis=0)
		max_idx = np.argmax(abs_r)
		min_idx = np.argmin(abs_r)
		max = np.amax(abs_r)
		min = np.amin(abs_r)
		a = (max + min)/2
		apoapsis = r[:,max_idx]
		periapsis = r[:, min_idx]
		return a, apoapsis, periapsis

if __name__ == "__main__":
	seed = 76117
	path = "./../verification_data"

	mission = SpaceMission(seed, path, False, True)
	system = SolarSys(seed, path, False, True)
	Tc = lambda t: system.year_convert_to(t, "E")

	years = 2
	dt_pr_yr = 1e-5
	destination = 1

	system.differential_orbits(years, dt_pr_yr)

	# T0, tH, dv1, dv2 = system.hohmann_transfer(destination)

	# cmds = [0.06, 1, 0.4] # coast for 0.06 years, do nothing, coast for 0.4 more
	# cmds = [0, dv1, tH, dv2]
	# cmds = [0.1]
	# launch_time = system.year_convert_to(T0, "L")
	launch_time = 0.75
	site = 0

	Volcano, Epstein = launch.do_launch(Rocket=Rocket, verb=False)
	launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)
	mission.verify_manual_orientation(*navigate(system, mission, path))

	Volcano.begin_interplanetary_journey(system, mission, destination=destination, k=1)
	# print(travel.remaining_fuel_mass)
	time = 0.11598196795767118
	r_pos = np.asarray([0.12979, 0.157862])
	r_vel = np.asarray([-6.74248, 6.84742])
	Volcano.teleport(time, r_pos, r_vel)

	t_idx = np.argmin(abs(system.time - time))
	planet_pos = system.d_pos[:, destination, t_idx]
	R = r_pos - planet_pos
	R *= const.AU	#position of spacecraft relative to Vogsphere [m]

	planet_vel = (system.d_pos[:,destination, t_idx+1] - planet_pos)/(system.one_year*dt_pr_yr)
	v = r_vel - planet_vel
	v = v*const.AU/const.yr	#Velocity relative to Vogsphere [m/s]

	period = 145650
	m = const.G*Volcano.M
	r = np.linalg.norm(R)
	tang_norm = np.asarray([-R[1], R[0]])/r
	r_norm = R/r
	v_stable = np.sqrt(m/r)*tang_norm
	stabilizer = v_stable - v
	v_theta = np.dot(v,tang_norm)
	v_r = np.dot(v, r_norm)
	h = r*v_theta
	p = h**2/m
	a, apoapsis, periapsis = Volcano.free_fall(R, v, 2*period, "Elliptical orbit of the spacecraft")
	e = np.sqrt(1-p/a)
	P = np.sqrt(4*np.pi**2*a**3/(const.G*Volcano.M))
	b = a*np.sqrt(1-e**2)
	print(stabilizer, v_stable)
	a2, apoapsis2, periapsis2 = Volcano.free_fall(R,v_stable,period, "Circularized orbit")
	v_theta2 = np.dot(v_stable, tang_norm)
	h2 = r*v_theta2
	p2 = h2**2/m
	e2 = np.sqrt(abs(1-p2/a2))
	print(e2)
	print(np.linalg.norm(stabilizer))
