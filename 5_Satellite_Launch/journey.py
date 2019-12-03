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

class SolarSys(SolarSys):
	def plot_orb_for_inter_jour(self, T0, T1, a=False):
		pos = self.d_pos[:, :2]

		t0_idx = np.argmin(abs(self.time - T0))
		t1_idx = np.argmin(abs(self.time - T1))
		print(t0_idx, t1_idx)

		pos = pos[:, :, t0_idx : t1_idx]

		self.plot_orbits(pos, init=False, a=a)

	# def hohmann_transfer(self, dest):
	# 	"""
	# 	Using analytical formulas from wikipedia for hohmann-transfer
	# 	"""
	# 	r1 = self.semi_major_axes[0]
	# 	r2 = self.semi_major_axes[dest]
	# 	mu = self.star_mass * const.G_sol
	# 	u = self.angles_of_all_times
	# 	u1 = u[:, 0]
	# 	u2 = u[:, dest]
	# 	U = u1 - u2

	# 	dv1 = np.sqrt(mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) -1)
	# 	dv2 = np.sqrt(mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
	# 	tH = np.pi * np.sqrt((r1 + r2) ** 3 / (8 * mu))
	# 	omega2 = np.sqrt(mu / r2 ** 3)
	# 	alpha = np.pi - omega2 * tH
	# 	print(alpha)
	# 	t0 = np.argmin(abs(U - alpha))
	# 	T0 = self.time[t0]
	# 	print(T0)

	# 	return T0, tH, dv1, dv2


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

	def teleport(self, t, pos, vel):
		if self.verb:
			print("Teleporting")
		self.travel_time = t
		# self.pos = np.concatenate((self.pos, np.transpose([pos])), axis=1)
		self.pos = np.transpose([pos])
		self.vel = np.copy(vel)

	def orient(self):
		return self.travel_time, self.pos[:, -1], self.vel

	# def commands(self, c_list):
	# 	# every other element in list is coast duration and boost ammount
	# 	coasts = c_list[::2]
	# 	boosts = c_list[1::2]

	# 	if len(coasts) != len(boosts):
	# 		boosts.append((0,0))
	# 		# make even list for zip

	# 	coasts = [[i] if type(i) != list else i for i in coasts]

	# 	if sum(coasts[:][0]) > self.system.year_convert_to(self.system.time[-1], "L"):
	# 		print("\nCoasting duration is exceeding simulated time!")
	# 		print("Terminating")
	# 		sys.exit()

	# 	for c, b in zip(coasts, boosts):
	# 		if c[0] != 0:
	# 			coasted = self.coast(*c)
	# 			if coasted.status:
	# 				print("We are close to planet")
	# 				self.travel_time = coasted.t_events[0][0]
	# 				print(f"Actual coast time: {self.system.year_convert_to(coasted.t[-1] - coasted.t[0], 'L')}")
	# 				break
	# 		self.boost(b)
	# 	self.boost(self.enter_stable_orbit_boost())
	# 	print("Stable orbit entered!")

	def coast(self, time, dt=1e-5, stop=True):
		"""
		method for finding pos after time
		uses solve_ivp

		extra planetpos is found by linear interpolate, np.repeat
		changing dt changes answer
		"""
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
	sys.exit()

	Volcano.begin_interplanetary_journey(system, mission, destination=destination, k=1)
	# print(travel.remaining_fuel_mass)
	time = 0.11598196795767118
	r_pos = np.asarray([0.12979, 0.157862])
	r_vel = np.asarray([-6.74248, 6.84742])
	Volcano.teleport(time, r_pos, r_vel)

	# dv1 = Volcano.leave_orbit_boost()
	# Volcano.boost(dv1)
	# Volcano.coast(*(0.5, 1e-6))
	# travel.boost(dv1)
	# travel.coast(Tc(0.5))

	# shortcut = SMS(mission, [97905])
	# print(Volcano.travel_time, "The traveled time")
	# shortcut.place_spacecraft_in_unstable_orbit(Volcano.travel_time, destination)

	t_idx = np.argmin(abs(system.time-Volcano.travel_time))
	R = r_pos - system.d_pos[:, Volcano.dest, t_idx]
	R *= const.AU
	print(R)
	


	# mission2 = shortcut.mission

	# travel = mission2.begin_interplanetary_travel()
	# travel.orient()
	# Volcano.teleport(*travel.orient())


	# mission2 = shortcut.mission

	# travel2 = mission2.begin_interplanetary_travel()
	# travel2.orient()


	# travel.orient()




	# Volcano.commands(cmds)
	# Volcano.coast(0.1, stop=False)

	# t1, pos1, vel1 = travel.orient()
	# plt.scatter(*pos1, label="actual_pos")


	# Volcano.coast(*(0.8, 1e-6))
	# Volcano.teleport(*travel.orient())
	# t2, pos2, vel2 = Volcano.orient()
	# print(t1, pos1, vel1)
	# print(t2, pos2, vel2)


	# travel.coast(Tc(0.1))
	# Volcano.coast(0.1, 1e-7)

	# t1, pos1, vel1 = travel.orient()
	# t2, pos2, vel2 = Volcano.orient()
	# print(t1, pos1, vel1)
	# print(t2, pos2, vel2)

	# travel.coast(Tc(0.3053763429352502))
	# Volcano.teleport(*travel.orient())
	# delta_v = Volcano.enter_stable_orbit_boost()
	# travel.boost(delta_v)
	# travel.coast(Tc(0.1))
	# t, pos, vel = travel.orient()

	# plt.scatter(*pos)
	# print(mission.time_after_launch, Volcano.travel_time)

	# Volcano.plot_journey()
	# system.plot_orb_for_inter_jour(mission.time_after_launch, Volcano.travel_time)

	# plt.legend(loc=1)
	# plt.axis("equal")
	# plt.show()
