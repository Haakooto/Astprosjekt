"""
Program for simularing av bane

All kode er egenskrevet
"""

import numpy as np
import sys, os
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

class SolarSys(SolarSys):
	def plot_orb_for_inter_jour(self, T0, T1, a=False):
		pos = self.d_pos[:, :2]

		t0_idx = np.argmin(abs(self.time - T0))
		t1_idx = np.argmin(abs(self.time - T1))

		pos = pos[:, :, t0_idx : t1_idx]

		self.plot_orbits(pos, init=False, a=a)

	def hohmann_transfer(self, dest):
		"""
		Using analytical formulas from wikipedia for hohmann-transfer
		"""
		r1 = self.semi_major_axes[0]
		r2 = self.semi_major_axes[dest]
		mu = self.star_mass * const.G_sol
		u = self.angles_of_all_times
		u1 = u[:, 0]
		u2 = u[:, dest]
		U = u1 - u2

		dv1 = np.sqrt(mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) -1)
		dv2 = np.sqrt(mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
		tH = np.pi * np.sqrt((r1 + r2) ** 3 / (8 * mu))
		omega2 = np.sqrt(mu / r2 ** 3)
		alpha = np.pi - omega2 * tH

		t0 = np.argmin(abs(U - alpha))
		T0 = self.time[t0]

		return T0, tH, dv1, dv2


class Rocket(Rocket):
	def begin_interplanetary_journey(self, system, mission, destination):
		"""
		Initiate travel phase
		"""
		self.system = system
		self.mission = mission

		self.travel_time = mission.time_after_launch
		self.pos = np.transpose([mission._position_after_launch])
		self.vel = np.copy(mission._velocity_after_launch)

		self.dest = destination
		self.k = 10

		self.Masses = np.concatenate(([system.star_mass], system.masses))

	def commands(self, c_list):
		print("\nStarting interplanetary travel\n")
		# every other element in list is coast duration and boost ammount
		coasts = c_list[::2]
		boosts = c_list[1::2]

		if len(coasts) != len(boosts):
			boosts.append((0,0))
			# make even list for zip

		coasts = [[i] if type(i) != list else i for i in coasts]

		if sum(coasts[:][0]) > self.system.year_convert_to(self.system.time[-1], "L"):
			print("\nCoasting duration is exceeding simulated time!")
			print("Terminating")
			sys.exit()

		for c, b in zip(coasts, boosts):
			if c != 0:
				coasted = self.coast(*c)
				if coasted.status:
					print("We are close to planet")
					self.travel_time = coasted.t_events[0][0]
					break
			self.boost(b)
		self.enter_stable_orbit()

	def coast(self, time, dt=1e-5, stop=True):
		"""
		method for finding pos after time
		uses solve_ivp

		extra planetpos is found by linear interpolate, np.repeat
		changing dt changes answer
		"""
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
			# Do nothing
			return None
		if stop:
			event = dominant_gravity
		else:
			event = None

		# time is number of L_years to coast
		T0 = self.travel_time
		self.travel_time += self.system.year_convert_to(time, "E")
		T1 = self.travel_time

		nT = int((T1 - T0) / dt)

		t0_idx = np.argmin(abs(self.system.time - T0))
		t1_idx = np.argmin(abs(self.system.time - T1))
		# Find where interval starts and ends

		planets_pos = np.zeros((2, len(self.Masses), (t1_idx - t0_idx)))

		planets_pos[:, 1:, :] = self.system.d_pos[:, :, t0_idx : t1_idx]
		# planetpos in specified interval

		T0 = round(T0, 8)
		T1 = round(T1, 8)

		N = round(nT / planets_pos.shape[-1]) # number of new points pr point in planetpos
		nT = N * planets_pos.shape[-1] # make nT and N exactly compatible
		Tlin = np.linspace(T0, T1, nT)

		planets_pos = np.repeat(planets_pos, N, axis = 2) # have planetpos in as many points as times
		self.ri = planets_pos

		u0 = np.concatenate((self.pos[:, -1], self.vel))

		U = si.solve_ivp(diffeq, (T0, T1), u0, method="Radau", t_eval=Tlin, events=event)
		# solves problem
		u = U.y

		pos, vel = np.split(u, 2)
		plt.scatter(*pos[:, -1], color="r")

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
				print(f"Boost, dv = {dv} AU/yr")
				self.vel += dv
				self.fuel -= self.fuel_use(dv)
			else:
				new_vel = self.vel * dv
				print(f"Boost, dv = {new_vel - self.vel} AU/yr, {dv}v_current, {new_vel}")
				self.fuel -= self.fuel_use(abs(new_vel - self.vel))
				self.vel = new_vel
		finally:
			if self.fuel < 0:
				print(f"We have run out of fuel!")

	def fuel_use(self, dv):
		return 0

	def enter_stable_orbit(self):
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

		self.boost(Vfinal - self.vel)
		print("Stable orbit entered!")

	def plot_journey(self):
		plt.plot(*self.pos, "r--", label="Rocket path")
		plt.scatter(*self.pos[:, -1], color="r", label="Final pos rocket")

	def animate_journey(self):
		print("animation not working")
		return 0
		from matplotlib.animation import FuncAnimation, FFMpegWriter

		fig = plt.figure()

		T1 = self.travel_time
		T0 = self.mission.time_after_launch

		planet_pos = self.system.d_pos[:, :2]

		t0_idx = np.argmin(abs(self.system.time - T0))
		t1_idx = np.argmin(abs(self.system.time - T1))
		planet_pos = planet_pos[:, :, t0_idx : t1_idx]

		N_points = planet_pos.shape[2]
		n_points = self.pos.shape[1]
		N = int(round(n_points / N_points))

		rocket_pos = self.pos[:, ::N]
		new_n = rocket_pos.shape[1]
		planet_pos = planet_pos[:, :, abs(N_points - new_n):]

		self.ani_pos = np.concatenate((rocket_pos.reshape(2, 1, rocket_pos.shape[1]), planet_pos), axis=1)

		# Configure figure
		plt.axis("equal")
		plt.axis("off")
		xmax = 2 * np.max(abs(self.ani_pos))
		plt.axis((-xmax, xmax, -xmax, xmax))

		# Make an "empty" plot object to be updated throughout the animation
		self.ani, = plt.plot([], [], "o", lw=1)
		# Call FuncAnimation
		self.animation = FuncAnimation(
			fig,
			self._next_frame,
			frames=range(self.ani_pos.shape[2]),
			repeat=True,
			interval=1,  # 000 * self.dt,
			blit=True,
			save_count=100,
		)

		plt.show()

	def _next_frame(self, i):
		self.ani.set_data((0, *self.ani_pos[0, :, i]), (0, *self.ani_pos[1, :, i]))
		# self.animation.set_label(("p1", "p2", "p3"))

		return (self.ani,)



if __name__ == "__main__":
	seed = 76117
	path = "./../verification_data"

	mission = SpaceMission(seed, path, False, True)
	system = SolarSys(seed, path, False, True)

	years = 20
	dt_pr_yr = 1e-4
	destination = 1

	system.differential_orbits(years, dt_pr_yr)

	T0, tH, dv1, dv2 = system.hohmann_transfer(destination)

	cmds = [0.06, 1, 0.4] # coast for 0.06 years, do nothing, coast for 0.4 more
	launch_time = 0.76
	site = -0.6

	Volcano, Epstein = launch.do_launch(Rocket=Rocket, verb=False)
	launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)
	pos, vel, ang = navigate(system, mission, path, doangle=False)

	Volcano.begin_interplanetary_journey(system, mission, destination=destination)

	Volcano.commands(cmds)

	Volcano.coast(0.1, dt=1e-7, stop=False)

	system.plot_orb_for_inter_jour(mission.time_after_launch, Volcano.travel_time)

	Volcano.plot_journey()
	plt.legend(loc=1)
	plt.axis("equal")
	plt.show()

