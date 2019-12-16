"""
Program for å simmulere rakettmotor

All kode er egenskrevet
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, time

import ast2000tools.constants as const
import distributions as dists

np.random.seed(14441111)


class Engine:
    def __init__(self, N_part, nozzle_size, Temp, Length, dt, ts):
        self.N = N_part
        self.T = Temp
        self.L = Length
        self.l = Length / 2
        self.nozzle = nozzle_size

        self.dt = dt
        self.ts = ts
        self.time = dt * ts

        self.sigma = np.sqrt(k_B * Temp / mass)

        self.R = (
            np.random.uniform(0, self.L, (self.N, 3)) - self.l
        )  # R[particle][dimension]
        self.V = np.random.normal(0, self.sigma, (self.N, 3))  # Like R

        self.consume = 0
        self.mom = 0  # momentum of particles that left nozzle
        self.avgV = np.zeros(self.ts - 1)  # avg exhaust velocity

        self.m = 0  # maximum particle distance from centre of box

    def ignite(self, verb=True):
        if verb:
            print("Engine ignited", end="\n\n")

        for t in range(self.ts):

            outside_box = np.where(abs(self.R) > self.l)
            # particles outside box, [particle][coord outside]

            absvel = abs(self.V[outside_box])
            # velocites in coordinate particle is outside of box

            if t != 0:  # at t = 0 all are inside
                self.avgV[t - 1] = np.mean(absvel)

            self.mom += sum(absvel) * mass
            self.consume += len(outside_box[0])  # number of particles that left box

            self.m = max(self.m, self.R.max())

            self.V[outside_box] *= -1  # Bounce ellastically off wall
            self.R += self.V * self.dt  # Updating position of particles.

        self.thrust = self.mom * self.nozzle / 6 / self.time  # [N]
        self.consume = self.consume * mass * self.nozzle / 6 / self.time  # [1]

        self.pressure = self.N * k_B * self.T * self.L ** (-3)  # [N/m**2]
        self.pres_thrust = self.pressure * self.nozzle * self.L ** 2 / 6  # [N]

        self.exhaust_v = np.mean(self.avgV)

    def test_performance(self):  # Some tests to validate simulation
        if self.m > self.l * 1.1:
            # if timestep is too high, particles will go far outside
            # This tests if particles always are relatively close to the box
            print("Particles left the box, timestep is too high")
        else:
            print("Timestep is good")

        # plot histogram of vx, vy ,vz and V
        for x, y in enumerate(["x", "y", "z"]):
            max_bolz = dists.P_mb(
                round(self.V[:, x].min()),
                round(self.V[:, x].max()),
                False,
                self.T,
                mass,
            )
            plt.plot(max_bolz[0], max_bolz[1], label="analytical velocity distribution")
            hist = plt.hist(
                self.V[:, x],
                bins="auto",
                density=True,
                label="system velocity distribution",
            )
            print(sum(hist[0]) * (hist[1][-1] - hist[1][-2]))
            print(sum(hist[0][np.where(abs(hist[1]) <= self.sigma)]) / sum(hist[0]))

            plt.xlabel(f"velocity in {y} direction")
            plt.ylabel(f"number of particles in bin")
            plt.title(f"Histogram of velocities in {y} direction")
            plt.legend(loc=5)
            plt.show()

        V = np.linalg.norm(self.V, axis=1)
        plt.hist(V, bins="auto", density=True, label="system velocity distribution")
        max_bolz = dists.P_mb(0, V.max(), True, self.T, mass)
        plt.plot(max_bolz[0], max_bolz[1], label="analytical velocity distribution")

        plt.xlabel("velocity")
        plt.ylabel("number of particles with velocity")
        plt.title("Histogram of velocities")
        plt.legend(loc=1)
        plt.show()


# Setting constants
k_B = const.k_B  # boltzmann constant
mass = const.m_H2  # particle mass

if __name__ == "__main__":

    # Variables
    T = 2800  # temperature in K
    L = 1e-7  # lengt of box in m
    N = int(1e5)  # number of praticles

    nozzle_size = 0.25  # area of nozzle in L**2

    dt = 1e-12
    ts = 1000

    inp = [N, nozzle_size, T, L, dt, ts]

    rocket_area = 16
    throttle = 1 / 12
    time = 343.82
    fuel_load = 50000

    rocket = Engine(*inp)
    rocket.ignite()
    rocket.test_performance()
    mass_loss = rocket.consume*time*rocket_area/L**2*throttle
    print("Thrust [N]: ", rocket.thrust)
    print("mass consumed [kg/s]: ", rocket.consume)
    print("exhaust V [m/s]: ", rocket.exhaust_v)
    print(f"mass consumed to reach escape velocity [kg]: {mass_loss}")
    print("Fuel mass remaining [kg]: ", fuel_load - mass_loss)
    print(1177.33*(1-np.exp(-1187/rocket.exhaust_v)))
