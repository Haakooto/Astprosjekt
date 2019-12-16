"""
Program for rakett klasse

All kode er egenskrevet

I resultatene presenterer vi plot av akselerasjon, fart og posisjon.
For dette måtte vi skrive om koden til å ta vare på verdiene for r og v,
men klarte senere å slette denne koden.
"""

import numpy as np
import sys, os
import ast2000tools.constants as const


class Rocket:
    def __init__(self, m0, r0, M, dt, verbose=True):
        if verbose:
            print("Build rocket")
        self.mass = m0  # rocket drymass
        self.r = r0  # rocket position
        self.R = r0  # planet radius
        self.M = M  # planet mass
        self.dt = dt  # timestep

        self.v = 0  # initial velocity
        self.time = 0  # start time

        self.verb = verbose

    def assemble(self, engine, fuel_mass, N_engines):
        self.engine = engine  # instance of engine-class
        self.fuel = fuel_mass
        self.Ne = N_engines  # Number of engine boxes

    def launch(self):
        self.engine.ignite(self.verb)

        self.thrust = self.engine.thrust * self.Ne
        self.burn_rate = self.engine.consume * self.Ne

        while True:

            # leapfrog integration
            a0 = self.acceleration()
            self.r += self.v * self.dt + 0.5 * a0 * self.dt ** 2
            a1 = self.acceleration()
            self.v += 0.5 * (a0 + a1) * self.dt

            self.fuel -= self.burn_rate * self.dt
            self.time += self.dt

            # 3 possible outcomes of launch:
            if self.escaped():
                self.status = ("Launch successful", 0)
                break
            elif self.fuel < 0:
                self.status = ("Burnout", 1)
                break
            elif self.r < self.R:  # if underground
                self.status = ("RUD in LAO", 1)
                # Rapid Unschedules Disassembly in Low Atmospheric Orbit
                break

        if self.verb:
            self.statusrapport()

    def escaped(self):
        # kinetic and gravitational potential energy
        ke = 0.5 * self.v ** 2
        pe = const.G * self.M / self.r
        return ke > pe

    def acceleration(self):
        # equation 6 in "Modelling a rocket launch"
        return self.thrust / (self.mass + self.fuel) - const.G * self.M / self.r ** 2

    def fuel_use(self, dv):
        # Equation derived in part 5, "Interplanetary spacetravel"
        dv = np.linalg.norm(dv)
        dm = (self.mass + self.fuel) * (1 - np.exp[-dv / self.engine.exhaust_v])

        return dm

    def statusrapport(self):
        print(self.status[0], "\n")
        print(f"Altitude: {self.r - self.R}")
        print(f"Speed: {self.v}")
        print(f"fuel left: {self.fuel}")
        print(f"Time: {self.time}\n")

        if self.status[1] == 1:
            sys.exit()
