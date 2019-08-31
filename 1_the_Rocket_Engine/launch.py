import numpy as np
from Engine import Engine
import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt

#util.check_for_newer_version()
np.random.seed(14441111)
seed = util.get_seed("haakooto")

mission = SpaceMission(seed)
system = SolarSystem(seed)

G = const.G
mass_p = system.masses[0]*const.m_sun
r_p = system.radii[0]*1000

def escape_v(r):
    return np.sqrt(2 * G * mass_p / r)

mass = mission.spacecraft_mass
area = mission.spacecraft_area

# Variables for rocketengine
T = 10000 #temperature in K
L = 1e-5 #lengt of box in m
N = int(1e6) #number of particles

Ne = area / L ** 2 #numer of engineboxes
nozzle = 0.5

dt = 1e-12
ts = 1000

inp = [N, Ne, nozzle, T, L, dt, ts]

rocket = Engine(*inp)
rocket.build()
rocket.ignite()
rocket.performance(mass, escape_v(r_p))

fuel_mass = rocket.dm
print("loaded fuel: ", fuel_mass)
m0 = mass + fuel_mass
r0 = r_p

def acc_func(r,m):
    return rocket.thrust/m - G*mass_p/r**2

N_loop = 100000
r = np.zeros(N_loop)
v = np.zeros(N_loop)
a = np.zeros(N_loop)
r[0] = r0

i = 0
dt = 0.01
m_live = m0
while v[i]<escape_v(r[i]) and i < N_loop-1:
    # a[i] = acc_func(r[i],m_live)
    # v[i+1] = v[i] + a[i]*dt
    # r[i+1] = r[i] + v[i+1]*dt
    # m_live -= rocket.consume*dt
    # i += 1

    A0 = acc_func(r[i], m_live)
    r[i + 1] = r[i] + v[i] * dt + 0.5 * A0 * dt ** 2
    A1 = acc_func(r[i + 1], m_live)
    v[i + 1] = v[i] + 0.5 * (A0 + A1) * dt
    m_live -= rocket.consume*dt
    i += 1

v = v[:i+1]
r = r[:i+1]
a = a[:i+1]
a[-1]=a[-2]

pos = r[-1]
vel = v[-1]
acc = a[-1]
space_time = dt*i
fuel_used = rocket.consume*space_time
print("time to space: ",space_time)
print("fueluse: ",fuel_used)

print(rocket.thrust/m0-G*mass_p/r_p**2)

# t = np.linspace(0,space_time,i+1)
# plt.plot(t,v)
# plt.show()
# plt.plot(t,r)
# plt.show()
# plt.plot(t,a)
# plt.show()
