import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from data_analysis import non_linear_reg
import ast2000tools.constants as const


font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)


def planet_mass_lower_bound(v, P, m_star):
	return m_star ** (2 / 3) * v * (2 * np.pi * const.G_sol) ** (-1 / 3) * P ** (1 / 3)


def singe_planet(info):
    data = np.load("CM_data/rad_vel_2bds_noise.npy")
    t, v = np.asarray(data, dtype=float)
    v -= np.mean(v)

    v0 = 0.0005
    p0 = 45
    t0 = 25
    Model = non_linear_reg(v, t)
    params = Model.solve(v0, p0, t0)

    v_found, p_found, t0_found = params
    print(f"The best value for v is {v_found}")
    print(f"The best value for P is {p_found}")

    plt.plot(t, v, label="raw data")
    plt.plot(t, Model.f(t, *params), label="Our model")
    plt.xlabel("time")
    plt.ylabel("radial velocity")
    plt.title("Radial velocity of observed star, noisy and model", fontsize=30)
    plt.legend()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    star_mass = info[3]
    actual_planet_mass = info[0]

    p_mass = planet_mass_lower_bound(v_found, p_found, star_mass)

    print()
    print(f"Our estimate for planet mass is  {round(p_mass, 8)} solar masses")
    print(f"The actual mass of the planet is {round(actual_planet_mass, 8)} solar masses")
    
    return v_found, p_mass

def spectral_line(info, vs, mp):
    data = np.load("CM_data/flux_curve_data.npy")
    t, F = np.asarray(data, dtype=float)
    t *= 1000
    plt.plot(t, F)

    p0 = [0, 1]
    p1 = [0.56, 1]
    p2 = [1, 0.9973]
    p3 = [8, 0.9973]
    p4 = [8.4, 1]
    p5 = [8.93, 1]
    p = np.asarray([p0, p1, p2, p3, p4, p5])
    plt.plot(*p.T)

    plt.show()

    ms = info[3]    

    t0 = p1[0]
    t1 = p2[0]

    vp = vs * ms / mp
    R = (vs + vp) * (t1 - t0) / 2
    R *= const.AU * 1e-6
    density = lambda r, m: 3 * m * const.m_sun / (4 * np.pi * (r * 1e3) ** 3)

    print(f"\nOur estimate for radius of planet is  {round(R, 3)}km")
    print(f"The actual radius of the planet is    {round(info[1], 3)}km") 
    print(f"\nOur estimate for density of planet is {round(density(R, mp),3)}kg/m3")
    print(f"The actual desity of the planet is    {round(density(info[1], info[0]),3)}kg/m3")


def multiple_planets(info):
    data = np.load("CM_data/radial_velocity_2_body_data.npy")
    t, v = np.asarray(data, dtype=float)
    plt.plot(t, v)
    plt.show()

def main():
    info = np.load("CM_data/info.npy")

    # vs, mp = singe_planet(info)
    # spectral_line(info, vs, mp)
    multiple_planets(info)

if __name__ == "__main__":
    main()
