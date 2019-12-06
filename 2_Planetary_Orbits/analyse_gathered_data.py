import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from data_analysis import non_linear_reg
import ast2000tools.constants as const


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)


def planet_mass_lower_bound(v, P, m_star):
	return m_star ** (2 / 3) * v * (2 * np.pi * const.G_sol) ** (-1 / 3) * P ** (1 / 3)


def singe_planet():
    data = np.load("npys/radial_velocity_curve_single.npy")
    t, v = np.asarray(data, dtype=float)
    v -= np.mean(v)

    v0 = 0.002
    p0 = 60
    t0 = 50
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

    info = np.load("npys/info.npy")
    star_mass = info[3]
    actual_planet_mass = info[0]

    p_mass = planet_mass_lower_bound(v_found, p_found, star_mass)

    print()
    print(f"Our estimate for planet mass is {p_mass}")
    print(f"The actual mass of the planet is {actual_planet_mass}")
    

def spectral_line():
    data = np.load("npys/flux_curve_data.npy")
    t, F = np.asarray(data, dtype=float)

    plt.plot(t, F)

    def mean_it(F_in):
        F_guess = np.zeros_like(F_in)
        F_guess[0] = np.mean(F_in[:3])
        F_guess[-1] = np.mean(F_in[-3:])

        for i in range(len(F_in)-2):
            F_guess[i + 1] = np.mean(F_in[i-1:i+2])

        return F_guess
        
    F_g = mean_it(F)
    for _ in range(100):
        F_g = mean_it(F_g)

    plt.plot(t, F_g)
    plt.show()

def multiple_planets():
    pass

def main():
    # singe_planet()
    spectral_line()
    multiple_planets()

if __name__ == "__main__":
    main()
