"""
Program for å analysere data samlet fra stjerneobservasjoner

Dersom dere ikke er interessert i bruk av kode og plotting av data,
er nesten det eneste interessante i denne fila i funksjonen multiple_planets()

All kode er egenskrevet
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, time

import ast2000tools.constants as const

from data_analysis import non_linear_reg, least_squares


font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}

plt.rc("font", **font)


def planet_mass_lower_bound(v, P, m_star, i=np.pi / 2):
    # equation 4 in part 2, Simulating planetary orbits
    return (
        m_star ** (2 / 3)
        * v
        * (2 * np.pi * const.G_sol) ** (-1 / 3)
        * P ** (1 / 3)
        / np.sin(i)
    )


def singe_planet(info):
    data = np.load("CM_data/radial_velocity_N_body_noise.npy")

    t, v = np.asarray(data, dtype=float)
    v -= np.mean(v)  # remove peculiar velocity

    # intiial guesses from looking at plot of data
    v0 = 0.0005
    p0 = 50
    t0 = 25
    Model = non_linear_reg(v, t)
    params = Model.solve(v0, p0, t0)
    # Using Gauss-Newton algorithm

    v_found, p_found, t0_found = params
    print(f"The best value for v is {v_found}")
    print(f"The best value for P is {p_found}")

    plt.plot(t, v, label="raw data")
    plt.plot(
        t,
        Model.f(t, *params),
        label=f"Our model. v_max = {round(v_found, 6)}, P = {round(p_found,2)}, t0 = {round(t0_found,2)}",
    )
    plt.xlabel("time [yr]")
    plt.ylabel("radial velocity [AU/ur]")
    plt.title("Radial velocity of star, raw data and Gauss-Newton model", fontsize=30)
    plt.legend(loc=1)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    star_mass = info[3]
    actual_planet_mass = info[0]

    p_mass = planet_mass_lower_bound(v_found, p_found, star_mass, i=np.pi * 2 / 3)

    print()
    print(f"Our estimate for planet mass is  {round(p_mass, 8)} solar masses")
    print(
        f"The actual mass of the planet is {round(actual_planet_mass, 8)} solar masses"
    )

    return v_found, p_mass


def singe_planet_least_square(info):
    # Does same as single_planet(), but uses least square from lecture notes
    data = np.load("CM_data/radial_velocity_N_body_noise.npy")
    t, v = np.asarray(data, dtype=float)
    v -= np.mean(v)

    Model = least_squares(v, t, 10)
    V, P, t0 = Model.find_best(0, 0.001, 30, 60, 10, 40)

    plt.plot(t, v, label="raw data")
    plt.plot(
        t,
        Model.f(t, V, P, t0),
        label=f"Our model. v_max = {round(V, 6)}, P = {round(P,2)}, t0 = {round(t0,2)}",
    )
    plt.xlabel("time [yr]")
    plt.ylabel("radial velocity [AU/yr]")
    plt.title("Radial velocity of star, raw data and regression-model", fontsize=30)
    plt.legend(loc=1)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


def spectral_line(info, vs, mp):
    data = np.load("CM_data/flux_curve_data.npy")
    t, F = np.asarray(data, dtype=float)
    plt.plot(t, F, label=f"Raw data")

    # Points found by looking at graph
    p0 = [0, 1]
    p1 = [0.56, 1]
    p2 = [1, 0.9973]
    p3 = [8, 0.9973]
    p4 = [8.4, 1]
    p5 = [8.93, 1]
    p = np.asarray([p0, p1, p2, p3, p4, p5])
    p[:, 0] /= 1000

    plt.title("Light curve")
    plt.plot(*p.T, label="Our estimation")
    plt.xlabel("time [yr]")
    plt.ylabel("Light flux [%]")
    plt.legend(loc=9)
    plt.show()

    ms = info[3]

    t0 = p1[0]
    t1 = p2[0]

    vp = vs * ms / mp
    R = (vs + vp) * (t1 - t0) / 2
    R *= const.AU * 1e-6
    density = lambda r, m: 3 * m * const.m_sun / (4 * np.pi * (r * 1e3) ** 3)
    # rho = M / V

    print(f"\nOur estimate for radius of planet is  {round(R, 3)}km")
    print(f"The actual radius of the planet is    {round(info[1], 3)}km")
    print(f"\nOur estimate for density of planet is {round(density(R, mp),3)}kg/m3")
    print(
        f"The actual desity of the planet is    {round(density(info[1], info[0]),3)}kg/m3"
    )


def multiple_planets():
    data = np.load("CM_data/radial_velocity_N_body_noise.npy")
    t, v = np.asarray(data, dtype=float)
    v -= np.mean(v)  # removing peculiar velocity

    plt.plot(t, v)
    plt.show()

    N = v.size
    dt = t[1] - t[0]

    # Fourier analysis
    sigfreq = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(N, d=dt)

    # period = 1 / frequency
    plt.plot(1 / freqs, np.abs(sigfreq))
    plt.xlabel("Period [yr]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()


def test_method():
    # Does same as multiple_planets(), but with our solar system,
    # and adds known periods
    data = np.load("npys/radial_velocity_curve_multiple.npy")
    from orbits import SolarSys

    S = SolarSys(76117)

    t, v = np.asarray(data, dtype=float)
    v -= np.mean(v)

    plt.plot(t, v)
    plt.show()

    N = v.size
    dt = t[1] - t[0]

    sigfreq = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(N, d=dt)

    plt.plot(1 / freqs, np.abs(sigfreq))
    plt.ylim([0, np.abs(sigfreq).max()])
    plt.xlabel("Period [yr]")
    plt.ylabel("Magnitude")
    plt.title("Fourier analysis on our own solar system")
    plt.grid()
    plt.show()

    for period, label in zip(S.one_years, S.names[S.ordered_planets]):
        plt.plot(np.asarray([period] * 2), np.array([-10, 1e7]), label=label)
    plt.plot(1 / freqs, np.abs(sigfreq))
    plt.ylim([0, np.abs(sigfreq).max()])
    plt.grid()
    plt.xlabel("Period [yr]")
    plt.ylabel("Magnitude")
    plt.title("Fourier analysis on our own solar system with known periods")
    plt.legend()
    plt.show()


def main():
    info = np.load("CM_data/info.npy")

    singe_planet_least_square(info)
    vs, mp = singe_planet(info)
    spectral_line(info, vs, mp)
    test_method()
    multiple_planets()


if __name__ == "__main__":
    main()
