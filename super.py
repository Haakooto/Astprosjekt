"""
Superscript for å kjøre hele romfartsoppgradet i en kjøring

all kode er egenskrevet

"""

# general imports
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time

# special imports
import ast2000tools.constants as const
import ast2000tools.utils as util
import ast2000tools.space_mission as SM
from our_utilities import *

# add our package to importlist
for odir in my_imports():
    sys.path.append(os.path.abspath(odir))

# import our files, preferebly using existing conventions
import launch as L_1, rocket as R_1, engine as E_1
import orbits as O_2, two_body_system as T_2, n_body_system as N_2, data_analysis as D_2
import habitable_zone as H_3
import angular_orientation as A_4, navigate as N_4
# import file5
# import file6
# import file7


# Start of file
if out_kjm_aud_3():
    util.check_for_newer_version()


seed = 76117
path = "./verification_data"

system = O_2.SolarSys(seed, path, False, True)
mission = SM.SpaceMission(seed, path, False, True)

simulation_years = 20
dt_pr_yr = 1e-4

system.differential_orbits(simulation_years, dt_pr_yr)
system.verify_planet_positions(simulation_years * system.one_year, system.d_pos)

launch_site = 3*np.pi/7
launch_time = 1.2932

Volcano, Epstein = L_1.do_launch()

L_1.change_reference(mission, system, Volcano, Epstein, launch_site, launch_time)

pos, vel, ang = N_4.navigate(system, mission, path)
mission.verify_manual_orientation(pos, vel, ang)



