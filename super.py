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
import our_utilities as ot
# import status_mesages as msg

# add our package to importlist
for odir in ot.my_imports():
    sys.path.append(os.path.abspath(odir))

# import our files, preferebly using existing conventions
import launch as L_1, rocket as R_1, engine as E_1
import orbits as O_2, two_body_system as T_2, n_body_system as N_2, data_analysis as D_2
import habitable_zone as H_3
import angular_orientation as A_4  # , navigate as N_4 # navigate is currently not equiped with if name == main and is messy
# import file5
# import file6
# import file7


# Start of file
if ot.have_internet():
    util.check_for_newer_version()


seed = 76117

system = O_2.SolarSys(seed, has_moons = False)
mission = SM.SpaceMission(seed, has_moons = False)


Volcano = R_1.Rocket(*L_1.rocket_build())
Epstein = E_1.Engine()

Epstein.build(*L_1.engine_build())
Volcano.assemble(*L_1.asseble(Epstein))

Volcano.launch()

L_1.verify(mission, system, Volcano, Epstein)
print()


system.differential_orbits(20, 1e-4)
system.verify_planet_positions(20 * system.one_year, system.d_pos)


