"""
Superscript for å kjøre hele romfartsoppgradet i en kjøring

all kode er egenskrevet

"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time
import requests
import urllib

import ast2000tools.constants as const
import ast2000tools.utils as util
from ast2000tools.space_mission import SpaceMission

import our_utilities as ot

for odir in ot.my_imports():
    sys.path.append(os.path.abspath(odir))
# sys.path.append(os.path.abspath("2_Planetary_Orbits"))
# sys.path.append(os.path.abspath("3_Habitable_zone"))
# sys.path.append(os.path.abspath("4_Onboard_Orientation_Software"))
# sys.path.append(os.path.abspath("5_Satellite_Launch"))
# sys.path.append(os.path.abspath("6_Preparing_for_Landing"))
# sys.path.append(os.path.abspath("7_Landing"))

import launch
import orbits, two_body_system, n_body_system, data_analysis
import habitable_zone
import angular_orientation  # , navigate

# import file5
# import file6
# import file7


# Start of file
if ot.have_internet():
    util.check_for_newer_version()


print("hi")
