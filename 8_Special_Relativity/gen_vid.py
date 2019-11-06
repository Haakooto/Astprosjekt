import numpy as np
import os

from ast2000tools.relativity import RelativityExperiments
import ast2000tools.utils as util
import ast2000tools.constants as const

seed = 76117
experiments = RelativityExperiments(seed)
p_idx = 1
path = "/home/hakon/Documents/MCAst/MCAst"
# endre pathen så den passe for deg

def main():
	# all take planet_idx, most can also increase height to go above clouds
	# This is defaulf False for all
	experiments.spaceship_duel(p_idx, False)
	experiments.cosmic_pingpong(p_idx)
	experiments.spaceship_race(p_idx)
	experiments.laser_chase(p_idx, False)
	experiments.neutron_decay(p_idx, False)
	experiments.antimatter_spaceship(p_idx, False)
	experiments.black_hole_descent(p_idx, 30, False, "") # number of signals [10, 100], consider light travel, path to txt_dir
	experiments.gps(p_idx, None, False) # angular pos around equator in rad; defualt random, increase height

	os.system(f"mv XMLs/*.xml {path}/data") # move all vids to dir



if __name__ == "__main__":
    main()
