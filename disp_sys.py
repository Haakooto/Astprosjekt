import ast2000tools.constants as const
import ast2000tools.utils as utils

seed = utils.get_seed("haakooto")

from ast2000tools.solar_system import SolarSystem
system = SolarSystem(seed)

system.print_info()          

r = 695510
print('My system has a {:g} solar mass star with a radius of {:g} sol.'
      .format(system.star_mass, system.star_radius/r))

for planet_idx in range(system.number_of_planets):
    print('Planet {:d} is a {} planet with a semi-major axis of {:g} AU.'
          .format(planet_idx, system.types[planet_idx], system.semi_major_axes[planet_idx]))


from ast2000tools.space_mission import SpaceMission
mission = SpaceMission(seed)

home_planet_idx = 0
print(mission.system.radii[home_planet_idx])

print(mission.spacecraft_mass, mission.spacecraft_area)
