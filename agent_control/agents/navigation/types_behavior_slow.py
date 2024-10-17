# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """
"""
more aggressive:

"""

scenario_trigger_flag = False
ignore_static_flag = False

class Cautious(object):
    """Class for Cautious agent."""
    max_speed = 20
    speed_lim_dist = 10
    speed_decrease = 12
    safety_time = 4
    min_proximity_threshold = 15
    braking_distance = 9
    overtake_counter = -1
    tailgate_counter = -1
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag

class Normal(object):
    """Class for Normal agent."""
    max_speed = 25
    speed_lim_dist = 10
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 13
    braking_distance = 8
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag

class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed = 30
    speed_lim_dist = 10
    speed_decrease = 8
    safety_time = 2
    min_proximity_threshold = 11
    braking_distance = 7
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag



class ExtremeAggressive(object):
    """Class for Aggressive agent."""
    max_speed = 35
    speed_lim_dist = 1
    speed_decrease = 1
    safety_time = 1
    min_proximity_threshold = 7
    braking_distance = 5
    overtake_counter = 0
    tailgate_counter = 0
    scenario_trigger = scenario_trigger_flag
    ignore_static = ignore_static_flag


