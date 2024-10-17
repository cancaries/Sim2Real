# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """
import re
import random
import numpy as np
from sklearn import neighbors
from agents.navigation.controller import Control
from agents.navigation.local_planner_behavior import LocalPlanner, RoadOption
from agents.navigation.types_behavior import Cautious, Aggressive, Normal, ExtremeAggressive
import sys
from scene_wo_render import Scene_wo_render
from .waypoint import Waypoint
from .tools.misc import is_within_distance, calculate_distance, positive, calculate_rotation
import copy
class BehaviorAgent():
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment,
    such as overtaking or tailgating avoidance. Adding to these are possible
    behaviors, the agent can also keep safety distance from a car in front of it
    by tracking the instantaneous time to collision and keeping it in a certain range.
    Finally, different sets of behaviors are encoded in the agent, from cautious
    to a more aggressive ones.
    """

    def __init__(self, config):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param ignore_traffic_light: boolean to ignore any traffic light
            :param behavior: type of agent to apply
        """
        self.vehicle_name = config['name']
        self.look_ahead_steps = 0
        self.end_route_flag = False
        # Vehicle information
        self.cur_waypoint = config['static_waypoint']
        self.last_waypoint = config['static_waypoint']
        self.speed = 0 # km/h
        self.velocity_xy = np.array([0, 0]) # not used km/h
        self.speed_limit = 0
        self.min_speed = 0
        self.max_speed = 0
        self.vehicle_type = config['vehicle_type']
        self.bounding_box = config['vehicle_bbox']
        self.if_overtake = False
        self.if_tailgate = False
        self.if_static = True
        self.cur_control = Control()


    def get_next_lane_id(self):
        """
        Get the next lane id of the given waypoint

            :param waypoint: current waypoint
            :return next_lane_id: next lane id
        """
        return self.cur_waypoint.lane_id

    def get_speed(self):
        """
        Get the current speed of the vehicle

            :return speed: current speed
        """
        return 0.0

    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle

            :return: control for braking
        """
        control = Control()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self.cur_control = control
        return control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        return self.emergency_stop()
