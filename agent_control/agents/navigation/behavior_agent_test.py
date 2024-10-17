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
from agents.navigation.types_behavior import Cautious, Aggressive, Normal, ExtremeAggressive, \
    Cautious_fast, Aggressive_fast, Normal_fast, ExtremeAggressive_fast,\
    Cautious_highway, Aggressive_highway, Normal_highway, ExtremeAggressive_highway
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
        self.cur_waypoint = config['initial_path'][0]
        self.last_waypoint = config['initial_path'][0]
        self.speed = 0 # km/h
        self.velocity_xy = [0, 0] # km/h
        self.acceleration_xy = [0, 0] # m/s^2
        self.omega = 0 # rad/s
        self.steer_value = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.start_waypoint = None
        self.end_waypoint = None
        self.plan_path = config['initial_path']
        self.min_speed = 5
        self.behavior = None
        self.vehicle_type = config['vehicle_type']
        self.bounding_box = config['vehicle_bbox']
        self.f_len = config['f_len']
        self.r_len = config['r_len']
        self.control_angel = config['control_angel']
        self.speed_type = config['speed_type']
        self.vehicle_type = config['vehicle_type']
        self._sampling_resolution = 4.5
        self.cur_control = Control()
        self.if_overtake = False
        self.overtake_direction = None
        self.if_tailgate = False
        self.overtake_end_waypoint = None
        self.vehicle_name_to_overtake = None
        self.if_static = False
        
        # self.features=Scene_wo_render._map.features

        # Parameters for agent behavior
        behavior = config['behavior']
        if behavior == 'cautious':
            if self.speed_type == 'fast':
                self.behavior = Cautious_fast()
            elif self.speed_type == 'highway':
                self.behavior = Cautious_highway()
            else:
                self.behavior = Cautious()

        elif behavior == 'normal':
            if self.speed_type == 'fast':
                self.behavior = Normal_fast()
            elif self.speed_type == 'highway':
                self.behavior = Normal_highway()
            else:
                self.behavior = Normal()

        elif behavior == 'aggressive':
            if self.speed_type == 'fast':
                self.behavior = Aggressive_fast()
            elif self.speed_type == 'highway':
                self.behavior = Aggressive_highway()
            else:
                self.behavior = Aggressive()

        elif behavior == 'extreme_aggressive':
            if self.speed_type == 'fast':
                self.behavior = ExtremeAggressive_fast()
            elif self.speed_type == 'highway':
                self.behavior = ExtremeAggressive_highway()
            else:
                self.behavior = ExtremeAggressive()

        self.scenario_trigger = self.behavior.scenario_trigger
        # self.behavior.min_proximity_threshold = self.behavior.min_proximity_threshold + max(self.bounding_box[0], self.bounding_box[1])/2
        # self.behavior.braking_distance = self.behavior.braking_distance + max(self.bounding_box[0], self.bounding_box[1])/2
        self._local_planner = LocalPlanner(self)
        self.speed_limit = self.behavior.max_speed
        self._local_planner.set_speed(self.speed_limit)
        self._local_planner.set_global_plan(self.plan_path,clean=True)
        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)

    def update_information(self, update_transform):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self.direction = self._local_planner.target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW
        self.look_ahead_steps = int((self.speed_limit) / 15)
        location = update_transform.location
        rotation = update_transform.rotation
        # 在行驶过程中可能最邻近车道会因为车辆行驶行为的控制越到其他车道上，因而需要重新获取当前车道的lane_id和lane_point_idx
        neighbour_ids = Scene_wo_render._map.get_all_neighbor_lane_ids(self.cur_waypoint.lane_id)
        cur_lane_id, cur_point_index, _ = Scene_wo_render._map.find_nearest_lane_point(location)
        # 如果当前车道不在邻近车道中，则将当前车道设置为当前车道，而cur_point_index由于在后续过程中并不需要使用，所以更新的正确与否并不重要
        # 主要使用的是lane_id所能代表的邻近车道关系
        if cur_lane_id not in neighbour_ids:
            cur_lane_id = self.cur_waypoint.lane_id
            cur_point_index = self.cur_waypoint.lane_point_idx
        cur_waypoint_config = Scene_wo_render._map.build_waypoint_config(location, rotation, self.direction, cur_lane_id, cur_point_index)
        self.cur_waypoint = Waypoint(cur_waypoint_config)
        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
        if self.incoming_waypoint is None:
            self.incoming_waypoint = self.cur_waypoint
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW
    
    def set_last_waypoint(self, last_waypoint):
        self.last_waypoint = copy.deepcopy(last_waypoint)

    def get_speed(self):
        """
        Getter for current speed of the agent.

            :return: speed of the agent
        """
        return self.speed

    def set_speed(self, speed):
        """
        Setter for current speed of the agent.

            :param speed: new speed for the agent

        """
        # print(f'{self.vehicle_name} set speed to {speed} from {self.speed}')
        self.speed = speed
        self.velocity_xy = [speed * np.cos(self.cur_waypoint.get_rotation()[2]),
                                     speed * np.sin(self.cur_waypoint.get_rotation()[2])]

    def set_omega(self, omega):
        self.omega = omega

    def set_steer_value(self, steer_value):
        self.steer_value = steer_value

    def set_acceleration(self, acceleration):
        self.acceleration_xy = [acceleration * np.cos(self.cur_waypoint.get_rotation()[2]),
                                     acceleration * np.sin(self.cur_waypoint.get_rotation()[2])]


    def rearrange_route(self, additional_route):
        """
        This method rearranges the route to follow the additional_route
        if it is possible to do so.

            :param additional_route: new route to follow
        """
        if additional_route is None or len(additional_route) == 0:
            return
        end_point_of_additional_route = copy.deepcopy(additional_route[-1])
        end_point_of_plan_path = self.plan_path[-1]
        new_route = self._trace_route(end_point_of_additional_route, end_point_of_plan_path)
        if new_route:
            route_combined = additional_route + new_route[1:]
            route_combined = route_combined[1:]
            self._local_planner.set_global_plan(route_combined, clean=True,clean_global=True)
            self.plan_path = copy.deepcopy(route_combined)
        else:
            print("Route Invalid. Keeping the current plan.")
            pass

    def reroute(self, cur_loc):
        """
        This method implements re-routing for vehicles approaching its destination.
        It finds a new target and computes another path to reach it.
            :param cur_loc: current location of the agent
        """

        print("Target almost reached, setting new destination...")
        new_route = Scene_wo_render._map.generate_overall_plan_waypoints(cur_loc)
        self._local_planner.set_global_plan(new_route, clean=True, clean_global=True)

    def reroute_all(self, new_plan):
        """
        This method implements re-routing for vehicles approaching its destination.
        It finds a new target and computes another path to reach it.
            :param new_plan: new plan to follow
        """
        # print(new_plan)
        new_plan = Scene_wo_render._map.refine_plan_waypoints(new_plan,2.0)
        self.plan_path = copy.deepcopy(new_plan)
        self._local_planner.set_global_plan(new_plan, clean=True, clean_global=True)


    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the
        optimal route from start_waypoint to end_waypoint.

            :param start_waypoint: initial position
            :param end_waypoint: final position
        """
        return Scene_wo_render.generate_route_w_waypoints(start_waypoint, end_waypoint)

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

    def _bh_is_vehicle_hazard(self, ego_wpt, vehicle_info_list,
                          proximity_th, up_angle_th, low_angle_th=0, lane_offset=0,scenario_flag=False):
        """
        检查是否有车辆在规定距离内，返回最近车辆的信息。
        :param ego_wpt: 自车的路点信息
        :param vehicle_info_list: 其他车辆的信息列表
        :param proximity_th: 距离阈值
        :param up_angle_th: 角度上限
        :param low_angle_th: 角度下限
        :param lane_offset: 车道偏移量
        :return: 一个包含True和最近车辆name以及distance的元组
        """
        ego_loc = ego_wpt.transform.location
        ego_yaw = ego_wpt.transform.rotation[2]
        ego_lane_id = ego_wpt.lane_id
        ego_exit_lane_id = self.get_next_lane_id()
        neighbour_ids = []
        if lane_offset == -1:
            neighbour_ids = Scene_wo_render._map.get_left_neighbor_lane_ids(ego_lane_id)
        elif lane_offset == 1:
            neighbour_ids = Scene_wo_render._map.get_right_neighbor_lane_ids(ego_lane_id) 
        # 存储符合距离要求的车辆信息
        vehicle_dict = {}

        for target_vehicle_info in vehicle_info_list:
            if 'static' in target_vehicle_info['name'] and self.behavior.ignore_static:
                continue
            target_vehicle_loc = target_vehicle_info['location']
            target_vehicle_yaw = target_vehicle_info['yaw']
            target_lane_id = target_vehicle_info['lane_id']
            target_exit_lane_id = target_vehicle_info['exit_lane_id']   
            if abs(ego_yaw - target_vehicle_yaw) > np.pi/8*7 or abs(ego_yaw - target_vehicle_yaw) < np.pi/8:
                continue
            # print(f"judge between {self.vehicle_name} {target_vehicle_info['name']}", abs(ego_yaw - target_vehicle_yaw))
            # 如果车辆不在相同的车道，跳过
            if scenario_flag:
                if target_lane_id not in neighbour_ids:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if next_wpt is None:
                        next_wpt = ego_wpt
                    next_neighbour_ids = []
                    if lane_offset == -1:
                        next_neighbour_ids = Scene_wo_render._map.get_left_neighbor_lane_ids(next_wpt.lane_id)
                    elif lane_offset == 1:
                        next_neighbour_ids = Scene_wo_render._map.get_right_neighbor_lane_ids(next_wpt.lane_id)
                    if target_lane_id not in next_neighbour_ids:
                        continue

        # 检查车辆是否在规定距离内
            if is_within_distance(target_vehicle_loc, ego_loc, ego_yaw,
                              proximity_th, up_angle_th, low_angle_th):
                distance = calculate_distance(target_vehicle_loc, ego_loc)
                # 将车辆名字，距离，exitlaneid和位置存储到字典中
                vehicle_dict[target_vehicle_info['name']] = distance, target_lane_id, target_exit_lane_id, target_vehicle_loc

        # 如果字典不为空，找到距离最小的车辆
        if vehicle_dict:
            # 初始化最小距离和最接近车辆名称的变量
            min_distance = float('inf')
            closest_vehicle_name = None
            # 遍历字典，查找最小距离的车辆
            for vehicle_name, (distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc) in vehicle_dict.items():
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle_name = vehicle_name
            closest_distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc = vehicle_dict[closest_vehicle_name]
            if ego_exit_lane_id is None:
                ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene_wo_render._map.features[ego_lane_id].polyline[-1])
            else:
                ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene_wo_render._map.features[ego_exit_lane_id].polyline[0])
            if ref_exit_lane_id is None:
                ref_exit_lane_distance = calculate_distance(ref_vehicle_loc, Scene_wo_render._map.features[ref_lane_id].polyline[-1])
            else:
                ref_exit_lane_distance = calculate_distance(ref_vehicle_loc,Scene_wo_render._map.features[ref_exit_lane_id].polyline[0])
            if ref_exit_lane_distance > ego_exit_lane_distance:
                return (False, None, closest_distance)
            
            return (True, closest_vehicle_name, closest_distance)

        # 没有车辆符合条件
        return (False, None, -1)

    def _bh_is_vehicle_hazard_straight(self, ego_wpt, vehicle_info_list,
                           proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle. We also check the next waypoint, just to be
        sure there's not a sudden road id change.

        WARNING: This method is an approximation that could fail for very large
        vehicles, which center is actually on a different lane but their
        extension falls within the ego vehicle lane. Also, make sure to remove
        the ego vehicle from the list. Lane offset is set to +1 for right lanes
        and -1 for left lanes, but this has to be inverted if lane values are
        negative.

            :param ego_wpt: waypoint of ego-vehicle
            :param ego_log: location of ego-vehicle
            :param vehicle_list: list of potential obstacle to check
            :param proximity_th: threshold for the agent to be alerted of
            a possible collision
            :param up_angle_th: upper threshold for angle
            :param low_angle_th: lower threshold for angle
            :param lane_offset: for right and left lane changes
            :return: a tuple given by (bool_flag, vehicle, distance), where:
            - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
            - vehicle is the blocker object itself
            - distance is the meters separating the two vehicles
        """
        ego_loc = ego_wpt.transform.location
        ego_yaw = ego_wpt.transform.rotation[2]
        ego_lane_id = ego_wpt.lane_id
        ego_exit_lane_id = self.get_next_lane_id()
        vehicle_dict = {}
        for target_vehicle_info in vehicle_info_list:
            # if 'static' in target_vehicle_info['name']:
            #     continue
            
            target_vehicle_loc = target_vehicle_info['location']
            target_vehicle_yaw = target_vehicle_info['yaw']
            target_lane_id = target_vehicle_info['lane_id']
            target_exit_lane_id = target_vehicle_info['exit_lane_id']
            if abs(ego_yaw - target_vehicle_yaw) > np.pi/8:
                continue
            # 如果两辆车大致方向完全相反，跳过,yaw的范围是-pi到pi
            # if abs(ego_yaw - target_vehicle_yaw) > np.pi/2:
            #     continue
            # print(f"judge between {self.vehicle_name} {target_vehicle_info['name']}", abs(ego_yaw - target_vehicle_yaw))
            if target_lane_id != ego_lane_id and ego_exit_lane_id != target_lane_id:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if next_wpt is None:
                    next_wpt = ego_wpt
                if  target_lane_id != next_wpt.lane_id:
                    continue
            if is_within_distance(target_vehicle_loc, ego_loc,
                                  ego_yaw,
                                  proximity_th, up_angle_th, low_angle_th):
                distance = calculate_distance(target_vehicle_loc, ego_loc)
                # 将车辆名字，距离，exitlaneid和位置存储到字典中
                vehicle_dict[target_vehicle_info['name']] = distance, target_lane_id, target_exit_lane_id, target_vehicle_loc
        # 如果字典不为空，找到距离最小的车辆
        if vehicle_dict:
            # 初始化最小距离和最接近车辆名称的变量
            min_distance = float('inf')
            closest_vehicle_name = None
            # 遍历字典，查找最小距离的车辆
            for vehicle_name, (distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc) in vehicle_dict.items():
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle_name = vehicle_name
            closest_distance, ref_lane_id, ref_exit_lane_id, ref_vehicle_loc = vehicle_dict[closest_vehicle_name]
            if ego_exit_lane_id is None:
                ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene_wo_render._map.features[ego_lane_id].polyline[-1])
            else:
                ego_exit_lane_distance = calculate_distance(ego_wpt.transform.location, Scene_wo_render._map.features[ego_exit_lane_id].polyline[0])
            if ref_exit_lane_id is None:
                ref_exit_lane_distance = calculate_distance(ref_vehicle_loc, Scene_wo_render._map.features[ref_lane_id].polyline[-1])
            else:
                ref_exit_lane_distance = calculate_distance(ref_vehicle_loc,Scene_wo_render._map.features[ref_exit_lane_id].polyline[0])
            if ref_exit_lane_distance > ego_exit_lane_distance:
                return (False, None, closest_distance)
            
            return (True, closest_vehicle_name, closest_distance)
        return (False, None, -1)



    def _overtake(self, waypoint, vehicle_list):
        """
        This method is in charge of overtaking behaviors.

            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_lines = waypoint.get_left_lane()
        right_lines = waypoint.get_right_lane()
        location = waypoint.transform.get_location()
        if left_lines is not None:
            new_vehicle_state, vehicle_name, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=-1,scenario_flag=True)
            if not new_vehicle_state:
                # print("Overtaking to the left!")
                self.behavior.overtake_counter = 200
                driving_mode, additional_route = Scene_wo_render._map.get_plan_waypoints_w_refine(location, driving_mode='CHANGELANELEFT')
                if additional_route is not None:
                    self.overtake_end_waypoint = additional_route[-1]
                    self.rearrange_route(additional_route)
                    self.if_overtake = True
                    self.vehicle_name_to_overtake = vehicle_name
                    self.overtake_direction = 'left'
        elif right_lines is not None:
            new_vehicle_state, vehicle_name, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=1,scenario_flag=True)
            if not new_vehicle_state:
                # print("Overtaking to the right!")
                self.behavior.overtake_counter = 200
                driving_mode, additional_route = Scene_wo_render._map.get_plan_waypoints(location, driving_mode='CHANGELANERIGHT')
                if additional_route is not None:
                    self.overtake_end_waypoint = additional_route[-1]
                    self.rearrange_route(additional_route)
                    self.if_overtake = True
                    self.vehicle_name_to_overtake = vehicle_name
                    self.overtake_direction = 'right'
        else:
            new_vehicle_state, vehicle_name, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=30)
            if vehicle_name:
                # 如果还没有到达上一次超车路径的最后一个点，不再超车
                if Scene_wo_render._agent_dict[vehicle_name].cur_waypoint.lane_id == waypoint.lane_id:
                    self.vehicle_name_to_overtake = vehicle_name
                    self._force_overtake()

    def _force_overtake(self,direction='left'):
        # concatencate deque
        cur_plan = list(self._local_planner._waypoint_buffer)+list(self._local_planner.waypoints_queue)
        driving_mode, additional_route = Scene_wo_render._map.generate_overtake_path_from_reference_path(cur_plan,direction=direction)
        if additional_route is not None: 
            self.behavior.overtake_counter = 200
            self.overtake_end_waypoint = additional_route[-1]
            self.rearrange_route(additional_route)
            self.overtake_direction = direction
            self.if_overtake = True
        else:
            # print("No overtaking path found!")
            pass
        
    def _force_turn_back(self):
        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=1)[0]
        if next_wpt is None:
            return
        if self.overtake_direction == 'left':
            direction = 'CHANGELANERIGHT'
        elif self.overtake_direction == 'right':
            direction = 'CHANGELANELEFT'
        driving_mode, additional_route = Scene_wo_render._map.generate_waypoint_path_from_two_points(self.cur_waypoint, next_wpt, direction=direction)
        self.rearrange_route(additional_route)
        self.overtake_direction = None
        self.if_overtake = False
        
    # 原地掉头
    def _force_turn_around(self):
        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=1)[0]
        if next_wpt.is_junction:
            return
        driving_mode, turn_around_route = Scene_wo_render._map.generate_turn_around_path(next_wpt)
        if turn_around_route is not None:
            self.reroute_all(turn_around_route)


    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_lines = waypoint.get_left_lane()
        right_lines = waypoint.get_right_lane()
        location = waypoint.transform.location
        behind_vehicle_state, behind_vehicle, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
            self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self.speed < Scene_wo_render.get_vehicle_speed(behind_vehicle):
            if  right_lines is not None:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1,scenario_flag=True)
                if not new_vehicle_state:
                    # print("Tailgating, moving to the right!")
                    self.behavior.tailgate_counter = 200
                    driving_mode, additional_route = Scene_wo_render._map.get_plan_waypoints_w_refine(location, driving_mode='CHANGELANERIGHT')
                    self.rearrange_route(additional_route)
                    self.if_tailgate = True
            elif left_lines is not None:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1,scenario_flag=True)
                if not new_vehicle_state:
                    # print("Tailgating, moving to the left!")
                    self.behavior.tailgate_counter = 200
                    driving_mode, additional_route = Scene_wo_render._map.get_plan_waypoints_w_refine(location, driving_mode='CHANGELANELEFT')
                    self.rearrange_route(additional_route)
                    self.if_tailgate = True

    def get_next_lane_id(self):
        """
        This method is in charge of getting the next lane id.

        :return next_lane_id: next lane id
        """
        cur_lane_id = self.cur_waypoint.lane_id
        next_lane_id = None
        step_num = 1
        while next_lane_id is None or next_lane_id == cur_lane_id:
            if step_num > 10:
                break
            next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=step_num)[0]
            # If there is no next waypoint, we return the current lane id
            if next_wpt is None:
                next_wpt = self.cur_waypoint
                next_lane_id = next_wpt.lane_id
                break
            next_lane_id = next_wpt.lane_id
            # If the next lane id is different from the current lane id, we return it
            if next_lane_id != cur_lane_id:
                return next_lane_id
            step_num += 1
        return None

    def get_near_vehicle_info(self, max_distance=45):
        """
        This method is in charge of getting the information of the nearby vehicles.

            :param min_distance: minimum distance to consider
            :return vehicle_list: list of all the nearby vehicles
        """
        vehicle_info_list = []
        ego_vehicle_loc = Scene_wo_render.get_vehicle_location(self.vehicle_name)
        for vehicle_name, agent in Scene_wo_render._agent_dict.items():
            if vehicle_name != self.vehicle_name:
                vehicle_wp = Scene_wo_render.get_vehicle_waypoint(vehicle_name)
                vehicle_loc = vehicle_wp.transform.location
                yaw = vehicle_wp.transform.rotation[2]
                distance = calculate_distance(vehicle_loc, self.cur_waypoint.transform.location)
                # get exit lane id of the vehicle
                vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
                exit_lane_id = vehicle_agent.get_next_lane_id()
                if distance < max_distance:
                    vehicle_info_list.append({'name': vehicle_name, 'location': vehicle_loc,'yaw':yaw, 'lane_id': vehicle_wp.lane_id, 'exit_lane_id': exit_lane_id})

        return vehicle_info_list


    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible overtaking or tailgating chances.

            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_info_list = self.get_near_vehicle_info()

        if self.direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, vehicle_info_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, vehicle_info_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, vehicle_info_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=45)

            
            if self.scenario_trigger:
                # Check for overtaking
                if vehicle_state \
                        and not waypoint.is_junction and self.speed > 10 \
                        and self.behavior.overtake_counter == 0 and self.speed > Scene_wo_render.get_vehicle_speed(vehicle) and not self.if_overtake:
                    # print("Overtaking!")
                    self._overtake(waypoint, vehicle_info_list)

                # Check for tailgating

                elif not vehicle_state \
                        and not waypoint.is_junction and self.speed > 10 \
                        and self.behavior.tailgate_counter == 0 and not self.if_overtake:
                    # print("Tailgating!")
                    self._tailgating(waypoint, vehicle_info_list)

        return vehicle_state, vehicle, distance

    def straight_collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible overtaking or tailgating chances.

            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        # if waypoint.is_junction:
        #     return False, None, -1
        vehicle_info_list = self.get_near_vehicle_info()

        vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard_straight(
            waypoint, vehicle_info_list, min(
                self.behavior.min_proximity_threshold / 1.5, self.speed_limit / 3), up_angle_th=10)

        return vehicle_state, vehicle, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        vehicle_speed = Scene_wo_render.get_vehicle_speed(vehicle)
        delta_v = max(1, (self.speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        # print(f'{self.vehicle_name} is following {vehicle}, distance: {distance}, ttc: {ttc}')
        # Under safety time distance, slow down.
        if 2 * self.behavior.safety_time > ttc > 0.0:
            control = self._local_planner.run_step(
                target_speed=vehicle_speed-1.0, debug=debug)
        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 3 * self.behavior.safety_time > ttc >= self.behavior.safety_time:
            control = self._local_planner.run_step(
                target_speed=min(max(self.min_speed, vehicle_speed-1.0),
                                 min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)), debug=debug)
        # Normal behavior.
        else:
            control = self._local_planner.run_step(
                target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)

        return control
    
    def list_vehicles_in_junction(self, waypoint):
        """
        This method is in charge of listing all the vehicles
        in a junction.

            :param waypoint: current waypoint of the agent
            :return vehicle_list: list of all the nearby vehicles
        """
        vehicle_list = []
        for vehicle_name, agent in Scene_wo_render._agent_dict.items():
            if vehicle_name != self.vehicle_name:
                vehicle_wp = Scene_wo_render.get_vehicle_waypoint(vehicle_name)
                if vehicle_wp.is_junction:
                    vehicle_list.append({'name': vehicle_name, 'location': vehicle_wp.transform.location, 'lane_id': vehicle_wp.lane_id})
        return vehicle_list

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        if len(self._local_planner.waypoints_queue) < 1:
            self.end_route_flag = True
            self.cur_control = self.emergency_stop()
            return self.emergency_stop()
        control = None
        if self.behavior.tailgate_counter > 0:
            self.behavior.tailgate_counter -= 1
        if self.behavior.overtake_counter > 0:
            self.behavior.overtake_counter -= 1
        if self.if_overtake:
            if calculate_distance(self.overtake_end_waypoint.transform.location, self.cur_waypoint.transform.location) < 2:
                self.if_overtake = False
                self.overtake_direction = None
                self.overtake_end_waypoint = None
                self.vehicle_name_to_overtake = None
            # print(f'{self.vehicle_name} is overtaking', self.behavior.overtake_counter)
            if self.if_overtake and self.behavior.overtake_counter % 40 == 0:
                ego_loc = self.cur_waypoint.transform.location
                ego_yaw = self.cur_waypoint.transform.rotation[2]
                if not self.vehicle_name_to_overtake is None and self.vehicle_name_to_overtake in Scene_wo_render._agent_dict:
                    target_vehicle_wp = Scene_wo_render.get_vehicle_waypoint(self.vehicle_name_to_overtake)
                    target_vehicle_loc = target_vehicle_wp.transform.location
                    target_vehicle_yaw = target_vehicle_wp.transform.rotation[2]
                    target_vehicle_lane_id = target_vehicle_wp.lane_id
                    target_vehicle_exit_lane_id = Scene_wo_render._agent_dict[self.vehicle_name_to_overtake].get_next_lane_id()
                    vehicle_list = [{'name': self.vehicle_name_to_overtake, 'location': target_vehicle_loc, 'yaw': target_vehicle_yaw, 'lane_id': target_vehicle_lane_id, 'exit_lane_id': target_vehicle_exit_lane_id}]
                    overtake_vehicle_state, _, _ = self._bh_is_vehicle_hazard(self.cur_waypoint, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180,low_angle_th=90)
                    if not overtake_vehicle_state:
                        if self.overtake_direction == 'left':
                            driving_mode = 'CHANGELANERIGHT'
                        else:
                            driving_mode = 'CHANGELANELEFT'
                        self._force_turn_back()
                    control = self._local_planner.run_step(
                        target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)
                    self.cur_control = control
                    return control

        ego_vehicle_wp = Scene_wo_render.get_vehicle_waypoint(self.vehicle_name)

        vehicle_state, vehicle_name, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        straight_vehicle_state, straight_vehicle_name, straight_distance = self.straight_collision_and_car_avoid_manager(ego_vehicle_wp)
        if straight_vehicle_state:
            if 'static' in straight_vehicle_name:
                cur_plan = list(self._local_planner._waypoint_buffer)+list(self._local_planner.waypoints_queue)
                if 'ego' in self.vehicle_name:
                    new_path = Scene_wo_render.refine_route_w_static(cur_plan, self.bounding_box, if_ego=True)
                else:
                    new_path = Scene_wo_render.refine_route_w_static(cur_plan, self.bounding_box)
                self.reroute_all(new_path)
                control = self._local_planner.run_step(
                        target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)
                self.cur_control = control
                return control
            else:
                target_vehicle_bbox = Scene_wo_render.get_vehicle_bbox(straight_vehicle_name)    
                straight_distance = straight_distance - max(
                    target_vehicle_bbox[1]/2, target_vehicle_bbox[0]/2) - max(
                        self.bounding_box[1]/2, self.bounding_box[0]/2)
                if straight_distance < self.behavior.braking_distance:
                    # control = self.emergency_stop()
                    control = self._local_planner.run_step(
                            target_speed=0, debug=debug)
                    if control.brake < 0.6:
                        control.brake = 0.6
                    self.cur_control = control
                    return control
                else:
                    control = self.car_following_manager(straight_vehicle_name, straight_distance)
                    self.cur_control = control
                    return control

        if vehicle_state and 'static' not in vehicle_name:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            # Emergency brake if the car is very close.
            target_vehicle_bbox = Scene_wo_render.get_vehicle_bbox(vehicle_name)    
            distance_brake = distance - max(
                target_vehicle_bbox[1]/2, target_vehicle_bbox[0]/2) - max(
                    self.bounding_box[1]/2, self.bounding_box[0]/2)
                
            if distance_brake < self.behavior.braking_distance:
                # print(f'{self.vehicle_name} is too close to {vehicle_name}, distance: {distance_brake}')
                # contorl = self.emergency_stop()
                control = self._local_planner.run_step(
                        target_speed=0, debug=debug)
                if control.brake < 0.6:
                    control.brake = 0.6
                self.cur_control = control
                return control 
            else:
                if self.if_overtake:
                    control = self._local_planner.run_step(
                        target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)
                else:
                    control = self.car_following_manager(vehicle_name, distance_brake)
        elif vehicle_state and 'static' in vehicle_name:
            cur_plan = list(self._local_planner._waypoint_buffer)+list(self._local_planner.waypoints_queue)
            if 'ego' in self.vehicle_name:
                new_path = Scene_wo_render.refine_route_w_static(cur_plan, self.bounding_box, if_ego=True)
            else:
                new_path = Scene_wo_render.refine_route_w_static(cur_plan, self.bounding_box)
            self.reroute_all(new_path)
            control = self._local_planner.run_step(
                        target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)
        # 4: Intersection behavior
        # Checking if there's a junction nearby to slow down
        elif self.incoming_waypoint.is_junction or self.cur_waypoint.is_junction:
            # if '5' in self.vehicle_name:
            #     print(f'{self.vehicle_name} is in junction')
            control = self._local_planner.run_step(
                target_speed=min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist - 2), debug=debug)

        # 5: Normal behavior
        # Calculate controller based on no turn, traffic light or vehicle in front
        else:
            # if '5' in self.vehicle_name:
            #     print(f'{self.vehicle_name} is Normal behavior')
            #     print(min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist))
            control = self._local_planner.run_step(
                target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)
        # if '5' in self.vehicle_name:
        #     print(f'{self.vehicle_name} control: {control.steer}, {control.throttle}, {control.brake}')
        self.cur_control = control
        # if 'ego' in self.vehicle_name:
        #     print(f'{self.vehicle_name} control: {control.steer}, {control.throttle}, {control.brake}')
        return control
