import enum
import json
import math
import numpy as np
import sys
import os

from sklearn import neighbors
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from map_utils import *
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree
from navigation.waypoint import Waypoint
from navigation.local_planner_behavior import RoadOption
from navigation.tools.misc import calculate_rotation, calculate_distance
import datetime
from tqdm import tqdm
def convert_str2RoadOption(str):
    if str == 'LANEFOLLOW':
        return RoadOption.LANEFOLLOW
    elif str == 'TURNLEFT':
        return RoadOption.TURNLEFT
    elif str == 'TURNRIGHT':
        return RoadOption.TURNRIGHT
    elif str == 'STRAIGHT':
        return RoadOption.STRAIGHT
    elif str == 'CHANGELANELEFT':
        return RoadOption.CHANGELANELEFT
    elif str == 'CHANGELANERIGHT':
        return RoadOption.CHANGELANERIGHT
    else:
        return RoadOption.VOID

def remove_close_points(points, threshold=2.5):
    points = np.array(points)
    tree = cKDTree(points)
    to_keep = []
    for i, point in enumerate(points):
        if all(tree.query(point, k=2)[0][1] >= threshold for j in to_keep):
            to_keep.append(i)
    return points[to_keep].tolist()

class MapFeature:
    def __init__(self, feature_id, feature_data):
        self.feature_id = feature_id
        self.feature_type = feature_data['feature_type']
        self.polyline = feature_data.get('polyline', [])
        self.is_junction = feature_data.get('interpolating', False)
        self.road_edge_type = feature_data.get('road_edge_type', None)
        self.lane_type = feature_data.get('lane_type', None)
        self.speed_limit_mph = feature_data.get('speed_limit_mph', None)
        self.entry_lanes = feature_data.get('entry_lanes', [])
        self.exit_lanes = feature_data.get('exit_lanes', [])
        self.left_neighbors = feature_data.get('left_neighbors', [])
        self.right_neighbors = feature_data.get('right_neighbors', [])
        self.neighbor_relations = {
            "CHANGELANELEFT": [],
            "CHANGELANERIGHT": [],
            "LANEFOLLOW": [],
            "TURNLEFT": [],
            "TURNRIGHT": [],
        }
        
class Map:
    def __init__(self, map_data_path):
        self.map_data_path = map_data_path
        self.scene_path = os.path.dirname(map_data_path)
        with open(map_data_path, 'r') as f:
            map_data = json.load(f)
        self.graph = nx.DiGraph()
        self.features = {}
        self.lane_points = []
        self.lane_indices = []
        for feature_id, feature_data in map_data.items():
            # judge if empty dict
            if not feature_data:
                continue
            feature = MapFeature(feature_id, feature_data)
            self.features[feature_id] = feature
            if feature.feature_type == 'lane':
                if feature.lane_type == 1 or feature.lane_type == 2:
                    for i, point in enumerate(feature.polyline):
                        self.lane_points.append(point)
                        self.lane_indices.append((feature_id, i))
        self.kd_tree = KDTree(self.lane_points)
        self.spawn_points = None
        self.spawn_distance = 10
        self.refine_relation()
        # self.refine_junction()
        self.build_graph()
        self.generate_spawn_points()
        self.refine_spawn_points()

    def generate_spawn_points(self):
        """
        遍历地图中的lane,按照一定距离生成生成可部署点
        """
        spawn_points = []
        for feature_id, feature in self.features.items():
            if feature_id in ['98', '54']:
                continue
            if feature.feature_type == 'lane':
                last_spawn_point = None
                cur_spawn_point = None
                for i in range(0, len(feature.polyline)-1, 20):
                    cur_spawn_point = feature.polyline[i]
                    if last_spawn_point is not None:
                        distance = calculate_distance(last_spawn_point, cur_spawn_point)
                        if distance > self.spawn_distance:
                            spawn_points.append(cur_spawn_point)
                            last_spawn_point = cur_spawn_point
                    else:
                        spawn_points.append(cur_spawn_point)
                        last_spawn_point = cur_spawn_point
        self.spawn_points = spawn_points

    def refine_spawn_points(self, min_plan_length=50, ego_init_point=None,distance_thre=3):
        """
        生成可部署点后,删除其中可生成路径过短的点
        """
        # print(len(self.spawn_points))
        # 删除与任意其他可部署点距离小于2.5m的点
        spawn_points_new = remove_close_points(self.spawn_points.copy(), distance_thre)
        if ego_init_point is not None:
            # 删除所有在ego初始点附近的可部署点
            ego_init_point = np.array(ego_init_point)
            spawn_points_new = [point for point in spawn_points_new if calculate_distance(point, ego_init_point) > distance_thre]
        self.spawn_points = spawn_points_new
        spawn_points_new = self.spawn_points.copy()
        # print(len(spawn_points_new))
        for spawn_point in self.spawn_points:
            plan_waypoints = self.generate_overall_plan_waypoints(spawn_point, driving_mode='Random', ignore_lanechange=True)
            if plan_waypoints is None or len(plan_waypoints) < min_plan_length:
                spawn_points_new.remove(spawn_point)
        # print(len(spawn_points_new))
        self.spawn_points = spawn_points_new

    def refine_spawn_points_w_location(self, location, distance_thre=2.0):
        """
        生成可部署点后,删除其中可生成路径过短的点
        """
        # 删除所有在ego初始点附近的可部署点
        location = np.array(location)
        spawn_points_new = self.spawn_points.copy()
        spawn_points_new = [point for point in spawn_points_new if calculate_distance(point, location) > distance_thre]
        self.spawn_points = spawn_points_new

    def get_spawn_points(self):
        """
        获取初始化后的可部署点
        """
        return self.spawn_points
                    
    def find_nearest_lane_point(self, query_point):
        """
        查询最近的车道点

        Args:
            query_point: 查询点的坐标[x, y, z]

        Returns:
            lane_id: 车道ID
            point_index: 车道点的索引
            distance: 查询点与车道点的距离
        """
        distance, index = self.kd_tree.query(query_point[:3])  # 使用KDTree查询最近点
        lane_id, point_index = self.lane_indices[index]
        return lane_id, point_index, distance

    def get_left_neighbor_info(self, cur_lane_id, left_lane_id):
        """
        获取左侧车道的信息

        Args:
            cur_lane_id: 当前车道ID
            left_lane_id: 左侧车道ID

        Returns:
            neighbor_info: 左侧车道信息
        """
        return [x for x in self.features[cur_lane_id].left_neighbors if x['feature_id'] == left_lane_id][0]

    def get_right_neighbor_info(self, cur_lane_id, right_lane_id):
        """
        获取右侧车道的信息

        Args:
            cur_lane_id: 当前车道ID
            right_lane_id: 右侧车道ID

        Returns:
            neighbor_info: 右侧车道信息
        """
        return [x for x in self.features[cur_lane_id].right_neighbors if x['feature_id'] == right_lane_id][0]

    def get_left_neighbor_lane_ids(self, lane_id):
        """
        获取左侧车道的ID

        Args:
            lane_id: 车道ID

        Returns:
            left_lane_id: 左侧车道ID
        """
        return [x['feature_id'] for x in self.features[lane_id].left_neighbors]

    def get_right_neighbor_lane_ids(self, lane_id):
        """
        获取右侧车道的ID

        Args:
            lane_id: 车道ID

        Returns:
            right_lane_id: 右侧车道ID
        """
        return [x['feature_id'] for x in self.features[lane_id].right_neighbors]

    def get_all_neighbor_lane_ids(self, lane_id):
        """
        获取所有邻近车道的ID

        Args:
            lane_id: 车道ID

        Returns:
            neighbor_lane_ids: 邻近车道ID列表
        """
        neighbor_lane_ids = []
        for neighbor in self.features[lane_id].left_neighbors:
            neighbor_lane_ids.append(neighbor['feature_id'])
        for neighbor in self.features[lane_id].right_neighbors:
            neighbor_lane_ids.append(neighbor['feature_id'])
        for neighbor_relation in self.features[lane_id].neighbor_relations:
            neighbor_lane_ids += self.features[lane_id].neighbor_relations[neighbor_relation]
        return neighbor_lane_ids
    
    def get_all_junction_neighbor_lane_ids(self, lane_id):
        """
        获取所有邻近路口车道的ID

        Args:
            lane_id: 车道ID

        Returns:
            neighbor_lane_ids: 邻近路口车道ID列表
        """
        neighbor_lane_ids = []
        for neighbor in self.features[lane_id].left_neighbors:
            if self.features[neighbor['feature_id']].is_junction:
                neighbor_lane_ids = neighbor_lane_ids + [self.get_all_neighbor_lane_ids(neighbor['feature_id'])]
        for neighbor in self.features[lane_id].right_neighbors:
            if self.features[neighbor['feature_id']].is_junction:
                neighbor_lane_ids = neighbor_lane_ids + [self.get_all_neighbor_lane_ids(neighbor['feature_id'])]
        for neighbor_relation in self.features[lane_id].neighbor_relations:
            for neighbor_id in self.features[lane_id].neighbor_relations[neighbor_relation]:
                if self.features[neighbor_id].is_junction:
                    neighbor_lane_ids = neighbor_lane_ids + [self.get_all_neighbor_lane_ids(neighbor['feature_id'])]
        return neighbor_lane_ids

    def build_waypoint_config(self, location, rotation, road_option=None, lane_id=None, lane_point_idx=None):
        if lane_id is None or lane_point_idx is None:
            cur_lane_id, cur_lane_point_idx, _ = self.find_nearest_lane_point(location)
            lane_id = cur_lane_id
            lane_point_idx = cur_lane_point_idx
        if road_option is None:
            road_option = RoadOption.VOID
        lane_feature = self.features[lane_id]
        is_junction = lane_feature.is_junction
        left_lane_ids = lane_feature.neighbor_relations['CHANGELANELEFT']
        left_lane_valid = []
        for left_lane_id in left_lane_ids:
            neighbor_info = self.get_left_neighbor_info(lane_id, left_lane_id)
            self_start_idx = neighbor_info['self_start_index']
            self_end_idx = neighbor_info['self_end_index']
            if lane_point_idx >= self_start_idx and lane_point_idx <= self_end_idx - 2:
                left_lane_valid.append(left_lane_id)
        right_lane_ids = lane_feature.neighbor_relations['CHANGELANERIGHT']
        right_lane_valid = []
        for right_lane_id in right_lane_ids:
            neighbor_info = self.get_right_neighbor_info(lane_id, right_lane_id)
            self_start_idx = neighbor_info['self_start_index']
            self_end_idx = neighbor_info['self_end_index']
            if lane_point_idx >= self_start_idx and lane_point_idx <= self_end_idx - 2:
                right_lane_valid.append(right_lane_id)
        return {
            'transform': {
                'location': location,
                'rotation': rotation
            },
            'road_option': road_option,
            'lane_id': lane_id,
            'lane_point_idx': lane_point_idx,
            'is_junction': is_junction,
            'left_lane': left_lane_valid,
            'right_lane': right_lane_valid
        }

    def build_waypoint(self, waypoint_config):
        return Waypoint(waypoint_config)

    def get_location_by_lane_point(self, lane_point):
        """
        通过车道点获取坐标

        Args:
            lane_point: 车道点信息(lane_id, point_index)

        Returns:
            point: 车道点坐标[x, y, z]
        """
        lane_id, point_index = lane_point
        return self.features[lane_id].polyline[point_index]

    def check_lanechange(self, current_lane_id, current_point_index, roadoption):
        """
        检查是否可以变道

        Args:
            current_lane_id: 当前车道ID
            target_lane_id: 目标车道ID
            current_point_index: 当前车道点索引
            roadoption: 驾驶模式

        Returns:
            result: 是否可以变道
        """
        if roadoption == 'CHANGELANELEFT':
            if self.features[current_lane_id].neighbor_relations['CHANGELANELEFT']:
                for target_lane_id in self.features[current_lane_id].neighbor_relations['CHANGELANELEFT']:
                    self_start_index = [x['self_start_index'] for x in self.features[current_lane_id].left_neighbors if x['feature_id'] == target_lane_id][0]
                    self_end_index = [x['self_end_index'] for x in self.features[current_lane_id].left_neighbors if x['feature_id'] == target_lane_id][0]
                    if current_point_index < self_end_index:
                        return True
                    else:
                        return False
            else:
                return False
        elif roadoption == 'CHANGELANERIGHT':
            if self.features[current_lane_id].neighbor_relations['CHANGELANERIGHT']:
                for target_lane_id in self.features[current_lane_id].neighbor_relations['CHANGELANERIGHT']:
                    self_start_index = [x['self_start_index'] for x in self.features[current_lane_id].right_neighbors if x['feature_id'] == target_lane_id][0]
                    self_end_index = [x['self_end_index'] for x in self.features[current_lane_id].right_neighbors if x['feature_id'] == target_lane_id][0]
                    if current_point_index < self_end_index:
                        return True
                    else:
                        return False
        else:
            return False
        
    def refine_plan_waypoints(self, plan_waypoints, min_distance=4.5):
        """
        重新规划路径点,删除过于接近的点

        Args:
            plan_waypoints: 规划的路径点
            min_distance: 最小距离

        Returns:
            plan_waypoints: 重新规划的路径点
        """
        plan_waypoints_new = []
        lane_change_point_flag = False
        for i, waypoint in enumerate(plan_waypoints):
            if lane_change_point_flag:
                plan_waypoints_new.append(waypoint)
                lane_change_point_flag = False
                continue
            if i == 0:
                plan_waypoints_new.append(waypoint)
            else:
                if waypoint.road_option != RoadOption.STRAIGHT:
                    if waypoint.road_option == RoadOption.CHANGELANELEFT or waypoint.road_option == RoadOption.CHANGELANERIGHT:
                        plan_waypoints_new.append(waypoint)
                        lane_change_point_flag = True
                    else:
                        plan_waypoints_new.append(waypoint)
                else:
                    distance = calculate_distance(plan_waypoints_new[-1].transform.get_location(), waypoint.transform.get_location())
                    if distance >= min_distance:
                        plan_waypoints_new.append(waypoint)
        return plan_waypoints_new

    def get_plan_waypoints(self, current_location, driving_mode='Random',ignore_lanechange=True):
        """
        获取规划的路径点

        Args:
            current_location: 当前路径点[x,y,z]
            driving_mode: 驾驶模式,包括LANEFOLLOW, TURNLEFT, TURNRIGHT, CHANGELANELEFT, CHANGELANERIGHT, Random

        Returns:
            plan_waypoints: 规划的路径点
        """
        current_lane_id, current_point_index, _ = self.find_nearest_lane_point(current_location)
        current_lane = self.features[current_lane_id]
        if driving_mode == 'STRAIGHT':
            if current_point_index == len(current_lane.polyline) - 1:
                return 'STRAIGHT', None
            else:
                plan_waypoints = []
                # append remaining points in current lane
                for i in range(current_point_index, len(current_lane.polyline)-1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i+1])
                    waypoint_config = self.build_waypoint_config(current_lane.polyline[i], rotation, RoadOption.STRAIGHT, current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                # append the last point in current lane
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(current_lane.polyline[-1], rotation, RoadOption.STRAIGHT, current_lane_id, len(current_lane.polyline)-1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                return 'STRAIGHT', plan_waypoints
        elif driving_mode == 'LANEFOLLOW':
            if current_lane.neighbor_relations['LANEFOLLOW']:
                next_lane_id = current_lane.neighbor_relations['LANEFOLLOW'][0]
                next_lane = self.features[next_lane_id]
                plan_waypoints = []
                # append remaining points in current lane
                for i in range(current_point_index, len(current_lane.polyline)-1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i+1])
                    waypoint_config = self.build_waypoint_config(current_lane.polyline[i], rotation, RoadOption.STRAIGHT, current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                # append the last point in current lane
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(current_lane.polyline[-1], rotation, RoadOption.LANEFOLLOW, current_lane_id, len(current_lane.polyline)-1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                # append the first point in next lane
                rotation = calculate_rotation(current_lane.polyline[-1], next_lane.polyline[0])
                waypoint_config = self.build_waypoint_config(next_lane.polyline[0], rotation, RoadOption.STRAIGHT, next_lane_id, 0)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                # append the remaining points in next lane
                for i in range(1, len(next_lane.polyline)):
                    rotation = calculate_rotation(next_lane.polyline[i-1], next_lane.polyline[i])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[i], rotation, RoadOption.STRAIGHT, next_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                return 'LANEFOLLOW', plan_waypoints
            else:
                return 'LANEFOLLOW', None
        elif driving_mode == 'TURNLEFT':
            if current_lane.neighbor_relations['TURNLEFT']:
                next_lane_id = current_lane.neighbor_relations['TURNLEFT'][0]
                next_lane = self.features[next_lane_id]
                plan_waypoints = []
                # append remaining points in current lane, except the last point
                for i in range(current_point_index, len(current_lane.polyline)-1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i+1])
                    waypoint_config = self.build_waypoint_config(current_lane.polyline[i], rotation, RoadOption.STRAIGHT, current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                # append the last point in current lane
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(current_lane.polyline[-1], rotation, RoadOption.TURNLEFT, current_lane_id, len(current_lane.polyline)-1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                # append the first point in next lane
                rotation = calculate_rotation(current_lane.polyline[-1], next_lane.polyline[0])
                waypoint_config = self.build_waypoint_config(next_lane.polyline[0], rotation, RoadOption.STRAIGHT, next_lane_id, 0)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                # append the remaining points in next lane
                for i in range(1, len(next_lane.polyline)):
                    rotation = calculate_rotation(next_lane.polyline[i-1], next_lane.polyline[i])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[i], rotation, RoadOption.STRAIGHT, next_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                return 'TURNLEFT', plan_waypoints
            else:
                return 'TURNLEFT', None
        elif driving_mode == 'TURNRIGHT':
            if current_lane.neighbor_relations['TURNRIGHT']:
                next_lane_id = current_lane.neighbor_relations['TURNRIGHT'][0]
                next_lane = self.features[next_lane_id]
                plan_waypoints = []
                # append remaining points in current lane, except the last point
                for i in range(current_point_index, len(current_lane.polyline)-1):
                    rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i+1])
                    waypoint_config = self.build_waypoint_config(current_lane.polyline[i], rotation, RoadOption.STRAIGHT, current_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                # append the last point in current lane
                rotation = calculate_rotation(current_lane.polyline[-2], current_lane.polyline[-1])
                waypoint_config = self.build_waypoint_config(current_lane.polyline[-1], rotation, RoadOption.TURNRIGHT, current_lane_id, len(current_lane.polyline)-1)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                # append the first point in next lane
                rotation = calculate_rotation(current_lane.polyline[-1], next_lane.polyline[0])
                waypoint_config = self.build_waypoint_config(next_lane.polyline[0], rotation, RoadOption.STRAIGHT, next_lane_id, 0)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
                # append the remaining points in next lane
                for i in range(1, len(next_lane.polyline)):
                    rotation = calculate_rotation(next_lane.polyline[i-1], next_lane.polyline[i])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[i], rotation, RoadOption.STRAIGHT, next_lane_id, i)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                return 'TURNRIGHT', plan_waypoints
            else:
                return 'TURNRIGHT', None
        elif driving_mode == 'CHANGELANELEFT':
            if current_lane.neighbor_relations['CHANGELANELEFT']:
                next_lane_id = current_lane.neighbor_relations['CHANGELANELEFT'][0]
                # get neighbor lane start idx
                neighbor_info = self.get_left_neighbor_info(current_lane_id, next_lane_id)
                neightbor_start_idx = neighbor_info['neighbor_start_index']
                neightbor_end_idx = neighbor_info['neighbor_end_index']
                self_start_idx = neighbor_info['self_start_index']
                self_end_idx = neighbor_info['self_end_index']
                next_lane = self.features[next_lane_id]
                if current_point_index > self_end_idx:
                    return 'CHANGELANELEFT', None
                plan_waypoints = []
                if current_point_index < self_start_idx:
                    for i in range(current_point_index, self_start_idx):
                        rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i+1])
                        waypoint_config = self.build_waypoint_config(current_lane.polyline[i], rotation, RoadOption.STRAIGHT, current_lane_id, i)
                        waypoint = Waypoint(waypoint_config)
                        plan_waypoints.append(waypoint)
                    # append points of self_start_idx
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx-1], current_lane.polyline[self_start_idx])
                    waypoint_config = self.build_waypoint_config(current_lane.polyline[self_start_idx], rotation, RoadOption.CHANGELANELEFT, current_lane_id, self_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    # append points of next lane
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx], next_lane.polyline[neightbor_start_idx])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[neightbor_start_idx], rotation, RoadOption.STRAIGHT, next_lane_id, neightbor_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                elif current_point_index >= self_start_idx and current_point_index < self_end_idx:
                    idx_offset = current_point_index - self_start_idx
                    lanechange_idx = neightbor_start_idx + idx_offset
                    if lanechange_idx + 2 <= neightbor_end_idx:
                        lanechange_idx += 2
                    else:
                        pass
                    # append point of lanechange_idx + 1
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], next_lane.polyline[lanechange_idx-1])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[lanechange_idx-1], rotation, RoadOption.CHANGELANELEFT, next_lane_id, lanechange_idx-1)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    # append point of next lane
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], next_lane.polyline[lanechange_idx])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[lanechange_idx], rotation, RoadOption.STRAIGHT, next_lane_id, lanechange_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                return 'CHANGELANELEFT', plan_waypoints
            else:
                return 'CHANGELANELEFT', None
        elif driving_mode == 'CHANGELANERIGHT':
            if current_lane.neighbor_relations['CHANGELANERIGHT']:
                next_lane_id = current_lane.neighbor_relations['CHANGELANERIGHT'][0]
                next_lane = self.features[next_lane_id]
                # get neighbor lane start idx
                neighbor_info = self.get_right_neighbor_info(current_lane_id, next_lane_id)
                neightbor_start_idx = neighbor_info['neighbor_start_index']
                neightbor_end_idx = neighbor_info['neighbor_end_index']
                self_start_idx = neighbor_info['self_start_index']
                self_end_idx = neighbor_info['self_end_index']
                if current_point_index > self_end_idx:
                    return 'CHANGELANERIGHT', None
                plan_waypoints = []
                if current_point_index < self_start_idx:
                    for i in range(current_point_index, self_start_idx):
                        rotation = calculate_rotation(current_lane.polyline[i], current_lane.polyline[i+1])
                        waypoint_config = self.build_waypoint_config(current_lane.polyline[i], rotation, RoadOption.STRAIGHT, current_lane_id, i)
                        waypoint = Waypoint(waypoint_config)
                        plan_waypoints.append(waypoint)
                    # append points of self_start_idx
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx-1], current_lane.polyline[self_start_idx])
                    waypoint_config = self.build_waypoint_config(current_lane.polyline[self_start_idx], rotation, RoadOption.CHANGELANERIGHT, current_lane_id, self_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    # append points of next lane
                    rotation = calculate_rotation(current_lane.polyline[self_start_idx], next_lane.polyline[neightbor_start_idx])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[neightbor_start_idx], rotation, RoadOption.STRAIGHT, next_lane_id, neightbor_start_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                elif current_point_index >= self_start_idx and current_point_index < self_end_idx:
                    idx_offset = current_point_index - self_start_idx
                    lanechange_idx = neightbor_start_idx + idx_offset
                    if lanechange_idx + 2 <= neightbor_end_idx:
                        lanechange_idx += 2
                    else:
                        pass
                    # append point of lanechange_idx + 1
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], next_lane.polyline[lanechange_idx-1])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[lanechange_idx-1], rotation, RoadOption.CHANGELANERIGHT, next_lane_id, lanechange_idx-1)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                    # append point of next lane
                    rotation = calculate_rotation(current_lane.polyline[current_point_index], next_lane.polyline[lanechange_idx])
                    waypoint_config = self.build_waypoint_config(next_lane.polyline[lanechange_idx], rotation, RoadOption.STRAIGHT, next_lane_id, lanechange_idx)
                    waypoint = Waypoint(waypoint_config)
                    plan_waypoints.append(waypoint)
                return 'CHANGELANERIGHT', plan_waypoints                
            else:
                return 'CHANGELANERIGHT', None
        else:
            # if driving_mode is random, randomly choose a valid driving mode
            valid_driving_modes = []
            lane_change_left_flag = self.check_lanechange(current_lane_id, current_point_index, 'CHANGELANELEFT')
            lane_change_right_flag = self.check_lanechange(current_lane_id, current_point_index, 'CHANGELANERIGHT')
            basic_driving_modes = ['LANEFOLLOW', 'TURNLEFT', 'TURNRIGHT']
            if ignore_lanechange:
                for mode in basic_driving_modes:
                    if self.features[current_lane_id].neighbor_relations[mode]:
                        valid_driving_modes.append(mode)
            else:
                allowed_driving_modes = basic_driving_modes
                if lane_change_left_flag:
                    allowed_driving_modes.append('CHANGELANELEFT')
                if lane_change_right_flag:
                    allowed_driving_modes.append('CHANGELANERIGHT')
                for mode in allowed_driving_modes:
                    if self.features[current_lane_id].neighbor_relations[mode]:
                        valid_driving_modes.append(mode)
            if valid_driving_modes:
                driving_mode = np.random.choice(valid_driving_modes)
                return self.get_plan_waypoints(current_location, driving_mode)
            else:
                driving_mode, plan_waypoints = self.get_plan_waypoints(current_location, 'STRAIGHT')
                if plan_waypoints is not None:
                    return 'STRAIGHT', plan_waypoints
                else:
                    return driving_mode, None

    def get_plan_waypoints_w_refine(self, current_location, driving_mode='Random', ignore_lanechange=True):
        """
        获取规划的路径点,并进行简化

        Args:
            current_location: 当前路径点[x,y,z]
            driving_mode: 驾驶模式,包括LANEFOLLOW, TURNLEFT, TURNRIGHT, CHANGELANELEFT, CHANGELANERIGHT, Random

        Returns:
            plan_waypoints: 规划的路径点
        """
        driving_mode, plan_waypoints = self.get_plan_waypoints(current_location, driving_mode, ignore_lanechange)
        if plan_waypoints is not None:
            plan_waypoints = self.refine_plan_waypoints(plan_waypoints)
        else:
            'INVALID', None
        return driving_mode, plan_waypoints

    def generate_overall_plan_waypoints(self, current_location, driving_mode='Random', ignore_lanechange=True, max_plan_length=5000, min_plan_length=200):
        """
        不断地根据driving mode生成规划的路径点,如果当前车道无法继续行驶,则随机选择一个驾驶模式

        Args:
            current_location: 当前路径点[x,y,z]
            driving_mode: 驾驶模式,包括LANEFOLLOW, TURNLEFT, TURNRIGHT, CHANGELANELEFT, CHANGELANERIGHT, Random

        Returns:
            plan_waypoints: 规划的路径点
        """
        plan_waypoints = None
        flag_first_plan = True
        while True:
            if flag_first_plan:
                driving_mode, new_plan = self.get_plan_waypoints(current_location, driving_mode,ignore_lanechange=ignore_lanechange)
                flag_first_plan = False
                plan_waypoints = new_plan
                if (driving_mode == 'CHANGELANELEFT' or driving_mode == 'CHANGELANERIGHT') and plan_waypoints is None:
                    driving_mode, new_plan = self.get_plan_waypoints(current_location, 'LANEFOLLOW',ignore_lanechange=ignore_lanechange)
                    plan_waypoints = new_plan
                if plan_waypoints is None:
                    break
            else:
                driving_mode, new_plan = self.get_plan_waypoints(plan_waypoints[-1].transform.get_location(), driving_mode='Random',ignore_lanechange=ignore_lanechange)
                if new_plan is None:
                    break
                last_point_in_current_plan = plan_waypoints[-1]
                first_point_in_new_plan = new_plan[0]
                if last_point_in_current_plan.lane_id != first_point_in_new_plan.lane_id:
                    plan_waypoints[-1].road_option = driving_mode
                else:
                    pass
                plan_waypoints += new_plan
            if len(plan_waypoints) > max_plan_length or plan_waypoints is None:
                break
        return plan_waypoints

    def generate_overall_plan_waypoints_w_refine(self, current_location, driving_mode='Random', ignore_lanechange=True, max_plan_length=5000, min_plan_length=200):
        """
        不断地根据driving mode生成规划的路径点,如果当前车道无法继续行驶,则随机选择一个驾驶模式

        Args:
            current_location: 当前路径点[x,y,z]
            driving_mode: 驾驶模式,包括LANEFOLLOW, TURNLEFT, TURNRIGHT, CHANGELANELEFT, CHANGELANERIGHT, Random

        Returns:
            plan_waypoints: 规划的路径点
        """
        plan_waypoints = None
        flag_first_plan = True
        while True:
            if flag_first_plan:
                driving_mode, new_plan = self.get_plan_waypoints(current_location, driving_mode,ignore_lanechange=ignore_lanechange)
                flag_first_plan = False
                plan_waypoints = new_plan
                if (driving_mode == 'CHANGELANELEFT' or driving_mode == 'CHANGELANERIGHT') and plan_waypoints is None:
                    driving_mode, new_plan = self.get_plan_waypoints(current_location, 'LANEFOLLOW',ignore_lanechange=ignore_lanechange)
                    plan_waypoints = new_plan
                if plan_waypoints is None:
                    break
            else:
                driving_mode, new_plan = self.get_plan_waypoints(plan_waypoints[-1].transform.get_location(), driving_mode='Random',ignore_lanechange=ignore_lanechange)
                if new_plan is None:
                    break
                last_point_in_current_plan = plan_waypoints[-1]
                first_point_in_new_plan = new_plan[0]
                if last_point_in_current_plan.lane_id != first_point_in_new_plan.lane_id:
                    plan_waypoints[-1].road_option = driving_mode
                else:
                    pass
                plan_waypoints += new_plan
            if len(plan_waypoints) > max_plan_length or plan_waypoints is None:
                break
        plan_waypoints = self.refine_plan_waypoints(plan_waypoints)
        return plan_waypoints

    def build_graph(self):
        """
        构建地图的图结构

        Returns:
            graph: 地图的图结构
        """
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                polyline = feature.polyline
                # 按照相隔至少为0.5m的点为一个节点,构建图
                for i in range(0, len(polyline)-1):
                    start = polyline[i]
                    end = polyline[i + 1]
                    distance = calculate_distance(start, end)
                    self.graph.add_edge((feature_id, i), (feature_id, i + 1), weight=distance)
                # 关联exit lane和entry lane,需要考虑到与前面的点相同的情况
                for exit_lane_id in feature.exit_lanes:
                    if exit_lane_id not in self.features.keys():
                        print(f"Exit lane {exit_lane_id} not found")
                        continue
                    distance = calculate_distance(polyline[-1], self.features[exit_lane_id].polyline[0])
                    if distance < 0.01:
                        idx_offset = 1
                    else:
                        idx_offset = 0
                    if distance > 10:
                        # print(f"distance between {feature_id} and {entry_lane_id} is {distance}")
                        continue
                    self.graph.add_edge((feature_id, len(polyline) - 1 - idx_offset), (exit_lane_id, 0), weight=distance)
                for entry_lane_id in feature.entry_lanes:
                    if entry_lane_id not in self.features.keys():
                        print(f"Entry lane {entry_lane_id} not found")
                        continue
                    distance = calculate_distance(polyline[0], self.features[entry_lane_id].polyline[-1])
                    if distance < 0.01:
                        idx_offset = 1
                    else:
                        idx_offset = 0
                    if distance > 10:
                        # print(f"distance between {feature_id} and {entry_lane_id} is {distance}")
                        continue
                    self.graph.add_edge((entry_lane_id, len(self.features[entry_lane_id].polyline) - 1 - idx_offset), (feature_id, 0), weight=distance)
                # print(feature.neighbor_relations['CHANGELANELEFT'])
                # 关联左右车道
                for neighbor_id in feature.neighbor_relations['CHANGELANELEFT']:
                    if neighbor_id not in self.features.keys():
                        print(f"Left neighbor {neighbor_id} not found")
                        continue
                    neighbor = self.features[neighbor_id]
                    neighbor_info = [x for x in feature.left_neighbors if x['feature_id'] == neighbor_id][0]
                    # print(neighbor_info)
                    neightbor_start_idx = neighbor_info['neighbor_start_index']
                    neightbor_end_idx = neighbor_info['neighbor_end_index']
                    self_start_idx = neighbor_info['self_start_index']
                    self_end_idx = neighbor_info['self_end_index']
                    lane_offset = calculate_distance(polyline[self_start_idx], self.features[neighbor_id].polyline[neightbor_start_idx])
                    l_thre_distance_connect = math.sqrt(1*1 + lane_offset*lane_offset)
                    h_thre_distance_connect = math.sqrt(4*4 + lane_offset*lane_offset)
                    for i in range(self_start_idx, self_end_idx):
                        # last_distance = 0
                        for j in range(neightbor_start_idx, neightbor_end_idx):
                            distance = calculate_distance(polyline[i], neighbor.polyline[j])
                            if distance < l_thre_distance_connect or distance > h_thre_distance_connect:
                                continue
                            # if abs(distance - last_distance) < 0.5:
                            #     continue
                            self.graph.add_edge((feature_id, i), (neighbor_id, j), weight=distance)
                            # last_distance = distance
                            break

                for neighbor_id in feature.neighbor_relations['CHANGELANERIGHT']:
                    if neighbor_id not in self.features.keys():
                        print(f"Right neighbor {neighbor_id} not found")
                        continue
                    neighbor = self.features[neighbor_id]
                    neighbor_info = [x for x in feature.right_neighbors if x['feature_id'] == neighbor_id][0]
                    neightbor_start_idx = neighbor_info['neighbor_start_index']
                    neightbor_end_idx = neighbor_info['neighbor_end_index']
                    self_start_idx = neighbor_info['self_start_index']
                    self_end_idx = neighbor_info['self_end_index']
                    lane_offset = calculate_distance(polyline[self_start_idx], self.features[neighbor_id].polyline[neightbor_start_idx])
                    l_thre_distance_connect = math.sqrt(1 * 1 + lane_offset * lane_offset)
                    h_thre_distance_connect = math.sqrt(4 * 4 + lane_offset * lane_offset)
                    for i in range(self_start_idx, self_end_idx):
                        # last_distance = 0
                        for j in range(neightbor_start_idx, neightbor_end_idx):
                            distance = calculate_distance(polyline[i], neighbor.polyline[j])
                            if distance < l_thre_distance_connect or distance > h_thre_distance_connect:
                                continue
                            # if abs(distance - last_distance) < 0.5:
                            #     continue
                            self.graph.add_edge((feature_id, i), (neighbor_id, j), weight=distance)
                            # last_distance = distance
                            break
        # 删除自环
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))

    def plan_path(self, start_point, end_point):
        """
        规划路径

        Args:
            start_point: 起始点[x, y, z]
            end_point: 终点[x, y, z]

        Returns:
            path: 规划的路径
        """
        start_lane_id, start_idx, _ = self.find_nearest_lane_point(start_point)
        end_lane_id, end_idx, _ = self.find_nearest_lane_point(end_point)
        try:
            path = nx.shortest_path(self.graph, (start_lane_id, start_idx), (end_lane_id, end_idx), weight='weight')
            return path
        except nx.NetworkXNoPath:
            print("No path found")
            return []   

    def judge_roadoption(self, current_waypoint, next_waypoint):
        """
        判断当前车道点与下一个车道点的关系

        Args:
            current_waypoint: 当前车道点
            next_waypoint: 下一个车道点

        Returns:
            road_option: 驾驶模式
        """
        current_lane_id = current_waypoint.lane_id
        current_point_index = current_waypoint.lane_point_idx
        next_lane_id = next_waypoint.lane_id
        next_point_index = next_waypoint.lane_point_idx
        current_lane = self.features[current_lane_id]
        next_lane = self.features[next_lane_id]
        if current_lane_id == next_lane_id:
            return RoadOption.STRAIGHT
        if next_lane_id in current_lane.neighbor_relations['CHANGELANELEFT']:
            return RoadOption.CHANGELANELEFT
        if next_lane_id in current_lane.neighbor_relations['CHANGELANERIGHT']:
            return RoadOption.CHANGELANERIGHT
        if next_lane_id in current_lane.neighbor_relations['LANEFOLLOW']:
            return RoadOption.LANEFOLLOW
        if next_lane_id in current_lane.neighbor_relations['TURNLEFT']:
            return RoadOption.TURNLEFT
        if next_lane_id in current_lane.neighbor_relations['TURNRIGHT']:
            return RoadOption.TURNRIGHT
        return RoadOption.STRAIGHT

    def judge_roadoption_w_lane_id(self, current_lane_id, next_lane_id):
        """
        判断当前车道点与下一个车道点的关系

        Args:
            current_waypoint: 当前车道点
            next_waypoint: 下一个车道点

        Returns:
            road_option: 驾驶模式
        """
        current_lane = self.features[current_lane_id]
        next_lane = self.features[next_lane_id]
        if current_lane_id == next_lane_id:
            return RoadOption.STRAIGHT
        if next_lane_id in current_lane.neighbor_relations['CHANGELANELEFT']:
            return RoadOption.CHANGELANELEFT
        if next_lane_id in current_lane.neighbor_relations['CHANGELANERIGHT']:
            return RoadOption.CHANGELANERIGHT
        if next_lane_id in current_lane.neighbor_relations['LANEFOLLOW']:
            return RoadOption.LANEFOLLOW
        if next_lane_id in current_lane.neighbor_relations['TURNLEFT']:
            return RoadOption.TURNLEFT
        if next_lane_id in current_lane.neighbor_relations['TURNRIGHT']:
            return RoadOption.TURNRIGHT
        return RoadOption.STRAIGHT

    def plan_path_w_waypoints(self, start_waypoint, end_waypoint):
        """
        规划路径

        Args:
            start_point: 起始点[x, y, z]
            end_point: 终点[x, y, z]

        Returns:
            path: 规划的路径
        """
        start_lane_id = start_waypoint.lane_id
        start_idx = start_waypoint.lane_point_idx
        end_lane_id = end_waypoint.lane_id
        end_idx = end_waypoint.lane_point_idx
        try:
            path = nx.shortest_path(self.graph, (start_lane_id, start_idx), (end_lane_id, end_idx), weight='weight')
            plan_waypoints = []
            for idx, node in enumerate(path):
                lane_id, point_index = node
                location = self.features[lane_id].polyline[point_index]
                rotation = calculate_rotation(self.features[lane_id].polyline[point_index-1], self.features[lane_id].polyline[point_index])
                waypoint_config = self.build_waypoint_config(location, rotation, RoadOption.STRAIGHT, lane_id, point_index)
                waypoint = Waypoint(waypoint_config)
                plan_waypoints.append(waypoint)
            for i in range(len(plan_waypoints)-1):
                plan_waypoints[i].road_option = self.judge_roadoption(plan_waypoints[i], plan_waypoints[i+1])  
            if plan_waypoints is not None:
                plan_waypoints = self.refine_plan_waypoints(plan_waypoints)
            return plan_waypoints
        except nx.NetworkXNoPath:
            print("No path found")
            return []

    def generate_waypoint_path_from_transform_path(self, transform_path):
        """
        从transform路径生成waypoint路径

        Args:
            transform_path: transform路径

        Returns:
            waypoint_path: waypoint路径
        """
        waypoint_path = []
        for idx, transform in enumerate(transform_path):
            if idx == len(transform_path) - 1:
                break
            location = transform.get_location()
            rotation = transform.get_rotation()
            cur_lane_id, cur_point_index, _ = self.find_nearest_lane_point(location)
            next_lane_id, next_point_index, _ = self.find_nearest_lane_point(transform_path[idx+1].get_location())
            road_option = self.judge_roadoption_w_lane_id(cur_lane_id, next_lane_id)
            waypoint_config = self.build_waypoint_config(location, rotation, road_option, cur_lane_id, cur_point_index)
            waypoint = Waypoint(waypoint_config)
            waypoint_path.append(waypoint)
        return waypoint_path

    def generate_waypoint_path_from_two_points(self, cur_waypoint, next_waypoint, direction='CHANGELANERIGHT'):
        """
        从town_points生成waypoint路径

        Args:
            cur_waypoint: 当前waypoint
            next_waypoint: 下一个waypoint

        Returns:
            waypoint_path: waypoint路径
        """
        cur_lane_id = cur_waypoint.lane_id
        cur_point_index = cur_waypoint.lane_point_idx
        next_lane_id = next_waypoint.lane_id
        next_point_index = next_waypoint.lane_point_idx
        roadoption = convert_str2RoadOption(direction)
        first_rotation = calculate_rotation(cur_waypoint.transform.get_location(), self.features[next_lane_id].polyline[next_point_index]) 
        plan_waypoints = []
        first_waypoint_config = self.build_waypoint_config(cur_waypoint.transform.get_location(), first_rotation, roadoption, cur_lane_id, cur_point_index)
        first_waypoint = Waypoint(first_waypoint_config)
        plan_waypoints.append(first_waypoint)
        next_waypoint_config = self.build_waypoint_config(next_waypoint.transform.get_location(), next_waypoint.transform.get_rotation(), RoadOption.STRAIGHT, next_lane_id, next_point_index)
        next_waypoint = Waypoint(next_waypoint_config)
        plan_waypoints.append(next_waypoint)
        return direction, plan_waypoints

    def get_waypoint_w_offset(self, waypoint, offset=1.0, direction=None):
        """
        获取偏移后的waypoint

        Args:
            waypoint: 当前waypoint
            offset: 偏移量
            direction: 方向

        Returns:
            waypoint: 偏移后的waypoint
        """
        location = waypoint.transform.get_location()
        yaw = waypoint.transform.get_rotation()[2]
        lane_id = waypoint.lane_id
        if direction is None:
            if self.features[lane_id].neighbor_relations['CHANGELANELEFT']:
                direction = 'left'
            elif self.features[lane_id].neighbor_relations['CHANGELANERIGHT']:
                direction = 'right'
        else:
            direction = direction
        if direction == 'left':
            yaw += np.pi/2
        else:
            yaw -= np.pi/2
        x = location[0] + offset * math.cos(yaw)
        y = location[1] + offset * math.sin(yaw)
        waypoint_new = Waypoint(self.build_waypoint_config([x, y, location[2]], waypoint.transform.get_rotation(), waypoint.road_option, waypoint.lane_id, waypoint.lane_point_idx))
        return waypoint_new

    def get_close_z(self, location):
        """
        获取最近的z坐标

        Args:
            location: 当前位置[x, y, z]

        Returns:
            z: 最近的z坐标
        """
        lane_id, point_index, _ = self.find_nearest_lane_point(location)
        return self.features[lane_id].polyline[point_index][2]

    def generate_overtake_path_from_reference_path(self, reference_path,direction='left',overtake_offset=1.5):
        """
        从参考路径生成超车路径,超车路径不一定需要满足map定义的路径方向要求,而是根据原始的路径,取得当前路径情况下最长的可变道的直道
        也即从当前点(要求当前点不在路口)取到下一个转弯点(Roadoption不为Straight的点),以这段直道作为参考路径，取其左侧offset为overtake_offset的路径点

        Args:
            reference_path: 参考路径
            overtake_offset: 超车路径的左侧offset

        Returns:
            overtake_path: 超车路径
        """
        def calculate_point_w_offset(location, yaw, offset, direction='left'):
            if direction == 'left':
                yaw = yaw + np.pi/2
            else:
                yaw = yaw - np.pi/2
            x = location[0] + offset * math.cos(yaw)
            y = location[1] + offset * math.sin(yaw)
            return [x, y, location[2]]
        overtake_path = []
        reference_points = []
        if reference_path[0].is_junction:
            # print("In junction, no overtake path found")
            return 'INVALIDPATH', None
        for idx, waypoint in enumerate(reference_path):
            if idx == len(reference_path) - 1:
                break
            if waypoint.road_option != RoadOption.STRAIGHT:
                break
            reference_points.append(waypoint)
        if len(reference_points) <= 5:
            # print("No overtake path found")
            return 'INVALIDPATH', None
        # add first waypoint
        first_waypoint = reference_points[0]
        next_waypoint = reference_points[1]
        next_location_w_offset = calculate_point_w_offset(next_waypoint.transform.get_location(), next_waypoint.transform.get_rotation()[2], overtake_offset, direction)
        location = first_waypoint.transform.get_location()
        rotation = calculate_rotation(location, next_location_w_offset)
        first_waypoint_config = self.build_waypoint_config(location, rotation, RoadOption.CHANGELANELEFT, first_waypoint.lane_id, first_waypoint.lane_point_idx)
        first_waypoint = Waypoint(first_waypoint_config)
        overtake_path.append(waypoint)
        for idx,cur_waypoint in enumerate(reference_points):
            if idx == len(reference_points) - 2:
                break
            if idx == 0:
                continue
            cur_location_w_offset = calculate_point_w_offset(cur_waypoint.transform.get_location(), cur_waypoint.transform.get_rotation()[2],overtake_offset, direction)
            next_location_w_offset = calculate_point_w_offset(reference_points[idx+1].transform.get_location(), reference_points[idx+1].transform.get_rotation()[2], overtake_offset, direction)
            cur_rotation = calculate_rotation(cur_location_w_offset, next_location_w_offset)
            cur_waypoint_config = self.build_waypoint_config(cur_location_w_offset, cur_rotation, RoadOption.STRAIGHT, cur_waypoint.lane_id, cur_waypoint.lane_point_idx)
            waypoint_offset = Waypoint(cur_waypoint_config)
            overtake_path.append(waypoint_offset)
        # add last two waypoint
        last_2_waypoint = reference_points[-2]
        last_waypoint = reference_points[-1]
        last_2_location_w_offset = calculate_point_w_offset(last_2_waypoint.transform.get_location(), last_2_waypoint.transform.get_rotation()[2],overtake_offset, direction)
        last_2_rotation = calculate_rotation(last_2_location_w_offset, last_waypoint.transform.get_location())
        waypoint_config = self.build_waypoint_config(last_2_location_w_offset, last_2_rotation, RoadOption.CHANGELANERIGHT, last_2_waypoint.lane_id, last_2_waypoint.lane_point_idx)
        waypoint = Waypoint(waypoint_config)
        overtake_path.append(waypoint)
        overtake_path.append(reference_points[-1])
        return 'OVERTAKE', overtake_path


    def generate_turn_around_path(self,current_waypoint,turn_around_offset=2.5):
        """
        生成掉头路径

        Args:
            current_waypoint: 当前waypoint
            turn_around_offset: 掉头路径的左侧offset

        Returns:
            turn_around_path: 掉头路径
        """
        def calculate_point_w_offset(location, yaw, offset, direction='left'):
            if direction == 'left':
                yaw = yaw + np.pi/2
            else:
                yaw = yaw - np.pi/2
            x = location[0] + offset * math.cos(yaw)
            y = location[1] + offset * math.sin(yaw)
            return [x, y, location[2]]
        turn_around_path = []
        location = current_waypoint.transform.get_location()
        rotation = current_waypoint.transform.get_rotation()
        # 得到路径方向相反的点
        # 首先确认当前所在路径没有左邻居
        current_lane_id = current_waypoint.lane_id
        current_point_index = current_waypoint.lane_point_idx
        current_lane = self.features[current_lane_id]
        if current_lane.left_neighbors:
            # print("No left neighbor found")
            return 'INVALIDPATH', None
        # 得到路径id不同的点
        next_lane_id = None
        next_location_w_offset = location
        while True:
            next_location_w_offset = calculate_point_w_offset(next_location_w_offset, rotation[2], turn_around_offset, 'left')
            next_lane_id, next_point_index, _ = self.find_nearest_lane_point(next_location_w_offset)
            if next_lane_id != current_lane_id:
                break
        rotation = calculate_rotation(location, next_location_w_offset)
        waypoint_config = self.build_waypoint_config(location, rotation, RoadOption.CHANGELANELEFT, current_waypoint.lane_id, current_waypoint.lane_point_idx)
        waypoint = Waypoint(waypoint_config)
        turn_around_path.append(waypoint)
        # 由新的点开始生成随机路径
        new_route = self.generate_overall_plan_waypoints_w_refine(next_location_w_offset)
        turn_around_path = turn_around_path + new_route
        return 'TURNAROUND', turn_around_path

    def visualize_graph(self):
        pos = {}
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                for i, point in enumerate(feature.polyline):
                    pos[(feature_id, i)] = (point[0], point[1])
        
        plt.figure(figsize=(24, 16),dpi=200)
        # 自动忽略无法绘制的节点
        nx.draw(self.graph, pos, with_labels=False, node_size=10, font_size=8)

        plt.savefig('graph.png')

    def refine_relation(self):
        """
        为地图中的路径添加相对关系
        """
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                exit_lanes = []
                # TURNLEFT/TURNRIGHT/LANEFOLLOW
                self_lane_end_direction = calculate_direction(feature.polyline[-4:])
                for exit_lane_id in feature.exit_lanes:
                    if exit_lane_id not in self.features.keys():
                        print(f"Exit lane {exit_lane_id} not found")
                        continue
                    exit_feature = self.features[exit_lane_id]
                    if calculate_distance(feature.polyline[-1], exit_feature.polyline[0]) > 10:
                        continue
                    exit_direction = calculate_direction(exit_feature.polyline[-4:])
                    relation = calculate_turn_relation(self_lane_end_direction, exit_direction)
                    feature.neighbor_relations[relation].append(exit_lane_id)
                for neighbor in feature.left_neighbors:
                    lane_id = neighbor['feature_id']
                    if lane_id not in self.features.keys():
                        print(f"Left neighbor {lane_id} not found")
                        continue
                    neighbor_feature = self.features[lane_id]
                    self_start_index = neighbor['self_start_index']
                    self_end_index = neighbor['self_end_index']
                    neighbor_start_index = neighbor['neighbor_start_index']
                    neighbor_end_index = neighbor['neighbor_end_index']
                    neighbor_length = neighbor_end_index - neighbor_start_index
                    if neighbor_feature.feature_type == 'lane':
                        if neighbor_length < 20:
                            pass
                        else:
                            feature.neighbor_relations['CHANGELANELEFT'].append(lane_id)
                    elif neighbor_feature.feature_type == 'road_edge':
                        pass
                    else:
                        pass
                for neighbor in feature.right_neighbors:
                    lane_id = neighbor['feature_id']
                    if lane_id not in self.features.keys():
                        print(f"Right neighbor {lane_id} not found")
                        continue
                    neighbor_feature = self.features[lane_id]
                    
                    self_start_index = neighbor['self_start_index']
                    self_end_index = neighbor['self_end_index']
                    neighbor_start_index = neighbor['neighbor_start_index']
                    neighbor_end_index = neighbor['neighbor_end_index']
                    neighbor_length = neighbor_end_index - neighbor_start_index
                    if neighbor_feature.feature_type == 'lane':
                        if neighbor_length < 20:
                            pass
                        else:
                            feature.neighbor_relations['CHANGELANERIGHT'].append(lane_id)
                    elif neighbor_feature.feature_type == 'road_edge':
                        pass
                    else:
                        pass
            else:
                pass
        else:
            pass

    def refine_junction(self):
        """
        为地图中的路径添加路口信息
        """
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                exact_junction = False
                if feature.is_junction:
                    neighbours = feature.left_neighbors + feature.right_neighbors
                    for neighbor in neighbours:
                        lane_id = neighbor['feature_id']
                        if lane_id not in self.features.keys():
                            continue
                        neighbor_feature = self.features[lane_id]
                        if neighbor_feature.is_junction:
                            exact_junction = True
                    if exact_junction:
                        for neighbor in neighbours:
                            lane_id = neighbor['feature_id']
                            if lane_id not in self.features.keys():
                                print(f"Junction lane {lane_id} not found")
                                continue
                            neighbor_feature = self.features[lane_id]
                            if neighbor_feature.is_junction:
                                continue
                            else:
                                self.features[lane_id].is_junction = True

    def save_map_convertion(self, save_path):
        """
        保存地图的转换信息

        Args:
            save_path: 保存路径
        """
        road_edges = []
        crosswalk = []
        road_line = []
        map_feature_path = self.map_data_path
        ego_pose_path = os.path.join(self.scene_path,'ego_pose.json')
        with open(map_feature_path, 'r') as f:
            map_feature = json.load(f)
        with open(ego_pose_path, 'r') as f:
            ego_pose = json.load(f)
        ego_z = []
        ego_xy = []
        for ego_pose_tmp in ego_pose:
            ego_xy.append(ego_pose_tmp['location'][:2])
            ego_z.append(ego_pose_tmp['location'][2])
        min_ego_z = min(ego_z) - 5.0
        max_ego_z = max(ego_z) + 5.0
        for lane_id, lane_info in map_feature.items():
            if 'road_edge_type' not in lane_info:
                continue
            lane_new = []
            if lane_info['feature_type'] == 'road_edge':
                for point in lane_info['polyline']:
                    if point[2] < min_ego_z or point[2] > max_ego_z:
                        continue
                    lane_new.append(point)
                if len(lane_new) > 0:
                    road_edges.append(lane_new)

        for lane_id, lane_info in map_feature.items():
            lane_new = []
            if not 'feature_type' in lane_info.keys():
                continue
            if lane_info['feature_type'] == 'crosswalk':
                for point in lane_info['polyline']:
                    lane_new.append(point)
                lane_new.append(lane_info['polyline'][0])
                if len(lane_new) > 0:
                    crosswalk.append(lane_new)

        for lane_id, lane_info in map_feature.items():
            lane_new = []
            if not 'feature_type' in lane_info.keys():
                continue
            if lane_info['feature_type'] == 'road_line':
                for point in lane_info['polyline']:
                    if point[2] < min_ego_z or point[2] > max_ego_z:
                        continue
                    lane_new.append(point)
                if len(lane_new) > 0:
                    road_line.append(lane_new)
        save_file_name = os.path.join(save_path, 'map_feature.json')
        map_feature_new = {}
        map_feature_new['road_edges'] = road_edges
        map_feature_new['crosswalk'] = crosswalk
        map_feature_new['road_line'] = road_line
        with open(save_file_name, 'w') as f:
            json.dump(map_feature_new, f, indent=2)

    def draw_map(self):
        plt.figure(figsize=(24, 16), dpi=200)
        road_edges = []
        lanes = []
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)

        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:,0],edge[:,1],c='red')

        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:,0],lane[:,1],c='green')        
        plt.savefig('/home/ubuntu/sujiaqi/ChatSim/data/end2end_map_data/map.png')
        plt.close()
    
    def draw_map_w_spawn_points(self):
        plt.figure(figsize=(24, 16), dpi=200)
        road_edges = []
        lanes = []
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)

        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:,0],edge[:,1],c='red')

        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:,0],lane[:,1],c='green') 

        spawn_points = self.get_spawn_points()
        for point in spawn_points:
            plt.scatter(point[0], point[1], c='blue', s=10)
        plt.savefig('data/draw_pic/map_w_spawn_points.png')
        plt.close()
     
    def draw_map_w_traffic_flow(self,car_dict,text=''):
        save_folder = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/traffic_flow' + text
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.figure(figsize=(24, 16), dpi=200)
        road_edges = []
        lanes = []
        timestamp = time.time()
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
                # pass
            else:
                road_edges.append(feature.polyline)

        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:,0],edge[:,1],c='red')

        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:,0],lane[:,1],c='green')

        # draw cars with their 2D bbox
        for car_name, car_info in car_dict.items():
            car_centre_loc = car_info['loc'] # [x,y,z]
            car_rot = car_info['rot'][2] # yaw
            car_bbox = car_info['bbox'] # [w,l]
            car_bbox = np.array(car_bbox)
            w = car_bbox[0]
            l = car_bbox[1]
            car_bbox[0] = l
            car_bbox[1] = w
            car_bbox = car_bbox / 2
            car_bbox = np.array([[car_bbox[0],car_bbox[1]],[-car_bbox[0],car_bbox[1]],[-car_bbox[0],-car_bbox[1]],[car_bbox[0],-car_bbox[1]],[car_bbox[0],car_bbox[1]]])
            car_bbox = np.dot(np.array([[np.cos(car_rot),-np.sin(car_rot)],[np.sin(car_rot),np.cos(car_rot)]]),car_bbox.T).T
            car_bbox = car_bbox + np.array(car_centre_loc[:2])
            plt.plot(car_bbox[:,0],car_bbox[:,1],c='blue')
            

        plt.savefig(f'{save_folder}/{timestamp}.png')
        plt.close()

    def draw_map_w_traffic_flow_sequence(self,car_dict_sequence,text='',with_id=False,skip_frames=1):
        save_folder = '/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/' + text
        print(f"saving traffic flow to {save_folder}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # save car_dict_sequence in json
        with open(f'{save_folder}/car_dict_sequence.json','w') as f:
            json.dump(car_dict_sequence,f,indent=2)
        # skip frames 

        for idx,car_dict in tqdm(enumerate(car_dict_sequence)):
            if idx % skip_frames != 0:
                continue
            plt.figure(figsize=(24, 16), dpi=100)
            # plt.axis("equal")
            road_edges = []
            lanes = []
            timestamp = time.time()
            for feature_id, feature in self.features.items():
                if feature.feature_type == 'lane':
                    lanes.append(feature.polyline)
                    # pass
                else:
                    road_edges.append(feature.polyline)

            for edge in road_edges:
                edge = np.array(edge)
                plt.plot(edge[:,0],edge[:,1],c='red')

            for lane in lanes:
                lane = np.array(lane)
                plt.plot(lane[:,0],lane[:,1],c='green')

            # draw cars with their 2D bbox
            for car_name, car_info in car_dict.items():
                if car_name == 'ego_vehicle':
                    color = 'red'
                    car_id = 'ego'
                else:
                    car_id = car_name[0] + car_name.split('_')[-1]
                    if car_info['if_overtake']:
                        color = 'brown'
                    elif car_info['if_tailgate']:
                        color = 'orange'
                    elif car_info['if_static']:
                        color = 'black'
                    else:
                        color = 'blue'
                car_centre_loc = car_info['loc'] # [x,y,z]
                car_rot = car_info['rot'][2] # yaw
                car_bbox = car_info['bbox'] # [w,l]
                car_bbox = np.array(car_bbox)
                w = car_bbox[0]
                l = car_bbox[1]
                car_bbox[0] = l
                car_bbox[1] = w
                car_bbox = car_bbox / 2
                car_bbox = np.array([[car_bbox[0],car_bbox[1]],[-car_bbox[0],car_bbox[1]],[-car_bbox[0],-car_bbox[1]],[car_bbox[0],-car_bbox[1]],[car_bbox[0],car_bbox[1]]])
                car_bbox = np.dot(np.array([[np.cos(car_rot),-np.sin(car_rot)],[np.sin(car_rot),np.cos(car_rot)]]),car_bbox.T).T
                car_bbox = car_bbox + np.array(car_centre_loc[:2])
                plt.plot(car_bbox[:,0],car_bbox[:,1],c=color)
                if with_id:
                    plt.text(car_centre_loc[0],car_centre_loc[1],car_id,fontsize=10)
                

            plt.savefig(f'{save_folder}/{idx}.png')
            plt.close()

    def draw_map_w_waypoints(self, waypoints, file_path=None, spawn_point_num=None):
        plt.figure(figsize=(24, 16), dpi=200)
        road_edges = []
        lanes = []
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)

        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:,0],edge[:,1],c='red')

        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:,0],lane[:,1],c='green') 
        # 起点标绿
        plt.scatter(waypoints[0].transform.get_location()[0], waypoints[0].transform.get_location()[1], c='green', s=100)
        # 画上起点的方向
        # rotation = waypoints[0].transform.get_rotation()
        # rotation = rotation[2]
        # x = waypoints[0].transform.get_location()[0]
        # y = waypoints[0].transform.get_location()[1]
        # dx = 0.5 * math.cos(rotation)
        # dy = 0.5 * math.sin(rotation)
        # plt.arrow(x, y, dx, dy, head_width=2, head_length=2, fc='k', ec='k')
        # # 终点标红
        # plt.scatter(waypoints[-1].transform.get_location()[0], waypoints[-1].transform.get_location()[1], c='red', s=100)
        # plt.text(waypoints[0].transform.get_location()[0], waypoints[0].transform.get_location()[1], f"Spawn Point Num: {spawn_point_num}", fontsize=20)
        for waypoint in waypoints:
            # waypoint = np.array(waypoint.transform.get_location())
            # plt.scatter(waypoint[0], waypoint[1], c='blue', s=10)
            # 画上方向
            rotation = waypoint.transform.get_rotation()
            rotation = rotation[2]
            x = waypoint.transform.get_location()[0]
            y = waypoint.transform.get_location()[1]
            dx = 0.5 * math.cos(rotation)
            dy = 0.5 * math.sin(rotation)
            plt.arrow(x, y, dx, dy, head_width=2, head_length=2, fc='k', ec='k')

        # if not os.path.exists('data/draw_pic/spawn_points_w_waypoints'):
        #     os.makedirs('data/draw_pic/spawn_points_w_waypoints')
        # if spawn_point_num is not None:
        #     plt.savefig(f'data/draw_pic/spawn_points_w_waypoints/map_w_waypoints_{spawn_point_num}.png')
        # else: 
        #     plt.savefig('data/draw_pic/spawn_points_w_waypoints/map_w_waypoints.png')
        if file_path is None:
            time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(f'data/draw_pic/map_w_waypoints_{time}.png')
        else:
            plt.savefig(file_path)
        plt.close()

    def draw_map_w_key_waypoints(self, waypoints, spawn_point_num=None):
        plt.figure(figsize=(24, 16), dpi=200)
        road_edges = []
        lanes = []
        for feature_id, feature in self.features.items():
            if feature.feature_type == 'lane':
                lanes.append(feature.polyline)
            else:
                road_edges.append(feature.polyline)

        for edge in road_edges:
            edge = np.array(edge)
            plt.plot(edge[:,0],edge[:,1],c='red')

        for lane in lanes:
            lane = np.array(lane)
            plt.plot(lane[:,0],lane[:,1],c='green') 
        # 起点标绿
        plt.scatter(waypoints[0].transform.get_location()[0], waypoints[0].transform.get_location()[1], c='green', s=100)
        # 终点标红
        plt.scatter(waypoints[-1].transform.get_location()[0], waypoints[-1].transform.get_location()[1], c='red', s=100)
        plt.text(waypoints[0].transform.get_location()[0], waypoints[0].transform.get_location()[1], f"Spawn Point Num: {spawn_point_num}", fontsize=20)
        for waypoint in waypoints:
            waypoint_loc = np.array(waypoint.transform.get_location())
            if waypoint.road_option != RoadOption.STRAIGHT:
                plt.scatter(waypoint_loc[0], waypoint_loc[1], c='red', s=100)
            else:
                plt.scatter(waypoint_loc[0], waypoint_loc[1], c='blue', s=10)
        if not os.path.exists('data/draw_pic/spawn_points_w_waypoints'):
            os.makedirs('data/draw_pic/spawn_points_w_waypoints')
        if spawn_point_num is not None:
            plt.savefig(f'data/draw_pic/spawn_points_w_waypoints/map_w_waypoints_{spawn_point_num}.png')
        else: 
            plt.savefig('data/draw_pic/spawn_points_w_waypoints/map_w_waypoints.png')
        plt.close()


if __name__ == '__main__':
    map_json_path = '/home/ubuntu/sujiaqi/ChatSim/data/end2end_map_data/test_map.json'
    # 加载JSON数据
    with open(map_json_path, 'r') as file:
        map_data = json.load(file)

    # 创建地图对象
    map_obj = Map(map_data)

    # # 获取特定车道的变道可能性
    lane_id = '97'
    # print(map_obj.features[lane_id].neighbor_relations)

    # map_obj.visualize_graph()

    # 获取从97的第一个点到96的最后一个点的路径规划
    # start_point = map_obj.features['97'].polyline[0]
    # end_point = map_obj.features['96'].polyline[-2]
    # path = map_obj.plan_path(start_point, end_point)
    # print(path)
    map_obj.draw_map_w_spawn_points()
    spawn_points = map_obj.get_spawn_points()
    for idx, spawn_point in enumerate(spawn_points):
        print(f"Draw Waypoints of Spawn Point {idx}")
        waypoints = map_obj.generate_overall_plan_waypoints(spawn_point, driving_mode='Random', ignore_lanechange=False)
        # map_obj.draw_map_w_waypoints(waypoints, idx)
        map_obj.draw_map_w_key_waypoints(waypoints, idx)
        
    # map_obj.draw_map()
