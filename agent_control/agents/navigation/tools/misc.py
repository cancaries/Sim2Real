#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
import numpy as np
from agents.navigation.waypoint import Transform,Waypoint
from shapely.geometry import Polygon, LineString, Point
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_rotation(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    yaw = math.atan2(dy, dx)
    return [0, 0, yaw]

def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0

def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location[0] - current_location[0], target_location[1] - current_location[1]])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(orientation), math.sin(orientation)])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(orientation), math.sin(orientation)])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)

def distance_vehicle_by_loc(ego_loc, target_loc):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    x = ego_loc.x - target_loc.x
    y = ego_loc.y - target_loc.y

    return math.sqrt(x * x + y * y)

def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def positive(num):
    """
    Return the given number if positive, else 0

        :param num: value to check
    """
    return num if num > 0.0 else 0.0

def get_bbox_corners(location, heading, size):
    """
    根据车辆的位置、航向角、尺寸计算四个顶点的坐标
            
            :param location: 车辆的位置[x,y]
            :param heading: 航向角yaw
            :param size: 车辆的尺寸[w,l]
            
            :return: 四个顶点的坐标
    """
    x, y = location[:2]
    width, length = size[0], size[1] # 但这边size的顺序是[l,w]

    # 计算四个顶点的相对位置
    corners = np.array([
        [length/2, width/2],
        [-length/2, width/2],
        [-length/2, -width/2],
        [length/2, -width/2]
    ])

    # 旋转矩阵根据航向角进行旋转
    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])
    
    # 应用旋转矩阵和位置偏移
    rotated_corners = np.dot(corners, rotation_matrix)
    translated_corners = rotated_corners + np.array([x, y])
    
    return translated_corners
def calculate_relative_vector(v1, v2):
    """计算相对位置向量"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return v2 - v1

def calculate_movement_vector(current_position,previous_position):
    """计算车辆从上一帧到当前帧的运动向量"""
    current_position = np.array(current_position)
    previous_position = np.array(previous_position)
    movement_vector = current_position - previous_position
    return movement_vector

def calculate_angle_between_vectors(v1, v2):
    """计算两个向量之间的夹角"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def calculate_angel_from_vector1_to_vector2(v1, v2):
    """
    计算从v1旋转到v2的夹角，返回夹角范围为[0, 2pi]
    :param v1: 向量1
    :param v2: 向量2
    :return: 从v1到v2的夹角，单位为弧度，范围[0, 2pi]
    """
    # 归一化向量
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # 计算向量的点积并得到夹角的cos值
    dot_product = np.dot(v1_norm, v2_norm)
    
    # 防止由于数值误差导致cos值超出[-1, 1]范围
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算夹角，默认返回的夹角范围是[0, pi]
    angle = np.arccos(dot_product)
    
    # 使用叉积判断方向
    cross_product = np.cross(v1_norm, v2_norm)
    # print(cross_product)
    
    # 如果叉积的z分量为正，表示逆时针旋转；如果为负，表示顺时针旋转
    if cross_product > 0:
        angle = 2 * np.pi - angle
    
    return angle
def is_collision(target_vehicle_loc,
                     target_vehicle_yaw,
                     target_vehicle_bbox,
                     reference_vehicle_loc,
                     reference_vehicle_yaw,
                     reference_vehicle_bbox):
        # 计算两个车辆的四个顶点
        corners1 = get_bbox_corners(target_vehicle_loc, target_vehicle_yaw, target_vehicle_bbox)
        corners2 = get_bbox_corners(reference_vehicle_loc, reference_vehicle_yaw, reference_vehicle_bbox)
        
        # 使用shapely库判断两个多边形是否有重叠区域
        polygon1 = Polygon(corners1)
        polygon2 = Polygon(corners2)
        
        # if polygon1.intersects(polygon2) and target_vehicle_bbox[1]>3:
        #     # import ipdb; ipdb.set_trace()
        #     import matplotlib.pyplot as plt
        #     plt.plot(*polygon1.exterior.xy)
        #     plt.plot(*polygon2.exterior.xy)
        #     plt.savefig('/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/collision.png')
        #     plt.close()
        return polygon1.intersects(polygon2)
    

def detect_route_interaction(test_path, reference_path, interaction_range=10.0):
    """
    使用 Shapely 库检测路径交互。
    test_path: 测试路径 (列表形式，每个元素是 (x, y) 坐标)
    reference_path: 参考路径 (列表形式，每个元素是 (x, y) 坐标)
    interaction_range: 交互范围，默认5米
    返回 True 表示有交互，False 表示没有交互
    """
    # 将参考路径转换为 LineString
    reference_line = LineString(reference_path)
    
    # 为参考路径创建一个缓冲区范围（例如 5 米）
    reference_buffer = reference_line.buffer(interaction_range)
    
    # 检查测试路径的每个点是否与参考路径的缓冲区有交集
    for point in test_path:
        point_geometry = Point(point)
        if reference_buffer.intersects(point_geometry):
            return True  # 如果有点与缓冲区相交，则存在交互
    
    return False

def build_transform_path_from_ego_pose_data(ego_pose_data):
    transform_path = []
    for ego_pose in ego_pose_data:
        transform = Transform(ego_pose)
        transform_path.append(transform)
    return transform_path