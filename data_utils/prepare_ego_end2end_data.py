from typing import List
from matplotlib import pyplot as plt
import numpy as np
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
import tensorflow as tf
import json
import os
import math
def rotate(point, angle):
    """Rotates a point around the origin by the specified angle in radians."""
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, point)

def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car['cx']
    y = car['cy']
    z = car['cz']
    length = car['length']
    width = car['width']
    height = car['height']
    heading = car['heading']
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])

    # The relative positions of the vertices from the box center before rotation.
    relative_positions = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ]) * half_dims

    # Rotate each relative position and add the box center position.
    vertices = np.asarray([rotate(pos, heading) + box_center for pos in relative_positions])
    return vertices

# FILENAME = '/home/ubuntu/junhaoge/ChatSim/data/waymo_tfrecords/1.4.2/segment-13196796799137805454_3036_940_3056_940_with_camera_labels.tfrecord'
# segment-11379226583756500423_6230_810_6250_810_with_camera_labels
# FILENAME = '/home/ubuntu/junhaoge/ChatSim/data/waymo_tfrecords/1.4.2/segment-17761959194352517553_5448_420_5468_420_with_camera_labels.tfrecord'
# output_folder_path = '/home/ubuntu/junhaoge/ChatSim/data/waymo_multi_view/segment-17761959194352517553_5448_420_5468_420_with_camera_labels'
FILENAME = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_tfrecords/1.4.2/segment-10588771936253546636_2300_000_2320_000_with_camera_labels.tfrecord'
output_folder_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/segment-10588771936253546636_2300_000_2320_000_with_camera_labels'
pose_output_path = os.path.join(output_folder_path, 'ego_pose.json')
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

frames = []
for data in dataset:
    frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
    frames.append(frame)
# mean frame = [4:7]
frame_4 = np.reshape(np.array(frames[4].pose.transform), [4, 4])
frame_5 = np.reshape(np.array(frames[5].pose.transform), [4, 4])
frame_6 = np.reshape(np.array(frames[6].pose.transform), [4, 4])
frame_to_mean = [frame_4, frame_5, frame_6]
ref_transform = np.mean(frame_to_mean, axis=0)
ref_transform = np.linalg.inv(ref_transform)

def get_translation_from_matrix(matrix):
    return [matrix[0, 3], matrix[1, 3], matrix[2, 3]]

def get_rotation_from_matrix(matrix):
    # return roll pitch yaw
    return [np.arctan2(matrix[2, 1], matrix[2, 2]),
                     np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)),
                     np.arctan2(matrix[1, 0], matrix[0, 0])]

def get_matrix_from_rotation_and_translation(rotation, translation):
    matrix = np.eye(4)
    # rotation is roll pitch yaw
    matrix[:3, :3] = np.reshape(np.array(tf.transformations.euler_matrix(rotation[0], rotation[1], rotation[2], 'sxyz'))[:3, :3], [3, 3])
    matrix[:3, 3] = translation
    return matrix

def rpy2R(rpy): # [r,p,y] 单位rad
    rot_x = np.array([[1, 0, 0],
                    [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                    [0, math.sin(rpy[0]), math.cos(rpy[0])]])
    rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                    [0, 1, 0],
                    [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                    [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                    [0, 0, 1]])
    R = np.dot(rot_z, np.dot(rot_y, rot_x))
    return R

ego_pose_data = []
for frame in frames:
    cur_ego_pose_data = {}
    cur_ego_pose = frame.pose # 4*4 matrix
    # convert waymo transform type to matric
    cur_ego_pose_global = np.reshape(np.array(frame.pose.transform), [4, 4])
    # print(cur_ego_pose_global)
    cur_ego_pose = np.dot(ref_transform, cur_ego_pose_global)
    # print(cur_ego_pose)
    cur_ego_pose_data['location'] = get_translation_from_matrix(cur_ego_pose)
    cur_ego_pose_data['rotation'] = get_rotation_from_matrix(cur_ego_pose)
    trans_pose = rpy2R(cur_ego_pose_data['rotation'])
    # print(trans_pose)
    ego_pose_data.append(cur_ego_pose_data)

with open(pose_output_path, 'w') as f:
    json.dump(ego_pose_data, f, indent=2)

