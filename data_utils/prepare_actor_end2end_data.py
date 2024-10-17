from typing import List
from matplotlib import pyplot as plt
import numpy as np
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
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
frame_4 = np.reshape(np.array(frames[4].pose.transform), [4, 4])
frame_5 = np.reshape(np.array(frames[5].pose.transform), [4, 4])
frame_6 = np.reshape(np.array(frames[6].pose.transform), [4, 4])
frame_to_mean = [frame_4, frame_5, frame_6]
ref_transform_mean = np.mean(frame_to_mean, axis=0)
ref_transform = np.linalg.inv(ref_transform_mean)

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

all_static_actor_data = {}
actor_data_output_path = os.path.join(output_folder_path, 'actor_data')
if not os.path.exists(actor_data_output_path):
    os.makedirs(actor_data_output_path)
for frame_id,frame in enumerate(frames):
    actor_data = []
    for label in frame.laser_labels:
        label_id = label.id
        box = label.box
        meta = label.metadata
        # assume every bbox is visible in at least on camera
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            obj_class = "vehicle"
        elif label.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
            obj_class = "pedestrian"
        elif label.type == label_pb2.Label.Type.TYPE_SIGN:
            obj_class = "sign"
        elif label.type == label_pb2.Label.Type.TYPE_CYCLIST:
            obj_class = "cyclist"
        else:
            obj_class = "misc"
        speed = np.linalg.norm([meta.speed_x, meta.speed_y]) 
        point_num_in_lidar = label.num_lidar_points_in_box
        most_visible_camera_name = label.most_visible_camera_name
        if most_visible_camera_name not in ['FRONT','FRONT_LEFT','FRONT_RIGHT','SIDE_LEFT','SIDE_RIGHT']:
            continue
        print(most_visible_camera_name)
        # thresholding, use 1.0 m/s to determine whether the pixel is moving
        # follow EmerNeRF
        if_dynamic = bool(speed > 1.0)
        if_vehicle = bool(label.type == label_pb2.Label.Type.TYPE_VEHICLE)
        # build 3D bounding box dimension
        length, width, height = box.length, box.width, box.height
        size = [width, length, height]
        # build 3D bounding box pose
        tx, ty, tz = box.center_x, box.center_y, box.center_z
        heading = box.heading
        c = math.cos(heading)
        s = math.sin(heading)
        rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obj_pose_vehicle = np.eye(4)
        obj_pose_vehicle[:3, :3] = rotz_matrix
        obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
        # print(obj_pose_vehicle)
        # calculate the frame to ref frame transformation
        cur_frame_transform = np.reshape(np.array(frame.pose.transform), [4, 4])
        # print('cur_frame_transform:', cur_frame_transform)
        obj_pose_global = np.dot(cur_frame_transform, obj_pose_vehicle)
        # print('obj_pose_global:', obj_pose_global)
        obj_pose_ref = np.dot(ref_transform, obj_pose_global)
        # print('obj_pose_ref:', obj_pose_ref)
        tmp_dict = dict()
        tmp_dict['label_id'] = label_id
        # print(type(label_id))
        tmp_dict['size'] = size
        # print(type(size))
        tmp_dict['obj_class'] = obj_class
        tmp_dict['point_num_in_lidar'] = point_num_in_lidar
        tmp_dict['speed'] = speed
        tmp_dict['if_vehicle'] = if_vehicle
        tmp_dict['motion_state'] = if_dynamic
        # print(type(if_dynamic))
        # print(type(if_vehicle)) 
        tmp_dict['pose'] = obj_pose_ref.tolist()
        # print(type(tmp_dict['pose']))
        tmp_dict['location'] = get_translation_from_matrix(obj_pose_ref)
        tmp_dict['rotation'] = get_rotation_from_matrix(obj_pose_ref)
        if point_num_in_lidar < 20:
            continue
        actor_data.append(tmp_dict)
        if label_id in all_static_actor_data.keys():
            if np.linalg.norm(np.array(all_static_actor_data[label_id]['location']) - np.array(get_translation_from_matrix(obj_pose_ref))) > 2:
                all_static_actor_data.pop(label_id)
        if label_id not in all_static_actor_data.keys() and if_vehicle and not if_dynamic:
            close_flag = False
            # 确认没有位置相近的actor
            for key in all_static_actor_data.keys():
                if np.linalg.norm(np.array(all_static_actor_data[key]['location']) - np.array(get_translation_from_matrix(obj_pose_ref))) < 0.1:
                    print("duplicate actor")
                    close_flag = True
                    break
                else:
                    continue
            if close_flag:
                continue
            all_static_actor_data[label_id] = tmp_dict
    # frame id name 为类似于000001的格式
    frame_id_name = str(frame_id).zfill(6)
    with open(os.path.join(actor_data_output_path, frame_id_name + '.json'), 'w') as f:
        json.dump(actor_data, f, indent=2)
with open(os.path.join(output_folder_path, 'all_static_actor_data.json'), 'w') as f:
    json.dump(all_static_actor_data, f, indent=2)

        
            
