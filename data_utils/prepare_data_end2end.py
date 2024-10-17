import sys
import os
sys.path.append(os.getcwd())
# set cuda visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import cv2
import math
import imageio
from matplotlib import pyplot as plt
import tensorflow as tf
import json
import os
import argparse
import json
from tqdm import tqdm
# from simple_waymo_open_dataset_reader import WaymoDataFileReader
# from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
# from simple_waymo_open_dataset_reader import utils
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils
from box_utils import bbox_to_corner3d, get_bound_2d_mask
from img_utils import draw_3d_box_on_img
from graphics_utils import project_numpy

# castrack_path = '/nas/home/yanyunzhi/waymo/castrack/seq_infos/val/result.json'
# with open(castrack_path, 'r') as f:
#     castrack_infos = json.load(f)

camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT', 
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT', 
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}

image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}

opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

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

def get_extrinsic(camera_calibration):
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4) # camera to vehicle
    extrinsic = np.matmul(camera_extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
    return extrinsic
    
def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    return intrinsic

def project_label_to_image(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3], 
        K=intrinsic, 
        RT=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    return points_uv, valid

def project_label_to_mask(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    
    return mask
    
    
def parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, start_idx=None, end_idx=None, debug=False):
    print(f'Processing sequence {seq_name}...')
    print(f'Saving to {seq_save_dir}')
    try:
        with open(track_file, 'r') as f:
            castrack_infos = json.load(f)
    except:
        castrack_infos = dict()

    os.makedirs(seq_save_dir, exist_ok=True)
    
    seq_path = os.path.join(root_dir, seq_name+'.tfrecord')
    error_flag = False
    # set start and end timestep
    dataset = tf.data.TFRecordDataset(seq_path, compression_type='')
    frames = []
    for data in dataset:
        frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
        frames.append(frame)
    num_frames = len(frames)
    # print(f'Number of frames: {num_frames}')
    # print(datafile.get_record_table())
    start_idx = start_idx or 0
    end_idx = end_idx or num_frames - 1
    # calculate the reference transform
    transform_all = []
    # print(frame[0])
    for frame_id, frame in enumerate(frames):
        # print(frame)
        if frame_id < start_idx:
            continue
        if frame_id > end_idx:
            break
        transform = np.array(frame.pose.transform).reshape(4, 4)
        transform_all.append(transform)
    transform_all = np.stack(transform_all)
    ref_transform = np.mean(transform_all[:,:3, 3], axis=0)
    # ref_transform = np.linalg.inv(ref_transform)

    if 'map' in process_list:
        road_edges = []
        lanes = []
        driveway = []
        road_line = []
        crosswalk = []
        map_feature_dict = {}
        # first frame
        for i in range(len(frames[0].map_features)):
            feature_id = frames[0].map_features[i].id
            map_feature_dict[feature_id] = {}
            if len(frames[0].map_features[i].lane.polyline) > 0:
                if frames[0].map_features[i].lane.type:
                    map_feature_dict[feature_id]['lane_type'] = frames[0].map_features[i].lane.type
                else:
                    map_feature_dict[feature_id]['lane_type'] = None
                map_feature_dict[feature_id]['feature_type'] = 'lane'
                curr_lane = []
                for node in frames[0].map_features[i].lane.polyline:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_lane.append(node_position)
                curr_lane = np.stack(curr_lane)
                # curr_lane = np.transpose(np.matmul(ref_transform, np.transpose(curr_lane)))[:, 0:3]
                curr_lane = curr_lane[:,:3] - ref_transform
                map_feature_dict[feature_id]['polyline'] = curr_lane.tolist()
                lanes.append(curr_lane)
                if frames[0].map_features[i].lane.interpolating:
                    map_feature_dict[feature_id]['interpolating'] = frames[0].map_features[i].lane.interpolating
                else:
                    map_feature_dict[feature_id]['interpolating'] = None
                if frames[0].map_features[i].lane.speed_limit_mph:
                    map_feature_dict[feature_id]['speed_limit_mph'] = frames[0].map_features[i].lane.speed_limit_mph
                    # print(type(framess[0].map_features[i].lane.type))
                if frames[0].map_features[i].lane.entry_lanes:
                    entry_lanes = []
                    for entry_lane in frames[0].map_features[i].lane.entry_lanes:
                        entry_lanes.append(str(entry_lane))
                    map_feature_dict[feature_id]['entry_lanes'] = entry_lanes
                if frames[0].map_features[i].lane.exit_lanes:
                    exit_lanes = []
                    for exit_lane in frames[0].map_features[i].lane.exit_lanes:
                        exit_lanes.append(str(exit_lane))
                    map_feature_dict[feature_id]['exit_lanes'] = exit_lanes
                if frames[0].map_features[i].lane.left_neighbors:
                    map_feature_dict[feature_id]['left_neighbors'] = []
                    for left_neighbor in frames[0].map_features[i].lane.left_neighbors:
                        left_neighbor_dict = {'feature_id': str(left_neighbor.feature_id), 'self_start_index': left_neighbor.self_start_index, \
                                        'self_end_index': left_neighbor.self_end_index, 'neighbor_start_index': left_neighbor.neighbor_start_index, \
                                            'neighbor_end_index': left_neighbor.neighbor_end_index}
                        map_feature_dict[feature_id]['left_neighbors'].append(left_neighbor_dict)
                if frames[0].map_features[i].lane.right_neighbors:
                    map_feature_dict[feature_id]['right_neighbors'] = []
                    for right_neighbor in frames[0].map_features[i].lane.right_neighbors:
                        right_neighbor_dict = {'feature_id': str(right_neighbor.feature_id), 'self_start_index': right_neighbor.self_start_index, \
                                        'self_end_index': right_neighbor.self_end_index, 'neighbor_start_index': right_neighbor.neighbor_start_index, \
                                            'neighbor_end_index': right_neighbor.neighbor_end_index}
                        map_feature_dict[feature_id]['right_neighbors'].append(right_neighbor_dict)
            
            if len(frames[0].map_features[i].road_edge.polyline) > 0:
                if frames[0].map_features[i].road_edge.type:
                    map_feature_dict[feature_id]['road_edge_type'] = frames[0].map_features[i].road_edge.type
                else:
                    map_feature_dict[feature_id]['road_edge_type'] = None
                map_feature_dict[feature_id]['feature_type'] = 'road_edge'
                curr_edge = []
                for node in frames[0].map_features[i].road_edge.polyline:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_edge.append(node_position)
                curr_edge = np.stack(curr_edge)
                # curr_edge = np.transpose(np.matmul(ref_transform, np.transpose(curr_edge)))[:, 0:3]
                curr_edge = curr_edge[:,:3] - ref_transform
                map_feature_dict[feature_id]['polyline'] = curr_edge.tolist()
                road_edges.append(curr_edge)

            if len(frames[0].map_features[i].crosswalk.polygon) > 0:
                map_feature_dict[feature_id]['feature_type'] = 'crosswalk'
                curr_polygon = []
                for node in frames[0].map_features[i].crosswalk.polygon:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_polygon.append(node_position)
                curr_polygon = np.stack(curr_polygon)
                # curr_polygon = np.transpose(np.matmul(ref_transform, np.transpose(curr_polygon)))[:, 0:3]
                curr_polygon = curr_polygon[:,:3] - ref_transform
                crosswalk.append(curr_polygon)
                map_feature_dict[feature_id]['polyline'] = curr_polygon.tolist()

            if len(frames[0].map_features[i].road_line.polyline) > 0:
                map_feature_dict[feature_id]['feature_type'] = 'road_line'
                curr_polygon = []
                for node in frames[0].map_features[i].road_line.polyline:
                    node_position = np.ones(4)
                    node_position[0] = node.x
                    node_position[1] = node.y
                    node_position[2] = node.z
                    curr_polygon.append(node_position)
                curr_polygon = np.stack(curr_polygon)
                # curr_polygon = np.transpose(np.matmul(ref_transform, np.transpose(curr_polygon)))[:, 0:3]
                curr_polygon = curr_polygon[:,:3] - ref_transform
                road_line.append(curr_polygon)
                map_feature_dict[feature_id]['polyline'] = curr_polygon.tolist()
        if len(map_feature_dict.keys()) < 1:
            print('No map feature in the first frame')
            error_flag = True
            return False
        map_json_save_path = os.path.join(seq_save_dir, 'map_feature.json')
        with open(map_json_save_path, 'w') as f:
            json.dump(map_feature_dict, f, indent=2)

        if debug:
            x_min = -300
            x_max = 500
            y_min = -200
            y_max = 200
            cropped_road_edges = []
            for edge in road_edges:
                new_road_edge = []
                for i in range(edge.shape[0]):
                    if edge[i,0] < x_min or edge[i,0] > x_max or edge[i,1] < y_min or edge[i,1] > y_max:
                        continue
                    new_road_edge.append(edge[i])
                if len(new_road_edge) > 0:
                    new_road_edge = np.stack(new_road_edge)
                    cropped_road_edges.append(new_road_edge)

            cropped_lanes = []
            for lane in lanes:
                new_lane = []
                for i in range(lane.shape[0]):
                    if lane[i,0] < x_min or lane[i,0] > x_max or lane[i,1] < y_min or lane[i,1] > y_max:
                        continue
                    new_lane.append(lane[i])
                if len(new_lane) > 0:
                    new_lane = np.stack(new_lane)
                    cropped_lanes.append(new_lane)

            max_per_lane_node = 20

            for edge in cropped_road_edges:
                edge = np.array(edge)
                # edge = edge[::5]
                plt.plot(edge[:,0],edge[:,1],c='red')

            for lane in cropped_lanes:
                lane = np.array(lane)
                # lane = lane[::5]
                plt.plot(lane[:,0],lane[:,1],c='green')

            for driveway_edge in driveway:
                driveway_edge = np.array(driveway_edge)
                plt.plot(driveway_edge[:,0], driveway_edge[:,1], c='blue')

            for road_line in road_line:
                edge = np.array(road_line)
                plt.plot(edge[:,0],edge[:,1],c='yellow', linewidth=1)

            for crosswalk_edge in crosswalk:
                crosswalk_edge = np.array(crosswalk_edge)
                plt.plot(crosswalk_edge[:,0], crosswalk_edge[:,1], c='black')

            # 在整个fig的绝对位置的左下角处增加各个道路类型的图例，要求是一行文字，旁边是一个对应的色块
            y_offset = -0.15
            plt.text(0.1, 0.38+y_offset, 'Road Edge', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='red')
            plt.text(0.1, 0.34+y_offset, 'Lane', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='green')
            plt.text(0.1, 0.30+y_offset, 'Driveway', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='blue')
            plt.text(0.1, 0.26+y_offset, 'Road Line', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='yellow')
            plt.text(0.1, 0.22+y_offset, 'Crosswalk', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='black')

            plt_save_path = os.path.join(seq_save_dir, 'map_feature.png')
            plt.savefig(plt_save_path)
            plt.close()

    if 'egp_pose' in process_list:
        ego_pose_data = []
        print("Processing ego pose...")
        for frame_id, frame in enumerate(frames):
            cur_ego_pose_data = {}
            cur_ego_pose = frame.pose # 4*4 matrix
            # convert waymo transform type to matric
            cur_ego_pose_global = np.reshape(np.array(frame.pose.transform), [4, 4])
            # print(cur_ego_pose_global)
            # cur_ego_pose = np.dot(ref_transform, cur_ego_pose_global)
            cur_ego_pose = cur_ego_pose_global
            cur_ego_pose[:3,3] = cur_ego_pose_global[:3,3] - ref_transform.reshape(1, 3)
            # print(cur_ego_pose)
            cur_ego_pose_data['location'] = get_translation_from_matrix(cur_ego_pose)
            cur_ego_pose_data['rotation'] = get_rotation_from_matrix(cur_ego_pose)
            trans_pose = rpy2R(cur_ego_pose_data['rotation'])
            # print(trans_pose)
            ego_pose_data.append(cur_ego_pose_data)
        ego_pose_save_dir = os.path.join(seq_save_dir, 'ego_pose.json')
        # os.makedirs(ego_pose_save_dir, exist_ok=True)
        with open(ego_pose_save_dir, 'w') as f:
            json.dump(ego_pose_data, f, indent=2)

    if 'static_vehicle' in process_list:
        all_static_actor_data = {}
        actor_data_output_path = os.path.join(seq_save_dir, 'actor_data')
        if not os.path.exists(actor_data_output_path):
            os.makedirs(actor_data_output_path)
        frame_obj_dict = {}
        for frame_id,frame in enumerate(frames):
            actor_data = []
            for label in frame.laser_labels:
                label_id = label.id
                if label_id in frame_obj_dict.keys():
                    continue
                else:
                    frame_obj_dict[label_id] = []
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
                frame_obj_dict[label_id].append([tx, ty, tz])
                # print(obj_pose_vehicle)
                # calculate the frame to ref frame transformation
                cur_frame_transform = np.reshape(np.array(frame.pose.transform), [4, 4])
                # print('cur_frame_transform:', cur_frame_transform)
                obj_pose_global = np.dot(cur_frame_transform, obj_pose_vehicle)
                # print('obj_pose_global:', obj_pose_global)
                # obj_pose_ref = np.dot(ref_transform, obj_pose_global)
                obj_pose_ref = obj_pose_global
                obj_pose_ref[:3,3] = obj_pose_global[:3,3] - ref_transform
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
                # if point_num_in_lidar < 5:
                #     continue
                actor_data.append(tmp_dict)
                # if label_id in all_static_actor_data.keys():
                #     if np.linalg.norm(np.array(all_static_actor_data[label_id]['location']) - np.array(get_translation_from_matrix(obj_pose_ref))) > 2:
                #         all_static_actor_data.pop(label_id)
                # # if label_id not in all_static_actor_data.keys() and if_vehicle and not if_dynamic:
                # if label_id not in all_static_actor_data.keys() and not if_dynamic:
                #     close_flag = False
                #     # 确认没有位置相近的actor
                #     for key in all_static_actor_data.keys():
                #         if np.linalg.norm(np.array(all_static_actor_data[key]['location']) - np.array(get_translation_from_matrix(obj_pose_ref))) < 0.1:
                #             print("duplicate actor")
                #             close_flag = True
                #             break
                #         else:
                #             continue
                #     if close_flag:
                #         continue
                #     all_static_actor_data[label_id] = tmp_dict
                if not if_dynamic:
                    all_static_actor_data[label_id] = tmp_dict
            # frame id name 为类似于000001的格式
            frame_id_name = str(frame_id).zfill(3)
            with open(os.path.join(actor_data_output_path, frame_id_name + '.json'), 'w') as f:
                json.dump(actor_data, f, indent=2)
        # distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
        # dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
        for obj_id in frame_obj_dict.keys():
            if len(frame_obj_dict[obj_id]) < 1:
                continue
            distance = np.linalg.norm(np.array(frame_obj_dict[obj_id][0]) - np.array(frame_obj_dict[obj_id][-1]))
            dynamic = np.any(np.std(np.array(frame_obj_dict[obj_id]), axis=0) > 1) or distance > 2
            if dynamic:
                if obj_id in all_static_actor_data.keys():
                    all_static_actor_data.pop(obj_id)
        with open(os.path.join(seq_save_dir, 'all_static_actor_data.json'), 'w') as f:
            json.dump(all_static_actor_data, f, indent=2)

    return True
        

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_list', type=str, nargs='+', default=['map', 'egp_pose', 'static_vehicle'])
    parser.add_argument('--root_dir', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/data/waymo_tfrecords/1.4.2')
    parser.add_argument('--save_dir', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view')
    # parser.add_argument('--split_file', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/data/data_split/train_simulation_single.txt')
    parser.add_argument('--split_file', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/data/data_split/train_simulation_new.txt')
    parser.add_argument('--segment_file', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/data/data_split/segment_list_train.txt')
    args = parser.parse_args()
    
    process_list = args.process_list
    root_dir = args.root_dir
    save_dir = args.save_dir
    split_file = open(args.split_file, "r").readlines()[1:]
    scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    seq_names = [line.strip().split(",")[1] for line in split_file]
    segment_file = args.segment_file
    seq_lists = open(segment_file).read().splitlines()
    os.makedirs(save_dir, exist_ok=True)
    error_seq_list = []
    for i, scene_id in enumerate(scene_ids_list):
        print(f'Processing scene {scene_id}...')
        if scene_id not in [16, 19, 36]:
            continue
        assert seq_names[i][3:] == seq_lists[scene_id][8:14]
        seq_save_dir = os.path.join(save_dir, str(scene_id).zfill(3))
        # try:
        valid = parse_seq_rawdata(
            process_list=process_list,
            root_dir=root_dir,
            seq_name=seq_lists[scene_id],
            seq_save_dir=seq_save_dir,
            track_file=None,
            debug=True
        )
        if not valid:
            error_seq_list.append(seq_lists[scene_id])
        # except Exception as e:
        #     print(f'Error in sequence {seq_lists[scene_id]}: {e}')
        #     error_seq_list.append(seq_lists[scene_id])
        #     continue
    # save the error sequence list and delete the original file
    if len(error_seq_list) > 0:
        with open(os.path.join(save_dir, 'error_sequence_list.txt'), 'w') as f:
            import time
            time_now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            f.write(f'Error sequence list at {time_now}\n')
            for seq in error_seq_list:
                f.write(seq + '\n')
if __name__ == '__main__':
    main()