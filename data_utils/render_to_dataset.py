import os
import json
import argparse
import dis
import math
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
import cv2
from PIL import Image
import re
import imageio
import copy
import time
opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

def process_image(file_name):
    if file_name.endswith(".png"):
        image = Image.open(file_name)
    # resize image to 960p
    # image = image.resize((1280, 960), Image.ANTIALIAS)
    return np.array(image.convert("RGB"))

def create_video_from_images(image_folder, video_name, fps=30):
    images = []
    # 寻找所有 png 文件
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".png") and 'concat' in img]
    if len(image_files) < fps * 2:
        return False
    image_files = sorted(image_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    # print(image_files)
    # 利用线程池并行处理图像
    images = [process_image(os.path.join(image_folder,image)) for image in image_files]

    with imageio.get_writer(video_name, fps=fps) as video:
        for image in images:
            video.append_data(image)
    
    return True

def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def project_numpy(xyz, K, RT, H, W):
    '''
    input: 
    xyz: [N, 3], pointcloud
    K: [3, 3], intrinsic
    RT: [4, 4], w2c
    
    output:
    mask: [N], pointcloud in camera frustum
    xy: [N, 2], coord in image plane
    '''
    
    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = []
    valid_y = []
    for idx, x in enumerate(xyz_pixel[:, 0]):
        if x < 0 or x >= W:
            valid_x.append(False)
        else:
            valid_x.append(True)
    for idx, y in enumerate(xyz_pixel[:, 1]):
        if y < 0 or y >= H:
            valid_y.append(False)
        else:
            valid_y.append(True)

    valid_pixel = [valid_x.count(True) >= 4 and valid_y.count(True) >= 4]
    mask = np.logical_and(valid_depth, valid_pixel)
    # print(mask)
    return xyz_pixel, mask

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

def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    
    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

def get_ego_vehicle_matrices(json_file, ego_poses_ori_json=None):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    matrices = []
    ego_loc = []
    ego_yaw = []
    for item in data:
        if 'ego_vehicle' in item:
            ego_vehicle = item['ego_vehicle']
            loc = np.array(ego_vehicle['loc']) 
            ego_loc.append(loc)
            theta = ego_vehicle['rot'][2]  
            ego_yaw.append(theta)

    ego_z = [loc[2] + 2.11 for loc in ego_loc]
    sigma = 2.0
    filtered_z = gaussian_filter1d(ego_z, sigma=sigma)
    ego_loc_new = []

    for idx, loc_ori in enumerate(ego_loc):
        loc = loc_ori.copy()
        loc[2] = filtered_z[idx]                            
        ego_loc_new.append(loc)

    ego_poses_ref = []
    ego_poses_indices = []
    if ego_poses_ori_json is not None:
        with open(ego_poses_ori_json, 'r') as f:
            ego_poses_ori = json.load(f)
        for idx, item in enumerate(ego_poses_ori):
            loc = item['location']
            rot = item['rotation']
            ego_poses_ref.append(loc)
            ego_poses_indices.append(rot)
        kd_tree_rot = KDTree(ego_poses_ref)

    rolls = []
    for idx, loc in enumerate(ego_loc_new):
        roll = ego_poses_indices[kd_tree_rot.query(ego_loc_new[idx])[1]][0]
        rolls.append(roll)

    pitchs = []
    for idx, loc in enumerate(ego_loc_new):
        pitch = ego_poses_indices[kd_tree_rot.query(ego_loc_new[idx])[1]][1]
        pitchs.append(pitch)
    
    sigma = 0.5
    rolls = gaussian_filter1d(rolls, sigma=sigma)
    pitchs = gaussian_filter1d(pitchs, sigma=sigma)
    for idx, loc in enumerate(ego_loc_new):
        theta = ego_yaw[idx]
        roll = rolls[idx]
        pitch = pitchs[idx]
        R_z = rpy2R([roll, pitch, theta])
        T_vehicle_to_world = np.eye(4)
        T_vehicle_to_world[:3, :3] = R_z  
        T_vehicle_to_world[:3, 3] = loc   
        T_vehicle_to_world[3, 3] = 1
        T_vehicle_to_world = np.dot(T_vehicle_to_world,opencv2camera)
        matrices.append(T_vehicle_to_world)

    return matrices

def get_agent_vehicle_matrices(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

        agent_vehicle_info = []
        for item in data:
            tmp_agent_vehcile_info = dict()
            for key, vehicle_info in item.items():
                if key == 'ego_vehicle':
                    continue
                loc = np.array(vehicle_info['loc'])
                if not 'static' in key:
                    loc[2] = loc[2] + vehicle_info['bbox'][2] / 2
                tmp_agent_vehcile_info[key] = vehicle_info
                tmp_agent_vehcile_info[key]['loc'] = loc

            agent_vehicle_info.append(tmp_agent_vehcile_info)

    return agent_vehicle_info

    
def get_intrinsic(camera_intrinsic):
    fx = camera_intrinsic[0] / 1.2
    fy = camera_intrinsic[1] / 1.2
    cx = camera_intrinsic[2] / 1.2
    cy = camera_intrinsic[3] / 1.2
    intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    return intrinsic

def project_label_to_image(dim, obj_pose, calibration,img_width=1920, img_height=1280):
    bbox_w, bbox_l, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = calibration['extrinsic']
    intrinsic = calibration['intrinsic']
    width, height = img_width, img_height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3], 
        K=intrinsic, 
        RT=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    return points_uv, valid

def disassemble_matrix(matrix):
    """
    Disassemble a 4x4 transformation matrix into its rotation and translation components.

    Parameters:
        matrix (np.ndarray): The 4x4 transformation matrix.

    Returns:
        tuple: The rotation angles (in degrees) and translation vector.
    """
    # Extract the rotation matrix
    rotation = matrix[:3, :3]
    
    # Compute the rotation angles from the rotation matrix
    theta_x = np.arctan2(rotation[2, 1], rotation[2, 2])
    theta_y = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))
    theta_z = np.arctan2(rotation[1, 0], rotation[0, 0])
    
    # Convert the rotation angles from radians to degrees
    # theta_x_deg = np.rad2deg(theta_x)
    # theta_y_deg = np.rad2deg(theta_y)
    # theta_z_deg = np.rad2deg(theta_z)
    
    # Extract the translation vector
    translation = matrix[:3, 3]
    
    return theta_x, theta_y, theta_z, translation

def camera_intrinsic_transform(vfov=35,hfov=60,pixel_width=1600,pixel_height=900):
    camera_intrinsics = np.zeros((3,3))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

def pitch_rotation_matrix(pitch_deg, R_current):
    """
    根据当前的旋转矩阵 R_current 和俯瞰角度 pitch_deg，计算出新的旋转矩阵
    :param pitch_deg: 俯瞰的角度（度数）
    :param R_current: 当前的旋转矩阵
    :return: 俯瞰角度后的新的旋转矩阵
    """
    pitch_rad = np.deg2rad(pitch_deg)  # 将角度转为弧度

    # 提取当前旋转矩阵中的右向量（局部坐标系的 x 轴方向）
    right_vector = R_current[:, 0]  # x 轴方向向量

    # 使用罗德里格旋转公式来绕任意轴旋转
    x, y, z = right_vector
    c = np.cos(pitch_rad)
    s = np.sin(pitch_rad)
    R_pitch = np.array([
        [c + (1 - c) * x * x, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
        [(1 - c) * y * x + s * z, c + (1 - c) * y * y, (1 - c) * y * z - s * x],
        [(1 - c) * z * x - s * y, (1 - c) * z * y + s * x, c + (1 - c) * z * z]
    ])

    # 计算最终的旋转矩阵：先应用当前的 R，再应用新的俯瞰矩阵
    R_final = np.dot(R_pitch, R_current)

    return R_final

def rotate_matrix_by_deg(matrix, yaw_deg, pitch_deg=0, roll_deg=0):
    """
    Rotate a 4x4 matrix around the Z-axis by a specified angle.

    Parameters:
        matrix (np.ndarray): The original 4x4 transformation matrix.
        yaw_deg (float): The rotation angle in degrees (default is 25.2°).

    Returns:
        np.ndarray: The new 4x4 matrix after rotation.
    """
    # Convert angle from degrees to radians
    yaw_rad = np.deg2rad(yaw_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    roll_rad = np.deg2rad(roll_deg)
    
    # Define the Z-axis rotation matrix
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0,                  0,                 1]
    ])
    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    
    # Extract the original rotation and translation components from the input matrix
    original_rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    new_rotation = np.dot(Rz, original_rotation)
    
    new_rotation = pitch_rotation_matrix(-pitch_deg, new_rotation)
    # new_rotation = np.dot(rotation_matrix, original_rotation)
    # rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    
    # Reconstruct the new 4x4 transformation matrix
    new_matrix = np.eye(4)  # Create an identity matrix for the base
    new_matrix[:3, :3] = new_rotation
    new_matrix[:3, 3] = translation
    
    return new_matrix

def rotate_matrix(matrix,R):
    """
    Rotate a 4x4 matrix around the Z-axis by a specified angle.

    Parameters:
        matrix (np.ndarray): The original 4x4 transformation matrix.
        yaw_deg (float): The rotation angle in degrees (default is 25.2°).

    Returns:
        np.ndarray: The new 4x4 matrix after rotation.
    """
    # Define the Z-axis rotation matrix
    Rz = R[:3,:3]
    # Extract the original rotation and translation components from the input matrix
    original_rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    new_rotation = np.dot(Rz, original_rotation)
    
    # Reconstruct the new 4x4 transformation matrix
    new_matrix = np.eye(4)  # Create an identity matrix for the base
    new_matrix[:3, :3] = new_rotation
    new_matrix[:3, 3] = translation
    
    return new_matrix

# save simulation dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sg_data_path", default="/home/ubuntu/DATA3/waymo_train_already/", type=str)
    parser.add_argument("--render_path", default="/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp", type=str)
    parser.add_argument("--dataset_output_path", default='/home/ubuntu/DATA3/junhaoge/waymo_simulation_dataset', type=str)
    # parser.add_argument("--specific_scene", default=['036','019','046','053',',058','096','382',], type=list)
    # parser.add_argument("--specific_scene", default=['019','036','382'], type=list)
    parser.add_argument("--specific_scene", default=['402','427'], type=list)


    # parser.add_argument("--specific_scene", default=['019','046','053',',058','096','427'], type=list)
    # parser.add_argument("--specific_scene", default=['036'], type=list)
    parser.add_argument("--all_scene", default=False, type=bool)
    args = parser.parse_args()
    sg_data_path = args.sg_data_path
    render_path = args.render_path
    dataset_output_path = args.dataset_output_path
    specific_scene = args.specific_scene
    all_scene = args.all_scene
    camera_ids = [0, 1, 2]
    img_witdh = 1600
    img_height = 900
    
    for render_output_path in os.listdir(args.render_path):
        scene_name = render_output_path.split('_')[-1]
        if not os.path.isdir(os.path.join(args.render_path, render_output_path)):
            continue
        if all_scene or scene_name in specific_scene:
            try:
                dataset_output_path_scene = os.path.join(args.dataset_output_path, scene_name)
                os.makedirs(dataset_output_path_scene, exist_ok=True)
                sg_data_path_scene = os.path.join(args.sg_data_path, scene_name)
                sg_data_path_scene_car_info = os.path.join(sg_data_path_scene, 'car_info')
                map_feature_path = os.path.join(sg_data_path_scene_car_info, 'map_feature.json')
                scene_data_path = os.path.join(args.render_path, render_output_path, 'trajectory')
                for scene_exp in os.listdir(scene_data_path):
                    if not os.path.isdir(os.path.join(scene_data_path, scene_exp)):
                        continue
                    print(f'processing scene {scene_name} exp {scene_exp}')
                    start_time = time.time()
                    cur_time = start_time
                    scene_exp_path = os.path.join(scene_data_path, scene_exp)
                    valid_frame_idx = []
                    for file_name in os.listdir(scene_exp_path):
                        if file_name.endswith('rgb.png'):
                            valid_frame_idx.append(int(file_name.split('_')[0]))
                    valid_frame_idx = sorted(valid_frame_idx)
                    # print(valid_frame_idx)
                    scene_idx = scene_exp.split('_')[-1]
                    # if not '0' in scene_idx:
                    #     continue
                    scene_save_path = os.path.join(dataset_output_path_scene, scene_idx)
                    if os.path.exists(scene_save_path):
                        os.system(f'rm -r {scene_save_path}')
                    os.makedirs(scene_save_path, exist_ok=True)
                    map_feature_save_path = os.path.join(scene_save_path, 'map_feature.json')
                    os.system(f'cp {map_feature_path} {map_feature_save_path}')
                    car_dict_sequence_path = os.path.join(sg_data_path_scene_car_info, f'car_dict_sequence_{scene_idx}.json')
                    car_dict_sequence_save_path = os.path.join(scene_save_path, 'car_dict_sequence.json')
                    with open(car_dict_sequence_path, 'r') as f:
                        car_dict_sequence_ori_data = json.load(f)
                    for idx, item in enumerate(car_dict_sequence_ori_data):
                        for key, car_info in item.items():
                            if not 'static' in key:
                                car_info['loc'][2] = car_info['loc'][2] + car_info['bbox'][2] / 2
                            if car_info['type'] not in ['pedestrian', 'vehicle']:
                                car_info['type'] = 'vehicle'
                    os.system(f'cp {car_dict_sequence_path} {car_dict_sequence_save_path}')
                    intrinsic_path = os.path.join(sg_data_path_scene, 'intrinsics','0.txt')
                    camera_intrinsic = np.loadtxt(intrinsic_path)
                    ego_pose_ori_path = os.path.join(sg_data_path_scene_car_info, 'ego_pose_ori.json')
                    ego_poses = get_ego_vehicle_matrices(car_dict_sequence_save_path,ego_pose_ori_path)
                    agent_car_info = get_agent_vehicle_matrices(car_dict_sequence_save_path)
                    # 删除valid_frame_idx中重复的帧
                    valid_frame_idx = list(set(valid_frame_idx))
                    unique_frame_list = []
                    [unique_frame_list.append(x) for x in valid_frame_idx if x not in unique_frame_list]
                    unique_frame_list.sort()
                    # print(len(car_dict_sequence_ori_data))
                    for idx, frame_idx in enumerate(unique_frame_list):
                        cur_timestamp = cur_time + 5000
                        save_timestamp = round(cur_timestamp*1000)
                        # print(frame_idx)
                        frame_save_path = os.path.join(scene_save_path, f'{idx}')
                        os.makedirs(frame_save_path, exist_ok=True)
                        if frame_idx > len(car_dict_sequence_ori_data) - 1:
                            if os.path.exists(frame_save_path):
                                os.system(f'rm -r {frame_save_path}')
                            continue
                        frame_car_dict = car_dict_sequence_ori_data[frame_idx].copy()
                        frame_ego_dict = dict()
                        if 'ego_vehicle' in frame_car_dict.keys():
                            frame_ego_dict = frame_car_dict['ego_vehicle']
                            del frame_car_dict['ego_vehicle']
                        # 将frame_car_dict中所有type不为pedestrian和vehicle的变为vehicle
                        for key, car_info in frame_car_dict.items():
                            # if not 'static' in key:
                            #     car_info['loc'][2] = car_info['loc'][2] + car_info['bbox'][2] / 2
                            # if car_info['type'] not in ['pedestrian', 'vehicle']:
                            #     frame_car_dict[key]['type'] = 'vehicle'
                            if 'visible_camera_id' not in frame_car_dict[key].keys():
                                frame_car_dict[key]['visible_camera_id'] = []
                        cur_ego_pose = ego_poses[frame_idx]
                        # 转回到原始的ego_pose
                        ego_pose_ori_cor =  cur_ego_pose @ np.linalg.inv(opencv2camera)
                        roll, pitch, yaw, loc = disassemble_matrix(ego_pose_ori_cor)
                        frame_ego_dict['rot'] = [roll, pitch, yaw]
                        frame_ego_dict['loc'][2] = frame_ego_dict['loc'][2] + frame_ego_dict['bbox'][2] / 2
                        camera_calib_dict = dict()
                        camera_calib_dict['timestamp'] = save_timestamp
                        camera_ori_extrinsic_path = os.path.join(sg_data_path_scene, 'extrinsics')
                        concat_cameras = camera_ids
                        camera_ori_extrinsic = {}
                        for num_cam in range(len(concat_cameras)):
                            cam_id = concat_cameras[num_cam]
                            with open(os.path.join(camera_ori_extrinsic_path, f'{cam_id}.txt'), 'r') as f:
                                data = np.loadtxt(f)
                            camera_ori_extrinsic[cam_id] = data
                        # calculate the matrix from camera 0 to other cameras
                        camera_rotate_matrix = {}
                        for num_cam in range(len(concat_cameras)):
                            cam_id = concat_cameras[num_cam]
                            camera_rotate_matrix[cam_id] = np.eye(4)
                            camera_matric = camera_ori_extrinsic[cam_id]
                            # calculate the rotation matrix from camera 0 to camera cam_id
                            camera_rotate_matrix[cam_id][:3, :3] = np.dot(camera_matric[:3, :3], np.linalg.inv(camera_ori_extrinsic[0][:3, :3]))
                        for cam_id in camera_ids:
                            # print(f'processing scene {scene_name} exp {scene_exp} frame {frame_idx} cam {cam_id}')
                            rgb_path = os.path.join(scene_exp_path, f'{frame_idx}_{cam_id}_rgb.png')
                            # copy rgb image
                            rgb_save_path = os.path.join(frame_save_path, f'{idx}_{cam_id}_rgb.png')
                            # resize image to 1600*900
                            img = cv2.imread(rgb_path)
                            # 裁剪上方的图片，只保留下方900像素
                            img_height, img_witdh, _ = img.shape
                            img_height_start = int((img_height - 900))
                            img = img[img_height_start:, :]
                            cv2.imwrite(rgb_save_path, img)
                            depth_path = os.path.join(scene_exp_path, f'{frame_idx}_{cam_id}_depth.npy')
                            # 根据 cam_id 判断旋转角度并调整 cur_ego_pose
                            if cam_id == 0:
                                # 不变，直接使用原始的 cur_ego_pose
                                modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                            elif cam_id == 1:
                                # 向左旋转 25.2°
                                modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                                yaw_0 = math.atan2(camera_ori_extrinsic[0][1, 0], camera_ori_extrinsic[0][0, 0])
                                yaw_1 = math.atan2(camera_ori_extrinsic[1][1, 0], camera_ori_extrinsic[1][0, 0])
                                yaw_diff = abs(yaw_1 - yaw_0)
                                if yaw_diff < math.radians(60):
                                    yaw_diff = math.radians(60) - yaw_diff
                                    modified_ego_pose = rotate_matrix_by_deg(modified_ego_pose, math.degrees(yaw_diff))
                            elif cam_id == 2:
                                # 向右旋转 25.2°
                                modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                                yaw_0 = math.atan2(camera_ori_extrinsic[0][1, 0], camera_ori_extrinsic[0][0, 0])
                                yaw_2 = math.atan2(camera_ori_extrinsic[2][1, 0], camera_ori_extrinsic[2][0, 0])
                                yaw_diff = abs(yaw_2 - yaw_0)
                                if yaw_diff < math.radians(60):
                                    yaw_diff = math.radians(60) - yaw_diff
                                    modified_ego_pose = rotate_matrix_by_deg(modified_ego_pose, math.degrees(yaw_diff))
                            elif cam_id == 3:
                                modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                                # pitch_deg, roll_deg = compute_look_down_roll_pitch(modified_ego_pose[:3, :3], total_angle=15)
                                # modified_ego_pose = rotate_matrix(modified_ego_pose, 0.0, pitch_deg=pitch_deg, roll_deg=roll_deg)
                            elif cam_id == 4:
                                modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                                # pitch_deg, roll_deg = compute_look_down_roll_pitch(modified_ego_pose[:3, :3], total_angle=15)
                                # modified_ego_pose = rotate_matrix(modified_ego_pose, 0.0, pitch_deg=-pitch_deg, roll_deg=-roll_deg)
                            else:
                                # 如果有其他未定义的 cam_id, 这里可以根据需求处理，默认直接使用 cur_ego_pose
                                modified_ego_pose = cur_ego_pose
                            extrinsic = modified_ego_pose
                            cam_loc = extrinsic[:3, 3]
                            # camera_intrinsic_new = get_intrinsic(camera_intrinsic)
                            camera_intrinsic_new = camera_intrinsic_transform()
                            calibration = {
                                'intrinsic': camera_intrinsic_new,
                                'extrinsic': extrinsic
                            }
                            # x_factor = 1600 / 1600
                            # y_factor = 900 / 1066
                            # camera_intrinsic_1600_900 = camera_intrinsic_new.copy()
                            # camera_intrinsic_1600_900[0, 0] = camera_intrinsic_new[0, 0] * x_factor
                            # camera_intrinsic_1600_900[1, 1] = camera_intrinsic_new[1, 1]
                            # camera_intrinsic_1600_900[0, 2] = camera_intrinsic_new[0, 2] * x_factor
                            # camera_intrinsic_1600_900[1, 2] = camera_intrinsic_new[1, 2] - img_height_start
                            extrinsic_to_save = extrinsic @ np.linalg.inv(opencv2camera)
                            calibration_to_save = {
                                'intrinsic': camera_intrinsic_new.tolist(),
                                'extrinsic': extrinsic_to_save.tolist()
                            }
                            camera_calib_dict[cam_id] = calibration_to_save
                            agent_valid = []
                            points_valid = []
                            name_valid = []
                            cur_agent_car_info = agent_car_info[frame_idx]
                            for key, car_info in cur_agent_car_info.items():
                                dim = car_info['bbox']
                                loc = car_info['loc']
                                R_z = rpy2R([car_info['rot'][0], car_info['rot'][1], car_info['rot'][2]])
                                T_vehicle_to_world = np.eye(4)
                                T_vehicle_to_world[:3, :3] = R_z  
                                T_vehicle_to_world[:3, 3] = loc   
                                T_vehicle_to_world[3, 3] = 1
                                points_uv, valid = project_label_to_image(dim, T_vehicle_to_world, calibration, img_witdh, img_height)
                                if valid.all():
                                    points_valid.append(points_uv)
                                    name_valid.append(key)
                                    agent_valid.append(car_info)
                            filter_results_name = []
                            depth = np.load(depth_path)
                            for idx_valid, points_uv in enumerate(points_valid):
                                car_info = agent_valid[idx_valid]
                                car_loc = car_info['loc']
                                distance = calculate_distance(car_loc, cam_loc)
                                if distance > 100:
                                    continue
                                car_center_pixel = np.mean(points_uv, axis=0)
                                pixel_x = int(car_center_pixel[0])
                                pixel_y = int(car_center_pixel[1])
                                if car_center_pixel[0] < 0:
                                    pixel_x = 0
                                if car_center_pixel[0] >= img_witdh:
                                    pixel_x = img_witdh - 1
                                if car_center_pixel[1] < 0:
                                    pixel_y = 0
                                if car_center_pixel[1] >= img_height:
                                    pixel_y = img_height - 1

                                car_center_depth = depth[pixel_y, pixel_x][0]
                                # print('distance:', distance, 'depth:', car_center_depth)
                                if abs(distance - car_center_depth) < 5.0:
                                    frame_car_dict[name_valid[idx_valid]]['visible_camera_id'].append(cam_id)
                                else:
                                    valid_num = 0
                                    for point in points_uv:
                                        pixel_x = int(point[0])
                                        pixel_y = int(point[1])
                                        if pixel_x < 0 or pixel_x >= img_witdh or pixel_y < 0 or pixel_y >= img_height:
                                            continue
                                        depth_value = depth[pixel_y, pixel_x][0]
                                        if abs(distance - depth_value) < 5.0:
                                            valid_num += 1

                                    if valid_num >= 4:
                                        frame_car_dict[name_valid[idx_valid]]['visible_camera_id'].append(cam_id)
                        frame_car_dict_save_path = os.path.join(frame_save_path, 'agent_info.json')
                        with open(frame_car_dict_save_path, 'w') as f:
                            json.dump(frame_car_dict, f, indent=2)
                        frame_ego_dict_save_path = os.path.join(frame_save_path, 'ego_pose.json')
                        with open(frame_ego_dict_save_path, 'w') as f:
                            json.dump(frame_ego_dict, f, indent=2)
                        camera_calib_dict_save_path = os.path.join(frame_save_path, 'camera_calib.json')
                        with open(camera_calib_dict_save_path, 'w') as f:
                            json.dump(camera_calib_dict, f, indent=2)
            except:
                pass


                

                
                

            
        
