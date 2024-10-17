import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import json
import numpy as np
import math
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
import torch.cuda as cuda
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
def moving_average(z, window_size):
    pad_size = window_size // 2
    z_padded = np.pad(z, (pad_size, pad_size), mode='edge')
    return np.convolve(z_padded, np.ones(window_size)/window_size, mode='valid')

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


opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

def camera_intrinsic_transform(vfov=35,hfov=60,pixel_width=1600,pixel_height=900):
    camera_intrinsics = np.zeros((3,3))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics


def camera_intrinsic_fov(intrinsic):
    #计算FOV
    w, h = intrinsic[0][2]*2, intrinsic[1][2]*2
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    # Go
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))
    return fov_x, fov_y

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

    ego_z = [loc[2] for loc in ego_loc]
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

def render_novel():
    cfg.render.save_image = True
    cfg.render.save_video = True
    source_path = cfg.source_path
    scene_name = source_path.split('/')[-1]
    scene_number = cfg.scene_number
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', f"{scene_name}_{scene_number}")
        if os.path.exists(save_dir):
            os.system(f'rm -rf {save_dir}')
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))
        ego_poses_ori_json = os.path.join(source_path, 'car_info/ego_pose_ori.json')
        # ego_poses_ori_json = '/home/ubuntu/longtengfan/RT-bkgd/data/waymo/031/
        if scene_number is None:  
            ego_car_dict = os.path.join(source_path, 'car_info/car_dict_sequence_0.json')
        else:
            ego_car_dict = os.path.join(source_path, 'car_info/car_dict_sequence_{}.json'.format(scene_number))
        # ego_car_dict = os.path.join(source_path, 'car_info/car_dict_sequence.json')
        ego_poses = get_ego_vehicle_matrices(ego_car_dict, ego_poses_ori_json)
        basic_camera = cameras[0]
        basic_z = basic_camera.get_extrinsic()[2, 3]
        offset_z = abs(basic_z - ego_poses[0][2, 3])
        for idx, ego_pose in enumerate(ego_poses):
            ego_pose[2, 3] += 2.11
            ego_poses[idx] = ego_pose
        basic_pitch = basic_camera.get_pitch()
        pitch_offset = basic_pitch - math.asin(ego_poses[0][1, 2])
        ego_poses_ref = []
        ego_poses_indices = []
        with open(ego_poses_ori_json, 'r') as f:
            ego_poses_ori = json.load(f)
        for idx, item in enumerate(ego_poses_ori):
            loc = item['location']
            rot = item['rotation']
            ego_poses_ref.append(loc)
            ego_poses_indices.append(rot)
        kd_tree_ego_poses = KDTree(ego_poses_ref)
        camera_ori_extrinsic_path = os.path.join(source_path, 'extrinsics')
        concat_cameras = cfg.render.get('concat_cameras', [])
        camera_ori_extrinsic = {}
        camera_standard_intrinsic = camera_intrinsic_transform()
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


        for idx, cur_ego_pose in enumerate(tqdm(ego_poses, desc="Rendering Trajectory")):
            if idx < 20:
                continue
            if idx % 5 != 0:
                continue
            ego_loc = cur_ego_pose[:3, 3].reshape(1, 3)
            closest_idx = kd_tree_ego_poses.query(ego_loc)[1][0]
            if closest_idx >= len(cameras):
                closest_idx = len(cameras) - 1
            basic_camera = cameras[closest_idx]
            # basic_camera_fov = camera_intrinsic_fov(basic_camera.get_intrinsic())
            # print(f"Camera {closest_idx} FOV: {basic_camera_fov}")
            # print(f"Camera {closest_idx} pitch: {camera_intrinsic_fov(camera_standard_intrinsic)}")
            for num_cam in range(len(concat_cameras)):
                cam_id = concat_cameras[num_cam]
                basic_camera.image_name = f"{idx}_{cam_id}"
                basic_camera.set_render_frame_idx(idx)
                # print(basic_camera.FoVx)
                # basic_camera.set_FOV(70*np.pi/180)
                basic_camera.set_intrinsic(camera_standard_intrinsic)
                # 根据 cam_id 判断旋转角度并调整 cur_ego_pose
                if cam_id == 0:
                    # 不变，直接使用原始的 cur_ego_pose
                    modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                elif cam_id == 1:
                    # 向左旋转 25.2°
                    modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                    # 计算0与1的yaw的夹角，如果不满60°则额外旋转不足的角度
                    yaw_0 = math.atan2(camera_ori_extrinsic[0][1, 0], camera_ori_extrinsic[0][0, 0])
                    yaw_1 = math.atan2(camera_ori_extrinsic[1][1, 0], camera_ori_extrinsic[1][0, 0])
                    yaw_diff = abs(yaw_1 - yaw_0)
                    if yaw_diff < math.radians(60):
                        yaw_diff = math.radians(60) - yaw_diff
                        modified_ego_pose = rotate_matrix_by_deg(modified_ego_pose, math.degrees(yaw_diff))
                elif cam_id == 2:
                    # 向右旋转 25.2°
                    modified_ego_pose = rotate_matrix(cur_ego_pose, camera_rotate_matrix[cam_id])
                    # 计算0与2的yaw的夹角，如果不满60°则额外旋转不足的角度
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
                # 设置相机的外部参数为修改后的姿态
                basic_camera.set_extrinsic(modified_ego_pose)

                # 更新 basic_camera 的 ID 为当前 cam_id
                basic_camera.meta['cam'] = cam_id  # 将相机 ID 更新为当前 cam_id
                # print(basic_camera.ego_pose)

                result = renderer.render_all(basic_camera, gaussians)  
                visualizer.visualize(result, basic_camera)

        visualizer.summarize()

            
if __name__ == "__main__":
    with torch.cuda.device(0):
        print("Rendering " + cfg.model_path)
        safe_state(cfg.eval.quiet)
        render_novel()
