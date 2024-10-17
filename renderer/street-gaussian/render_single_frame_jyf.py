import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
# import pycuda.driver as cuda
# import pycuda.autoinit
# import pynvml
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "11"
# pynvml.nvmlInit()

# def get_free_gpu():
#     num_gpus = cuda.Device.count()
#     free_mem_list = []

#     # 遍历所有 GPU，并获取每个 GPU 的空闲显存
#     for i in range(num_gpus):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         free_mem_list.append((i, mem_info.free))

#     # 选择空闲显存最大的 GPU
#     best_gpu = max(free_mem_list, key=lambda x: x[1])[0]
#     return best_gpu
# # 手动设置使用的 GPU 设备
# cuda.init()
# # 获取最空闲的 GPU
# best_gpu = get_free_gpu()
# print(f"Selecting GPU: {best_gpu}")

# # 使用 PyCUDA 设置为该 GPU
# cuda.Device(best_gpu).make_context()
opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

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

    ego_z = [loc[2] + 2 for loc in ego_loc]
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

def rotate_matrix(matrix, theta_deg):
    """
    Rotate a 4x4 matrix around the Z-axis by a specified angle.

    Parameters:
        matrix (np.ndarray): The original 4x4 transformation matrix.
        theta_deg (float): The rotation angle in degrees (default is 25.2°).

    Returns:
        np.ndarray: The new 4x4 matrix after rotation.
    """
    # Convert angle from degrees to radians
    theta_rad = np.deg2rad(theta_deg)
    
    # Define the Z-axis rotation matrix
    Rz = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad),  np.cos(theta_rad), 0],
        [0,                  0,                 1]
    ])
    
    # Extract the original rotation and translation components from the input matrix
    original_rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    # Apply the Z-axis rotation to the original rotation
    new_rotation = Rz @ original_rotation
    
    # Reconstruct the new 4x4 transformation matrix
    new_matrix = np.eye(4)  # Create an identity matrix for the base
    new_matrix[:3, :3] = new_rotation
    new_matrix[:3, 3] = translation
    
    return new_matrix


def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getEgoCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))
                
def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)  
            visualizer.visualize(result, camera)

        visualizer.summarize()
        

def render_image():
    cfg.render.save_image = True
    cfg.render.save_video = False
    source_path = cfg.source_path
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))
        ego_car_dict = os.path.join(source_path, 'car_info/car_dict_sequence.json')
        ego_poses = get_ego_vehicle_matrices(ego_car_dict)
        
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            ori_ext = camera.get_extrinsic()
            new_ext = ori_ext
            cur_ego_pose = ego_poses[idx]
            camera.set_extrinsic(cur_ego_pose)
            result = renderer.render(camera, gaussians)  
            visualizer.visualize(result, camera)

def render_novel():
    cfg.render.save_image = True
    cfg.render.save_video = True
    source_path = cfg.source_path
    ckpt_path = os.path.join(source_path, 'agent.pth')

    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        if not os.path.exists(ckpt_path):
            print("\n Saving Checkpoint for agent")
            state_dict = gaussians.save_state_dict(is_final=True)
            # state_dict['iter'] = iteration
            torch.save(state_dict, ckpt_path)
        else:
            state_dict = torch.load(ckpt_path)
            
            print(f'Loading agent model from {ckpt_path}')
            gaussians.load_state_dict(state_dict)


        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))
        ego_poses_ori_json = os.path.join(source_path, 'car_info/ego_pose_ori.json')
        # ego_poses_ori_json = '/home/ubuntu/longtengfan/RT-bkgd/data/waymo/031/
        ego_car_dict = os.path.join(source_path, 'car_info/car_dict_sequence.json')
        ego_poses = get_ego_vehicle_matrices(ego_car_dict, ego_poses_ori_json)
        basic_camera = cameras[0]
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

        concat_cameras = cfg.render.get('concat_cameras', [])

        for idx, cur_ego_pose in enumerate(tqdm(ego_poses, desc="Rendering Trajectory")):
            ego_loc = cur_ego_pose[:3, 3].reshape(1, 3)
            closest_idx = kd_tree_ego_poses.query(ego_loc)[1][0]
            if closest_idx >= len(cameras):
                closest_idx = len(cameras) - 1
            basic_camera = cameras[closest_idx]
            for num_cam in range(len(concat_cameras)):
                cam_id = concat_cameras[num_cam]
                basic_camera.image_name = f"{idx}_{cam_id}"
                basic_camera.set_render_frame_idx(idx)
                # 根据 cam_id 判断旋转角度并调整 cur_ego_pose
                if cam_id == 0:
                    # 不变，直接使用原始的 cur_ego_pose
                    modified_ego_pose = cur_ego_pose
                elif cam_id == 1:
                    # 向左旋转 25.2°
                    modified_ego_pose = rotate_matrix(cur_ego_pose, 25.2)
                elif cam_id == 2:
                    # 向右旋转 25.2°
                    modified_ego_pose = rotate_matrix(cur_ego_pose, -25.2)
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
        # render_image()
