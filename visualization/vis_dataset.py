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

    valid_pixel = [valid_x.count(True) >= 3 and valid_y.count(True) >= 3]
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
            box_z = ego_vehicle['bbox'][2]

    ego_z = [loc[2] + 2.11 - box_z/2 for loc in ego_loc]
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
            # if not 'static' in key:
            #     loc[2] = loc[2] + vehicle_info['bbox'][2] / 2
            tmp_agent_vehcile_info[key] = vehicle_info
            tmp_agent_vehcile_info[key]['loc'] = loc

        agent_vehicle_info.append(tmp_agent_vehcile_info)

    return agent_vehicle_info

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

# save simulation dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='/home/ubuntu/DATA3/junhaoge/waymo_simulation_dataset', type=str)
    # parser.add_argument("--data_path", default='/home/ubuntu/DATA3/junhaoge/waymo_simulation_dataset/022/0', type=str)
    parser.add_argument("--save_path", default='/home/ubuntu/DATA3/junhaoge/waymo_simulation_dataset/vis', type=str)
    args = parser.parse_args()
    data_root = args.data_root
    save_path = args.save_path
    camera_ids = [0, 1, 2]
    img_witdh = 1600
    img_height = 900
    for scene_name in os.listdir(data_root):
        if '022' in scene_name or 'vis' in scene_name:
            continue
        if scene_name not in ['019']:
            continue
        scene_path = os.path.join(data_root, scene_name)
        for scene_idx in os.listdir(scene_path):
            # if scene_idx not in ['0']:
            #     continue
            scene_data_path = os.path.join(scene_path, scene_idx)
            # scene_name = data_path.split('/')[-2]
            # scene_idx = data_path.split('/')[-1]
            scene_save_path = os.path.join(save_path, scene_name, scene_idx)
            os.makedirs(scene_save_path, exist_ok=True)
            frame_idx_list = []
            frame_idx_list = [int(frame_idx) for frame_idx in os.listdir(scene_data_path) if frame_idx.isdigit()]
            frame_idx_list.sort()
            for frame_idx in frame_idx_list:
                frame_path = os.path.join(scene_data_path, str(frame_idx))
                camera_calib_path = os.path.join(frame_path, 'camera_calib.json')
                with open(camera_calib_path, 'r') as f:
                    camera_calib = json.load(f)
                agent_car_info_path = os.path.join(frame_path, 'agent_info.json')
                with open(agent_car_info_path, 'r') as f:
                    agent_car_info = json.load(f)
                for cam_id in camera_ids:
                    rgb_path = os.path.join(frame_path, f'{frame_idx}_{cam_id}_rgb.png')
                    extrinsic = np.array(camera_calib[str(cam_id)]['extrinsic'])
                    extrinsic = extrinsic @ opencv2camera
                    cam_loc = extrinsic[:3, 3]
                    calibration = {
                        'intrinsic': np.array(camera_calib[str(cam_id)]['intrinsic']),
                        'extrinsic': extrinsic
                    }
                    points_valid = []
                    key_valid = []
                    cur_agent_car_info = agent_car_info
                    for key, car_info in cur_agent_car_info.items():
                        if not cam_id in car_info['visible_camera_id']:
                            continue
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
                            key_valid.append(key)
                    img = cv2.imread(rgb_path)
                    for idx_valid, points_uv in enumerate(points_valid):
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
                        points_uv = points_uv.astype(np.int32)
                        cv2.line(img, tuple(points_uv[0]), tuple(points_uv[1]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[1]), tuple(points_uv[3]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[3]), tuple(points_uv[2]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[2]), tuple(points_uv[0]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[4]), tuple(points_uv[5]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[5]), tuple(points_uv[7]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[7]), tuple(points_uv[6]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[6]), tuple(points_uv[4]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[0]), tuple(points_uv[4]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[1]), tuple(points_uv[5]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[2]), tuple(points_uv[6]), (0, 255, 0), 2)
                        cv2.line(img, tuple(points_uv[3]), tuple(points_uv[7]), (0, 255, 0), 2)
                        cv2.putText(img, key_valid[idx_valid], (pixel_x, pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # save image
                    cv2.imwrite(os.path.join(scene_save_path, f'{frame_idx}_{cam_id}_rgb.png'), img)
                # concat images
                # images = []
                # for cam_id in [3,1,0,2,4]:
                #     img = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_{cam_id}_rgb.png'))
                #     images.append(img)
                # concat_img = np.concatenate(images, axis=1)
                # 将img分两行合并，第一行的idx是1,0,2，第二行的idx是3,4
                images = []
                img_0_sample = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_0_rgb.png'))
                img_height, img_width, _ = img_0_sample.shape
                for cam_id in [1,0,2]:
                    img = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_{cam_id}_rgb.png'))
                    images.append(img)
                concat_img1 = np.concatenate(images, axis=1)
                # # 3,4中间增加一片空白区域进行合并，以使得第一行和第二行的图片对齐
                # blank_img = np.zeros((img_height, img_width, 3), np.uint8)
                # white = (255, 255, 255)
                # blank_img[:] = white
                # images = []
                # img = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_3_rgb.png'))
                # images.append(img)
                # images.append(blank_img)
                # img = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_4_rgb.png'))
                # images.append(img)
                # concat_img2 = np.concatenate(images, axis=1)
                # concat_img = np.concatenate([concat_img1, concat_img2], axis=0)
                cv2.imwrite(os.path.join(scene_save_path, f'{frame_idx}_concat_rgb.png'), concat_img1)
                # cv2.imwrite(os.path.join(scene_save_path, f'{frame_idx}_concat_rgb.png'), concat_img)
            # create video
            video_name = os.path.join(scene_save_path, f'{scene_name}_{scene_idx}.mp4')
            create_video_from_images(scene_save_path, video_name, fps=2)
            # 删除所有图片
            for file in os.listdir(scene_save_path):
                if file.endswith(".png"):
                    os.remove(os.path.join(scene_save_path, file))
            print(f'Create video: {video_name}')





                    



                    

                    
                    

                
            
