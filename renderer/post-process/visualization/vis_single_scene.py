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

def process_image(file_name):
    if file_name.endswith(".png"):
        image = Image.open(file_name)
    # resize image to 960p
    # image = image.resize((1280, 960), Image.ANTIALIAS)
    return np.array(image.convert("RGB"))

def create_video_from_images(image_folder, video_name, fps=30):
    images = []
    # 寻找所有 png 文件
    # image_files = [img for img in os.listdir(image_folder) if img.endswith(".png") and '_relight' in img and 'concat' not in img]
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.split('_')[1] == 'relight.png']
    if len(image_files) < fps * 2:
        return False
    # 按文件名中的数字排序
    image_files = sorted(image_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    # 获取第一个图像并设定尺寸
    first_image_path = os.path.join(image_folder, image_files[0])
    with Image.open(first_image_path) as img:
        target_size = img.size  # 设定目标尺寸为第一个图像的尺寸
    # print(image_files)

    # 利用线程池并行处理图像，并调整为统一尺寸
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = process_image(image_path)
        
        # 转换为 PIL 图像并调整尺寸
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize(target_size)
        
        # 转换回 NumPy 数组并添加到列表中
        images.append(np.array(resized_image))
        
    with imageio.get_writer(video_name, fps=fps) as video:
        for image in images:
            video.append_data(image)
    
    return True


# save simulation dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='/media/magic-4090/47236903-9d2a-4bc7-9828-df4fa4b40bd0/user/blender_workspace/IC-Light/output', type=str)
    args = parser.parse_args()
    data_root = args.data_root
    for scene_name in os.listdir(data_root):
        scene_path = os.path.join(data_root, scene_name)
        scene_data_path = scene_path
        scene_save_path = scene_data_path
        os.makedirs(scene_save_path, exist_ok=True)
        frame_list = [x.split('_')[0] for x in os.listdir(scene_data_path) if x.endswith('.png')]
        cam_id_list = [x.split('_')[1] for x in os.listdir(scene_data_path) if x.endswith('.png')]
        # 删除重复值
        frame_list = list(set(frame_list))
        cam_id_list = list(set(cam_id_list))
        frame_list.sort()
        for frame_idx in frame_list:
            images = []
            img_0_sample = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_0_relight.png'))
            img_height, img_width, _ = img_0_sample.shape
            for cam_id in [1,0,2]:
                img = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_{cam_id}_relight.png'))
                images.append(img)
            concat_img1 = np.concatenate(images, axis=1)
            # # 3,4中间增加一片空白区域进行合并，以使得第一行和第二行的图片对齐
            blank_img = np.zeros((img_height, img_width, 3), np.uint8)
            white = (255, 255, 255)
            blank_img[:] = white
            images = []
            img = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_3_relight.png'))
            images.append(img)
            images.append(blank_img)
            img = cv2.imread(os.path.join(scene_save_path, f'{frame_idx}_4_relight.png'))
            images.append(img)
            concat_img2 = np.concatenate(images, axis=1)
            concat_img = np.concatenate([concat_img1, concat_img2], axis=0)
            cv2.imwrite(os.path.join(scene_save_path, f'{frame_idx}_concat_relight.png'), concat_img)
            # create video
        video_name = os.path.join(scene_save_path, f'{scene_name}.mp4')
        create_video_from_images(scene_save_path, video_name, fps=4)
        # 删除所有图片
        # for file in os.listdir(scene_save_path):
        #     if file.endswith(".png"):
        #         os.remove(os.path.join(scene_save_path, file))
        print(f'Create video: {video_name}')





                    



                    

                    
                    

                
            
