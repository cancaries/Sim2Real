# create a video from a list of images from a directory

import cv2
import os
import re
import numpy as np
from moviepy.editor import VideoFileClip
import time
import datetime
import concurrent.futures
import imageio
from PIL import Image
# def create_video_from_images(image_folder, video_name, fps=30):
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#     images = sorted(images, key=lambda x: int(re.findall(r'\d+', x)[0]))
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     # 降低图片分辨率至480p
#     frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
#     height, width, layers = frame.shape
#     video = cv2.VideoWriter(video_name, 0, fps, (width, height))
#     img_num = 0
#     img_num_threshold = 200
#     for image in images:
#         img_num += 1
#         # if img_num > img_num_threshold:
#         #     break
#         frame = cv2.imread(os.path.join(image_folder, image))
#         # 降低图片分辨率至480p
#         frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
#         video.write(frame)
#     cv2.destroyAllWindows()
#     video.release()

# def Video2Mp4(videoPath, outVideoPath):
#     capture = cv2.VideoCapture(videoPath)
#     fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
#     size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     # fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     suc = capture.isOpened()  # 是否成功打开

#     allFrame = []
#     while suc:
#         suc, frame = capture.read()
#         if suc:
#             allFrame.append(frame)
#     capture.release()

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     videoWriter = cv2.VideoWriter(outVideoPath, fourcc, fps, size)
#     for aFrame in allFrame:
#         videoWriter.write(aFrame)
#     videoWriter.release()

def process_image(file_name):
    if file_name.endswith(".png"):
        image = Image.open(file_name)
    # resize image to 960p
    image = image.resize((1280, 960), Image.ANTIALIAS)
    return np.array(image.convert("RGB"))

def create_video_from_images(image_folder, video_name, fps=30):
    images = []
    # 寻找所有 png 文件
    image_files = [img for img in os.listdir(image_folder) if img.endswith(".png")]
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


if __name__ == '__main__':
    # # time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # comment = 'visualize_traffic'
    # image_folder = '/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/001_20240920_145846'
    # time_str = image_folder.split('_')[-2] + '_' + image_folder.split('_')[-1]
    # scene_name = image_folder.split('/')[-1]
    # output_path = f'/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/{comment}_{time_str}.avi'
    # output_path = f'/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/{scene_name}.avi'

    # create_video_from_images(image_folder, output_path, 10)
    # # # 将生成的视频文件转换为mp4
    # # os.system(f'/usr/bin/ffmpeg -i {output_path} {output_path.replace(".avi", ".mp4")}')
    # # 读取 AVI 文件
    # clip = VideoFileClip(output_path)

    # # 将视频转换为 MP4 格式
    # clip.write_videofile(f'{output_path.replace(".avi", ".mp4")}')
    # # remove avi
    # os.remove(output_path)
    # # Video2Mp4(output_path, output_path.replace(".avi", ".mp4"))
    scene_img_folder = '/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/20240928184236/0/022/traffic_pic'
    output_path = '/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/20240928184236/0/video/022.mp4'
    valid = create_video_from_images(scene_img_folder, output_path, 10)