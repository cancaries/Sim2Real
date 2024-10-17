
import time
from scene_wo_render import Scene_wo_render
import random
from agents.navigation.waypoint import Transform
import yaml
import json
import os
import argparse
from tqdm import tqdm
import cv2
import os
import re
from moviepy.editor import VideoFileClip
def create_video_from_images(image_folder, video_name, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if len(images) < fps * 2:
        return False
    images = sorted(images, key=lambda x: int(re.findall(r'\d+', x)[0]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    # 降低图片分辨率至480p
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, fps, (width, height))
    img_num = 0
    img_num_threshold = 200
    
    for image in images:
        img_num += 1
        # if img_num > img_num_threshold:
        #     break
        frame = cv2.imread(os.path.join(image_folder, image))
        # 降低图片分辨率至480p
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()
    return True

def Video2Mp4(videoPath, outVideoPath):
    capture = cv2.VideoCapture(videoPath)
    fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    suc = capture.isOpened()  # 是否成功打开

    allFrame = []
    while suc:
        suc, frame = capture.read()
        if suc:
            allFrame.append(frame)
    capture.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(outVideoPath, fourcc, fps, size)
    for aFrame in allFrame:
        videoWriter.write(aFrame)
    videoWriter.release()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_setting_config', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/agent_control/agent_configs/default.yaml')
    args = parser.parse_args()
    scene_setting_config = yaml.load(open(args.scene_setting_config, 'r'), Loader=yaml.FullLoader)
    random.seed(scene_setting_config['simulation']['random_seed'])
    # 地图配置文件路径
    data_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view'
    start_time = time.strftime("%m%d%H%M%S", time.localtime())
    skip_frames = 5
    for scene_name in os.listdir(data_path):
        print('Simulating scene: ', scene_name)
        if not scene_name.isdigit():
            continue
        # if not scene_name == '031':
        #     continue
        if scene_name in ['001','016','023']:
            continue
        _data_root = os.path.join(data_path, scene_name)
        map_config_path = os.path.join(_data_root, 'map_feature.json')
        ego_pose_path = os.path.join(_data_root, 'ego_pose.json')
        static_actor_config_path = os.path.join(_data_root, 'all_static_actor_data.json')
        scene_config = dict()
        scene_config['_data_root'] = _data_root
        scene_config['_map_config_path'] = map_config_path
        scene_config['_ego_pose_path'] = ego_pose_path
        scene_config['_static_actor_config_path'] = static_actor_config_path
        scene_config['_scene_name'] = _data_root.split('/')[-1]
        scene_config['_FPS'] = scene_setting_config['simulation']['FPS']
        scene_config['mode'] = scene_setting_config['simulation']['mode']
        scene_config['ego_vehicle_config'] = scene_setting_config['ego_vehicle']
        scene_config['other_agents_config'] = scene_setting_config['other_agents']
        scene_config['agent_spawn_config'] = scene_setting_config['simulation']['agent_spawn']
        Scene_wo_render.initialize_scene(scene_config)
        
        # 让背景车辆按照agent行为以帧为单位进行行动,agent.run_step输出的是steer,throttle,brake这些控制信号，根据这些控制信号以0.2s为间隔进行计算
        # 请注意，这里的agent是一个字典，key为agent的名字，value为agent对象
        scene_length = scene_setting_config['simulation']['max_steps']
        stay_time_steps = scene_setting_config['simulation']['ego_control_steps']
        # stay_time_steps = len(transform_path) - 1
        end_route_flag = False
        
        for i in tqdm(range(scene_length)):
            if Scene_wo_render._end_scene:
                break
            for agent_name, agent in Scene_wo_render._agent_dict.items():
                agent_control = agent.run_step()
                Scene_wo_render._agent_control_dict[agent_name] = agent_control
            if end_route_flag:
                break
            if i < stay_time_steps:
                Scene_wo_render.run_step_w_ego(i)
            else:
                Scene_wo_render.run_step_w_ego(i, True)
        # 生成背景车辆的轨迹图
        if Scene_wo_render._mode == 'debug':
            Scene_wo_render._map.draw_map_w_traffic_flow_sequence(Scene_wo_render._car_dict_sequence,text=f'{start_time}/{scene_name}',skip_frames=skip_frames)
        elif Scene_wo_render._mode == 'datagen':
            Scene_wo_render.save_traffic_flow(Scene_wo_render._car_dict_sequence,text=f'{start_time}/{scene_name}',skip_frames=skip_frames)
        Scene_wo_render.reset()
    # 生成视频
    scenes_folder = f'/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/{start_time}'
    video_path = os.path.join(scenes_folder, 'video')
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    # scenes_folder = f'/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/{start_time}'
    for scene_name in os.listdir(scenes_folder):
        if scene_name == 'video':
            continue
        scene_folder = os.path.join(scenes_folder, scene_name)
        scene_name = scene_folder.split('/')[-1]
        output_path = os.path.join(video_path,f'{scene_name}.avi')
        valid = create_video_from_images(scene_folder, output_path, int(scene_setting_config['simulation']['FPS']/skip_frames))
        # # 将生成的视频文件转换为mp4
        # os.system(f'/usr/bin/ffmpeg -i {output_path} {output_path.replace(".avi", ".mp4")}')
        # 读取 AVI 文件
        if valid:
            clip = VideoFileClip(output_path)

            # 将视频转换为 MP4 格式
            clip.write_videofile(f'{output_path.replace(".avi", ".mp4")}')
            # remove avi
            os.remove(output_path)


if __name__ == '__main__':
    main()