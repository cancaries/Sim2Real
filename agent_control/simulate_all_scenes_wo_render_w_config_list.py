
import time
import random
import yaml
import json
import os
import argparse
import os
import re
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image
from agents.navigation.waypoint import Transform
from scene_wo_render import Scene_wo_render


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_setting_config_list', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/agent_control/agent_configs')
    args = parser.parse_args()
    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    simulate_num = 10
    skip_frames = 5
    max_agent_num = None
    exclude_scenes = ['001','023','047','205','007','015','044','056']
    # prefer_scenes = ['036','093','022','016','019']
    simple_scenes = ['036','019','053','382','402','427']
    highway_scenes = ['046','096']
    # prefer_scenes = ['019']
    # prefer_scenes = []
    for scene_setting_config in os.listdir(args.scene_setting_config_list):
        gen_idx = 0
        if not scene_setting_config.endswith('.yaml'):
            continue
        if 'debug' in scene_setting_config:
            continue
        # if not 'test' in scene_setting_config:
        #     continue
        if 'test' in scene_setting_config:
            continue
        if 'slow' in scene_setting_config:
            continue
        if 'highway' in scene_setting_config:
            prefer_scenes = highway_scenes
            simulation_start_num = 10
            max_agent_num = 20
        else:
            prefer_scenes = simple_scenes
        
        if 'wo_agent' in scene_setting_config:
            simulation_start_num = 20
            max_agent_num = 0
        elif 'more_agent' in scene_setting_config:
            simulation_start_num = 10
            max_agent_num = 20
        else:
            max_agent_num = None

        gen_idx += simulation_start_num
        scene_setting_config_path = os.path.join(args.scene_setting_config_list, scene_setting_config)
        scene_setting_config = yaml.load(open(scene_setting_config_path, 'r'), Loader=yaml.FullLoader)
        if max_agent_num is not None:
            scene_setting_config['simulation']['agent_spawn']['max_agent_num'] = max_agent_num
        for _ in range(simulate_num):
            scene_setting_config['simulation']['random_seed'] = random.randint(0, 1000)
            random.seed(scene_setting_config['simulation']['random_seed'])
            # 地图配置文件路径
            data_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view'
            save_folder_name = f'{start_time}/{gen_idx}'
            scenes_folder = f'/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/{save_folder_name}'
            gen_idx += 1
            os.makedirs(scenes_folder, exist_ok=True)
            video_path = os.path.join(scenes_folder, 'video')
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            with open(os.path.join(scenes_folder, 'config.yaml'), 'w') as f:
                yaml.dump(scene_setting_config, f)
            
            for scene_name in os.listdir(data_path):
                if not scene_name.isdigit():
                    continue
                if scene_name in exclude_scenes:
                    continue
                # if scene_name not in ['025']:
                #     continue
                if len(prefer_scenes) > 0 and scene_name not in prefer_scenes:
                    continue
                print('Simulating scene: ', scene_name)
                _data_root = os.path.join(data_path, scene_name)
                save_dir = os.path.join(scenes_folder, scene_name)
                map_config_path = os.path.join(_data_root, 'map_feature.json')
                ego_pose_path = os.path.join(_data_root, 'ego_pose.json')
                static_actor_config_path = os.path.join(_data_root, 'all_static_actor_data.json')
                scene_config = dict()
                scene_config['_data_root'] = _data_root
                scene_config['_save_dir'] = save_dir
                scene_config['_available_asset_dir'] = scene_setting_config['data']['available_asset_dir']
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
                    Scene_wo_render._map.draw_map_w_traffic_flow_sequence(Scene_wo_render._car_dict_sequence,text=f'{save_folder_name}/{scene_name}/traffic_pic',skip_frames=skip_frames)
                elif Scene_wo_render._mode == 'datagen':
                    Scene_wo_render._map.draw_map_w_traffic_flow_sequence(Scene_wo_render._car_dict_sequence,text=f'{save_folder_name}/{scene_name}/traffic_pic',skip_frames=skip_frames)
                    Scene_wo_render.save_traffic_flow()
                    if scene_name == 'video' or os.path.isfile(os.path.join(scenes_folder, scene_name)):
                        continue
                    scene_img_folder = os.path.join(scenes_folder, scene_name,'traffic_pic')
                    output_path = os.path.join(video_path,f'{scene_name}.mp4')
                    valid = create_video_from_images(scene_img_folder, output_path, int(scene_setting_config['simulation']['FPS']/skip_frames))
                Scene_wo_render.reset()
                


if __name__ == '__main__':
    main()