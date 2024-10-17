from scene_wo_render import Scene_wo_render
import random
from agents.navigation.waypoint import Transform
import yaml
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_setting_config', type=str, default='/home/ubuntu/junhaoge/real_world_simulation/agent_control/agent_configs/default.yaml')
    args = parser.parse_args()
    scene_setting_config = yaml.load(open(args.scene_setting_config, 'r'), Loader=yaml.FullLoader)
    random.seed(scene_setting_config['simulation']['random_seed'])
    # 地图配置文件路径
    _data_root = scene_setting_config['data']['data_root']
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
    for i in range(scene_length):
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
    if Scene_wo_render._debug_mode:
        Scene_wo_render._map.draw_map_w_traffic_flow_sequence(Scene_wo_render._car_dict_sequence,text=f'{Scene_wo_render._start_time}',skip_frames=1)

if __name__ == '__main__':
    main()