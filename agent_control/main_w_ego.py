from scene_wo_render import Scene_wo_render
import random
from agents.navigation.waypoint import Transform
import json
import os
def build_transform_path_from_ego_pose_data(ego_pose_data):
    transform_path = []
    for ego_pose in ego_pose_data:
        transform = Transform(ego_pose)
        transform_path.append(transform)
    return transform_path

if __name__ == '__main__':
    random_seed = 303
    # 设置随机数种子
    import random
    random.seed(random_seed)
    # 地图配置文件路径
    # map_config_path = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/test_map_lanechange.json'
    # _data_root = '/home/ubuntu/junhaoge/ChatSim/data/waymo_multi_view/segment-17761959194352517553_5448_420_5468_420_with_camera_labels'.
    # map_config_path = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/test_map_datatest.json'
    _data_root = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/segment-10588771936253546636_2300_000_2320_000_with_camera_labels'
    map_config_path = os.path.join(_data_root, 'map_feature.json')
    static_actor_config_path = os.path.join(_data_root, 'all_static_actor_data.json')
    scene_config = dict()
    scene_config['_map_config_path'] = map_config_path
    scene_config['_static_actor_config_path'] = static_actor_config_path
    scene_config['_data_root'] = _data_root
    scene_config['_scene_name'] = _data_root.split('/')[-1]
    scene_config['_FPS'] = 10
    ego_pose_path = os.path.join(_data_root, 'ego_pose.json')
    ego_pose_data = json.load(open(ego_pose_path, 'r'))
    Scene_wo_render.initialize_scene(scene_config)
    transform_path = build_transform_path_from_ego_pose_data(ego_pose_data)
    Scene_wo_render.initialize_ego_vehicle_w_transform_path(transform_path)
    Scene_wo_render._map.refine_spawn_points(ego_init_point=transform_path[0].location)
    # Scene_wo_render.generate_background_agents(random_agents=True) 
    # Scene_wo_render.generate_background_agents_for_test(random_agents=False, spawn_num=1, ego_forward=True, forward_threshold=30) 
    # Scene_wo_render.generate_background_agents(random_agents=True) 

    Scene_wo_render.generate_background_agents(spawn_num=10) 
    # Scene_wo_render.generate_background_agents(random_agents=True,ego_forward_clear=True) 
    Scene_wo_render.set_debug_mode(True)
    # 让背景车辆按照agent行为以帧为单位进行行动,agent.run_step输出的是steer,throttle,brake这些控制信号，根据这些控制信号以0.2s为间隔进行计算
    # 请注意，这里的agent是一个字典，key为agent的名字，value为agent对象
    scene_length = 200 # 250帧
    scene_time = scene_length / 10 # scene_time = (scene_length * 0.2）s
    stay_time = 0
    stay_time_steps = stay_time * 10 
    # stay_time_steps = len(transform_path) - 1
    for i in range(scene_length):
        for agent_name, agent in Scene_wo_render._agent_dict.items():
            agent_control = agent.run_step()
            Scene_wo_render._agent_control_dict[agent_name] = agent_control
        if i < stay_time_steps:
            Scene_wo_render.run_step_w_ego(i)
        else:
            Scene_wo_render.run_step_w_ego(i, True)
    # 生成背景车辆的轨迹图
    if Scene_wo_render._debug_mode:
        Scene_wo_render._map.draw_map_w_traffic_flow_sequence(Scene_wo_render._car_dict_sequence,text=f'{Scene_wo_render._start_time}',skip_frames=1)
