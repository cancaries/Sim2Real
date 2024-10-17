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
    random.seed(random_seed)
    # 地图配置文件路径
    # map_config_path = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/test_map_lanechange.json'
    # _data_root = '/home/ubuntu/junhaoge/ChatSim/data/waymo_multi_view/segment-17761959194352517553_5448_420_5468_420_with_camera_labels'
    map_config_path = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/test_map_datatest.json'
    _data_root = '/home/ubuntu/junhaoge/ChatSim/data/waymo_multi_view/segment-11379226583756500423_6230_810_6250_810_with_camera_labels'
    # 创建scene对象
    scene_config = dict()
    scene_config['_map_config_path'] = map_config_path
    scene_config['_data_root'] = _data_root
    scene_config['_scene_name'] = _data_root.split('/')[-1]
    scene_config['_FPS'] = 10
    Scene_wo_render.initialize_scene(scene_config)
    ego_pose_path = os.path.join(_data_root, 'ego_pose.json')
    ego_pose_data = json.load(open(ego_pose_path, 'r'))
    Scene_wo_render.initialize_scene(scene_config)
    transform_path = build_transform_path_from_ego_pose_data(ego_pose_data)
    Scene_wo_render._map.refine_spawn_points(ego_init_point=transform_path[0].location)
    Scene_wo_render.initialize_ego_vehicle_w_transform_path(transform_path)
    # Scene_wo_render.generate_background_agents(spawn_num=50)
    Scene_wo_render.generate_background_agents(spawn_num=50)

    Scene_wo_render.set_debug_mode(True)
    scene_length = 400
    scene_time = scene_length / 10
    stay_time = 5
    stay_time_steps = stay_time * 10 
    stay_time_steps = len(transform_path) - 1
    overtake_path_loc_list = []
    overtake_path = None
    # for agent_name, agent in Scene_wo_render._agent_dict.items():
    #     if 'ego' in agent_name:
    #         continue
    #     agent._force_overtake()
    #     overtake_path = agent.plan_path
    #     Scene_wo_render._map.draw_map_w_waypoints(overtake_path,file_path=f'/home/ubuntu/junhaoge/ChatSim/data/draw_pic/overtake_path_{agent_name}.png')
    # for waypoint in overtake_path:
    #     overtake_path_loc_list.append(waypoint.transform.location)
    # with open('/home/ubuntu/junhaoge/ChatSim/data/draw_pic/overtake_path.json', 'w') as f:
    #     json.dump(overtake_path_loc_list, f, indent=2)
    #     Scene_wo_render._map.draw_map_w_waypoints(agent.plan_path)
    #     agent._force_overtake()
    #     overtake_path = agent.plan_path
    #     Scene_wo_render._map.draw_map_w_waypoints(overtake_path)
    for i in range(scene_length):
        for agent_name, agent in Scene_wo_render._agent_dict.items():
            agent_control = agent.run_step()
            Scene_wo_render._agent_control_dict[agent_name] = agent_control
        if i < stay_time_steps:
            Scene_wo_render.run_step_w_ego(i)
        else:
            Scene_wo_render.run_step_w_ego(i, True)
    
    for agent_name, agent in Scene_wo_render._agent_dict.items():
        overtake_path = agent.plan_path
        Scene_wo_render._map.draw_map_w_waypoints(overtake_path,file_path=f'/home/ubuntu/junhaoge/ChatSim/data/draw_pic/overtake_path_{agent_name}.png')
    # 生成背景车辆的轨迹图
    if Scene_wo_render._debug_mode:
        Scene_wo_render._map.draw_map_w_traffic_flow_sequence(Scene_wo_render._car_dict_sequence,text=f'background_agents_{Scene_wo_render._start_time}')
    

    

            