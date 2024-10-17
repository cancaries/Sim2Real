from scene_wo_render import Scene_wo_render
import random
if __name__ == '__main__':
    random_seed = 303
    # 设置随机数种子
    import random
    random.seed(random_seed)
    # 地图配置文件路径
    map_config_path = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/test_map.json'
    _data_root = '/home/ubuntu/junhaoge/ChatSim/data/waymo_multi_view/segment-17761959194352517553_5448_420_5468_420_with_camera_labels'
    
    # 创建scene对象
    scene_config = dict()
    scene_config['_map_config_path'] = map_config_path
    scene_config['_data_root'] = _data_root
    scene_config['_scene_name'] = _data_root.split('/')[-1]
    scene_config['_FPS'] = 10
    Scene_wo_render.initialize_scene(scene_config)
    Scene_wo_render.generate_background_agents()
    print(Scene_wo_render._agent_dict.keys())   
    Scene_wo_render.set_debug_mode(True)
    # 让背景车辆按照agent行为以帧为单位进行行动,agent.run_step输出的是steer,throttle,brake这些控制信号，根据这些控制信号以0.2s为间隔进行计算
    # 请注意，这里的agent是一个字典，key为agent的名字，value为agent对象
    scene_length = 250 # 250帧
    scene_time = scene_length * 0.2 # scene_time = (scene_length * 0.2）s
    for i in range(scene_length):
        for agent_name, agent in Scene_wo_render._agent_dict.items():
            agent_control = agent.run_step()
            Scene_wo_render._agent_control_dict[agent_name] = agent_control
        Scene_wo_render.run_step()
    # 生成背景车辆的轨迹图
    if Scene_wo_render._debug_mode:
        Scene_wo_render._map.draw_map_w_traffic_flow_sequence(Scene_wo_render._car_dict_sequence,text=f'background_agents_{Scene_wo_render._start_time}')

            