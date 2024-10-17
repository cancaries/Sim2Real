import os
import json
import argparse


# save car_dict_sequence_{}.json
# ego_pose_ori.json/map_feature.json/all_static_actor_data.json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sg_data_path", default="/home/ubuntu/DATA3/waymo_train_already/", type=str)
    parser.add_argument("--agent_data_path", default="/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/", type=str)
    parser.add_argument("--waymo_multi_view_path", default="/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view", type=str)
    parser.add_argument("--specific_scene", default=['036','019','046','053',',058','096','382','402','427'], type=list)
    parser.add_argument("--all_scene", default=False, type=bool)
    args = parser.parse_args()
    sg_data_path = args.sg_data_path
    agent_data_path = args.agent_data_path
    waymo_multi_view_path = args.waymo_multi_view_path
    specific_scene = args.specific_scene
    all_scene = args.all_scene
    scene_exp_number_dict = dict()
    for exp_time in os.listdir(agent_data_path):
        if not os.path.isdir(os.path.join(agent_data_path, exp_time)):
            continue
        if 'test' in exp_time:
            continue
        exp = os.listdir(os.path.join(agent_data_path, exp_time))
        exp.sort()
        for exp_idx in exp:
            if not os.path.isdir(os.path.join(agent_data_path, exp_time, exp_idx)):
                continue
            for scene_name in os.listdir(os.path.join(agent_data_path, exp_time, exp_idx)):
                if not scene_name.isdigit():
                    continue
                if not os.path.isdir(os.path.join(agent_data_path, exp_time, exp_idx, scene_name)):
                    continue
                if scene_name not in scene_exp_number_dict:
                    scene_exp_number_dict[scene_name] = 0
                if all_scene or scene_name in specific_scene:
                    scene_path = os.path.join(agent_data_path, exp_time, exp_idx, scene_name)
                    scene_data_path = os.path.join(sg_data_path, scene_name)
                    os.makedirs(scene_data_path, exist_ok=True)
                    car_info_path = os.path.join(sg_data_path, scene_name, 'car_info')
                    os.makedirs(car_info_path, exist_ok=True)
                    map_feature_path = os.path.join(car_info_path, 'map_feature.json')
                    ego_pose_ori_path = os.path.join(car_info_path, 'ego_pose_ori.json')
                    # car_dict_sequence_path = os.path.join(car_info_path, f'car_dict_sequence_{scene_exp_number_dict[scene_name]}.json')
                    car_dict_sequence_path = os.path.join(car_info_path, f'car_dict_sequence_{exp_idx}.json')
                    static_actor_path = os.path.join(car_info_path, 'all_static_actor_data.json')
                    # if not os.path.exists(car_dict_sequence_path):
                    os.system(f'cp {scene_path}/car_info_dict.json {car_dict_sequence_path}')
                    # if not os.path.exists(map_feature_path):
                    os.system(f'cp {scene_path}/map_feature.json {map_feature_path}')
                    # if not os.path.exists(ego_pose_ori_path):
                    os.system(f'cp {waymo_multi_view_path}/{scene_name}/ego_pose.json {ego_pose_ori_path}')
                    # if not os.path.exists(static_actor_path):
                    os.system(f'cp {waymo_multi_view_path}/{scene_name}/all_static_actor_data.json {static_actor_path}')
                    scene_exp_number_dict[scene_name] = scene_exp_number_dict[scene_name] + 1
