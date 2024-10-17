import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../..'))
import json
import numpy as np
import matplotlib.pyplot as plt

root_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/'
for path in os.listdir(root_path):
    road_edges = []
    crosswalk = []
    road_line = []

    scene_path = os.path.join(root_path,path)
    if not os.path.isdir(scene_path):
        continue
    map_feature_path = os.path.join(scene_path,'map_feature.json')
    
    ego_pose_path = os.path.join(scene_path,'ego_pose.json')
    scene_name = map_feature_path.split('/')[-2]
    # if scene_name != '031':
    #     continue
    with open(map_feature_path, 'r') as f:
        map_feature = json.load(f)
    with open(ego_pose_path, 'r') as f:
        ego_pose = json.load(f)
    ego_z = []
    ego_xy = []
    for ego_pose_tmp in ego_pose:
        ego_xy.append(ego_pose_tmp['location'][:2])
        ego_z.append(ego_pose_tmp['location'][2])
    min_ego_z = min(ego_z) - 5.0
    max_ego_z = max(ego_z) + 5.0
    
    for lane_id, lane_info in map_feature.items():
        if 'road_edge_type' not in lane_info:
            continue
        lane_new = []
        if lane_info['feature_type'] == 'road_edge':
            for point in lane_info['polyline']:
                if point[2] < min_ego_z or point[2] > max_ego_z:
                    continue
                lane_new.append(point)
            if len(lane_new) > 0:
                road_edges.append(np.array(lane_new))

    for lane_id, lane_info in map_feature.items():
        lane_new = []
        if not 'feature_type' in lane_info.keys():
            continue
        if lane_info['feature_type'] == 'crosswalk':
            for point in lane_info['polyline']:
                lane_new.append(point)
            lane_new.append(lane_info['polyline'][0])
            if len(lane_new) > 0:
                crosswalk.append(np.array(lane_new))

    for lane_id, lane_info in map_feature.items():
        lane_new = []
        if not 'feature_type' in lane_info.keys():
            continue
        if lane_info['feature_type'] == 'road_line':
            for point in lane_info['polyline']:
                if point[2] < min_ego_z or point[2] > max_ego_z:
                    continue
                lane_new.append(point)
            if len(lane_new) > 0:
                road_line.append(np.array(lane_new))


    
    for i in range(len(road_edges)):
        road_edges[i] = np.array(road_edges[i])
        # plt.plot(road_edges[i][:, 0], road_edges[i][:, 1], markersize='1', marker='o')
        plt.plot(road_edges[i][:, 0], road_edges[i][:, 1], markersize='1', marker='o',color='black')
    
    for i in range(len(crosswalk)):
        crosswalk[i] = np.array(crosswalk[i])
        # plt.plot(ped_crossing[i][:, 0], ped_crossing[i][:, 1], markersize='1', marker='o')
        plt.plot(crosswalk[i][:, 0], crosswalk[i][:, 1], markersize='1', marker='o',color='blue')

    for i in range(len(road_line)):
        road_line[i] = np.array(road_line[i])
        # plt.plot(road_line[i][:, 0], road_line[i][:, 1], markersize='1', marker='o')
        plt.plot(road_line[i][:, 0], road_line[i][:, 1], markersize='1', marker='o',color='red')
        
    save_path = f'/home/ubuntu/junhaoge/real_world_simulation/data_utils/map_boundary/map_boundary_{scene_name}.png'
    plt.savefig(save_path)
    plt.close()