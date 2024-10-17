from typing import List
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
import tensorflow as tf
import json
import os
def rotate(point, angle):
    """Rotates a point around the origin by the specified angle in radians."""
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, point)

def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car['cx']
    y = car['cy']
    z = car['cz']
    length = car['length']
    width = car['width']
    height = car['height']
    heading = car['heading']
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])

    # The relative positions of the vertices from the box center before rotation.
    relative_positions = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ]) * half_dims

    # Rotate each relative position and add the box center position.
    vertices = np.asarray([rotate(pos, heading) + box_center for pos in relative_positions])
    return vertices

# FILENAME = '/home/ubuntu/junhaoge/ChatSim/data/waymo_tfrecords/1.4.2/segment-13196796799137805454_3036_940_3056_940_with_camera_labels.tfrecord'
# FILENAME = '/home/ubuntu/junhaoge/ChatSim/data/waymo_tfrecords/1.4.2/segment-17761959194352517553_5448_420_5468_420_with_camera_labels.tfrecord'
# 11379226583756500423_6230_810_6250_810_with_camera_labels
FILENAME = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_tfrecords/1.4.2/segment-10588771936253546636_2300_000_2320_000_with_camera_labels.tfrecord'
seg_name = FILENAME.split('/')[-1].split('.')[0]
save_path = f'/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/{seg_name}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

frames = []
for data in dataset:
    frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
    frames.append(frame)
plt.figure(figsize=(20, 20),dpi=100)
# 隐藏坐标轴，不显示坐标轴的值，并减少页边距
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# 增大字号
plt.rcParams.update({'font.size': 22})
# 加粗字体
plt.rcParams.update({'font.weight': 'bold'})
transform = np.reshape(np.array(frames[5].pose.transform), [4, 4])
transform = np.linalg.inv(transform)
road_edges = []
lanes = []
driveway = []
road_line = []
crosswalk = []
map_feature_dict = {}
# lane and road_edge
for i in range(len(frames)):
    print(frames[i].map_features)
for i in range(len(frames[0].map_features)):
    feature_id = frames[0].map_features[i].id
    map_feature_dict[feature_id] = {}
    if len(frames[0].map_features[i].lane.polyline) > 0:
        if frames[0].map_features[i].lane.type:
            map_feature_dict[feature_id]['lane_type'] = frames[0].map_features[i].lane.type
        else:
            map_feature_dict[feature_id]['lane_type'] = None
        map_feature_dict[feature_id]['feature_type'] = 'lane'
        curr_lane = []
        for node in frames[0].map_features[i].lane.polyline:
            node_position = np.ones(4)
            node_position[0] = node.x
            node_position[1] = node.y
            node_position[2] = node.z
            curr_lane.append(node_position)
        curr_lane = np.stack(curr_lane)
        curr_lane = np.transpose(np.matmul(transform, np.transpose(curr_lane)))[:, 0:3]
        map_feature_dict[feature_id]['polyline'] = curr_lane.tolist()
        lanes.append(curr_lane)
        if frames[0].map_features[i].lane.interpolating:
            map_feature_dict[feature_id]['interpolating'] = frames[0].map_features[i].lane.interpolating
        else:
            map_feature_dict[feature_id]['interpolating'] = None
        if frames[0].map_features[i].lane.speed_limit_mph:
            map_feature_dict[feature_id]['speed_limit_mph'] = frames[0].map_features[i].lane.speed_limit_mph
            # print(type(frames[0].map_features[i].lane.type))
        if frames[0].map_features[i].lane.entry_lanes:
            entry_lanes = []
            for entry_lane in frames[0].map_features[i].lane.entry_lanes:
                entry_lanes.append(str(entry_lane))
            map_feature_dict[feature_id]['entry_lanes'] = entry_lanes
        if frames[0].map_features[i].lane.exit_lanes:
            exit_lanes = []
            for exit_lane in frames[0].map_features[i].lane.exit_lanes:
                exit_lanes.append(str(exit_lane))
            map_feature_dict[feature_id]['exit_lanes'] = exit_lanes
        if frames[0].map_features[i].lane.left_neighbors:
            map_feature_dict[feature_id]['left_neighbors'] = []
            for left_neighbor in frames[0].map_features[i].lane.left_neighbors:
                left_neighbor_dict = {'feature_id': str(left_neighbor.feature_id), 'self_start_index': left_neighbor.self_start_index, \
                                 'self_end_index': left_neighbor.self_end_index, 'neighbor_start_index': left_neighbor.neighbor_start_index, \
                                    'neighbor_end_index': left_neighbor.neighbor_end_index}
                map_feature_dict[feature_id]['left_neighbors'].append(left_neighbor_dict)
        if frames[0].map_features[i].lane.right_neighbors:
            map_feature_dict[feature_id]['right_neighbors'] = []
            for right_neighbor in frames[0].map_features[i].lane.right_neighbors:
                right_neighbor_dict = {'feature_id': str(right_neighbor.feature_id), 'self_start_index': right_neighbor.self_start_index, \
                                 'self_end_index': right_neighbor.self_end_index, 'neighbor_start_index': right_neighbor.neighbor_start_index, \
                                    'neighbor_end_index': right_neighbor.neighbor_end_index}
                map_feature_dict[feature_id]['right_neighbors'].append(right_neighbor_dict)
    
    if len(frames[0].map_features[i].road_edge.polyline) > 0:
        if frames[0].map_features[i].road_edge.type:
            map_feature_dict[feature_id]['road_edge_type'] = frames[0].map_features[i].road_edge.type
        else:
            map_feature_dict[feature_id]['road_edge_type'] = None
        map_feature_dict[feature_id]['feature_type'] = 'road_edge'
        curr_edge = []
        for node in frames[0].map_features[i].road_edge.polyline:
            node_position = np.ones(4)
            node_position[0] = node.x
            node_position[1] = node.y
            node_position[2] = node.z
            curr_edge.append(node_position)
        curr_edge = np.stack(curr_edge)
        curr_edge = np.transpose(np.matmul(transform, np.transpose(curr_edge)))[:, 0:3]
        map_feature_dict[feature_id]['polyline'] = curr_edge.tolist()
        road_edges.append(curr_edge)
    
    # if len(frames[0].map_features[i].crosswalk.polygon) > 0:
    #     map_feature_dict[feature_id]['feature_type'] = 'crosswalk'
    #     curr_polygon = []
    #     for node in frames[0].map_features[i].crosswalk.polygon:
    #         node_position = np.ones(4)
    #         node_position[0] = node.x
    #         node_position[1] = node.y
    #         node_position[2] = node.z
    #         curr_polygon.append(node_position)
    #     curr_polygon = np.stack(curr_polygon)
    #     curr_polygon = np.transpose(np.matmul(transform, np.transpose(curr_polygon)))[:, 0:3]
    #     crosswalk.append(curr_polygon)
    #     map_feature_dict[feature_id]['polyline'] = curr_polygon.tolist()

    
    # if len(frames[0].map_features[i].driveway.polygon) > 0:
    #     map_feature_dict[feature_id]['feature_type'] = 'driveway'
    #     curr_polygon = []
    #     for node in frames[0].map_features[i].driveway.polygon:
    #         node_position = np.ones(4)
    #         node_position[0] = node.x
    #         node_position[1] = node.y
    #         node_position[2] = node.z
    #         curr_polygon.append(node_position)
    #     curr_polygon = np.stack(curr_polygon)
    #     curr_polygon = np.transpose(np.matmul(transform, np.transpose(curr_polygon)))[:, 0:3]
    #     driveway.append(curr_polygon)
    #     map_feature_dict[feature_id]['polyline'] = curr_polygon.tolist()

    # if len(frames[0].map_features[i].road_line.polyline) > 0:
    #     map_feature_dict[feature_id]['feature_type'] = 'road_line'
    #     curr_polygon = []
    #     for node in frames[0].map_features[i].road_line.polyline:
    #         node_position = np.ones(4)
    #         node_position[0] = node.x
    #         node_position[1] = node.y
    #         node_position[2] = node.z
    #         curr_polygon.append(node_position)
    #     curr_polygon = np.stack(curr_polygon)
    #     curr_polygon = np.transpose(np.matmul(transform, np.transpose(curr_polygon)))[:, 0:3]
    #     road_line.append(curr_polygon)
    #     map_feature_dict[feature_id]['polyline'] = curr_polygon.tolist()

map_json_save_path = os.path.join(save_path, 'map_feature.json')
with open(map_json_save_path, 'w') as f:
    json.dump(map_feature_dict, f, indent=2)

x_min = -300
x_max = 500
y_min = -200
y_max = 200
cropped_road_edges = []
for edge in road_edges:
    new_road_edge = []
    for i in range(edge.shape[0]):
        if edge[i,0] < x_min or edge[i,0] > x_max or edge[i,1] < y_min or edge[i,1] > y_max:
            continue
        new_road_edge.append(edge[i])
    if len(new_road_edge) > 0:
        new_road_edge = np.stack(new_road_edge)
        cropped_road_edges.append(new_road_edge)

cropped_lanes = []
for lane in lanes:
    new_lane = []
    for i in range(lane.shape[0]):
        if lane[i,0] < x_min or lane[i,0] > x_max or lane[i,1] < y_min or lane[i,1] > y_max:
            continue
        new_lane.append(lane[i])
    if len(new_lane) > 0:
        new_lane = np.stack(new_lane)
        cropped_lanes.append(new_lane)

max_per_lane_node = 20

for edge in cropped_road_edges:
    edge = np.array(edge)
    # edge = edge[::5]
    plt.plot(edge[:,0],edge[:,1],c='red')

for lane in cropped_lanes:
    lane = np.array(lane)
    # lane = lane[::5]
    plt.plot(lane[:,0],lane[:,1],c='green')

for driveway_edge in driveway:
    driveway_edge = np.array(driveway_edge)
    plt.plot(driveway_edge[:,0], driveway_edge[:,1], c='blue')

for road_line in road_line:
    edge = np.array(road_line)
    plt.plot(edge[:,0],edge[:,1],c='yellow', linewidth=1)

for crosswalk_edge in crosswalk:
    crosswalk_edge = np.array(crosswalk_edge)
    plt.plot(crosswalk_edge[:,0], crosswalk_edge[:,1], c='black')

# 在整个fig的绝对位置的左下角处增加各个道路类型的图例，要求是一行文字，旁边是一个对应的色块
y_offset = -0.15
plt.text(0.1, 0.38+y_offset, 'Road Edge', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='red')
plt.text(0.1, 0.34+y_offset, 'Lane', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='green')
plt.text(0.1, 0.30+y_offset, 'Driveway', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='blue')
plt.text(0.1, 0.26+y_offset, 'Road Line', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='yellow')
plt.text(0.1, 0.22+y_offset, 'Crosswalk', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='black')




plt_save_path = '/home/ubuntu/junhaoge/real_world_simulation/data/end2end_map_data/test_map.png'
plt.savefig(plt_save_path)


# cars = np.load('/dssg/home/acct-giftxhy/giftxhy/yuxiwei/nerf-factory/data/waymo/segment-10247954040621004675_2180_000_2200_000_with_camera_labels/3d_boxes.npy', allow_pickle = True).item()

# vertices = generate_vertices(cars['0'])

# plt.plot(vertices[::2,0],vertices[::2,1])

# plt.scatter([29.623],[-4.65])
# plt.savefig('/home/ubuntu/yuxiwei/debug/map.png')
# import ipdb; ipdb.set_trace()
# output = {"centerline":cropped_lanes,"boundary":cropped_road_edges}
# import pickle 
# with open('/dssg/home/acct-giftxhy/giftxhy/waymo_tfrecord/1.4.2/map_data.pkl', 'wb') as f:
#     pickle.dump(output, f)

# import ipdb; ipdb.set_trace()

    
# def vis_map_debug(map, motion, save_path=None):
#                 from matplotlib import pyplot as plt
#                 cropped_road_edges = map['boundary']
#                 cropped_lanes = map['centerline']
#                 for edge in cropped_road_edges:
#                     edge = np.array(edge)
#                     # edge = edge[::5]
#                     plt.plot(edge[:,0],edge[:,1],c='red')

#                 for lane in cropped_lanes:
#                     lane = np.array(lane)
#                     # lane = lane[::5]
#                     plt.plot(lane[:,0],lane[:,1],c='green')

#                 if motion is not None:
                        
#                         # lane = lane[::5]
#                     plt.plot(motion[:,0],motion[:,1],c='blue')
#                 if save_path:
#                     plt.savefig(save_path)
#                 else:
#                     plt.savefig('/home/ubuntu/junhaoge/ChatSim/data/test_map')