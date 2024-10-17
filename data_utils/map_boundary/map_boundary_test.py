import numpy as np
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon, MultiPolygon

# 假设 tolerance_z 是允许的 z 轴变化幅度
tolerance_z = 1.0

# 计算两点的欧氏距离，并且判断z轴差异是否在容忍范围内
def distance_with_tolerance(p1, p2, tolerance_z):
    xy_dist = np.linalg.norm(p1[:2] - p2[:2])  # 只计算 x, y 轴的距离
    z_diff = abs(p1[2] - p2[2])  # 计算 z 轴的高度差
    return xy_dist if z_diff <= tolerance_z else float('inf')  # 如果z差异过大，返回无穷大

# 找到最近的未连接的 line
def find_nearest_line(end_point, remaining_lines, tolerance_z):
    min_distance = float('inf')
    nearest_line = None
    nearest_index = -1
    reverse = False  # 用来记录是否需要翻转方向

    # 遍历剩余的lines，找到距离end_point最近的line
    for i, line in enumerate(remaining_lines):
        start_point = line[0]  # line的起点
        end_point_of_line = line[-1]  # line的终点

        # 比较终点和下一个line的起点
        dist_to_start = distance_with_tolerance(end_point, start_point, tolerance_z)
        dist_to_end = distance_with_tolerance(end_point, end_point_of_line, tolerance_z)

        # 选择距离最近的点
        if dist_to_start < min_distance:
            min_distance = dist_to_start
            nearest_line = line
            nearest_index = i
            reverse = False  # 不需要翻转
        if dist_to_end < min_distance:
            min_distance = dist_to_end
            nearest_line = line
            nearest_index = i
            reverse = True  # 需要翻转

    return nearest_line, nearest_index, reverse

map_feature_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/003/map_feature.json'
import json
with open(map_feature_path, 'r') as f:
    map_feature = json.load(f)

lines = []
# 遍历所有的 road_edge 并构建 polyline
for lane_id, lane_info in map_feature.items():
    if 'road_edge_type' not in lane_info:
        continue
    if lane_info['feature_type'] == 'road_edge' and lane_info['road_edge_type'] == 1:
        # 只取 x, y, z 的坐标
        lines.append(np.array(lane_info['polyline']))

# 起始连接从第一条line开始
connected_lines = [lines.pop(0)]  # 将第一个road_edge加入已连接的lines

# 不断寻找并连接最接近的line
while lines:
    # 当前已连接line的最后一个终点
    last_line = connected_lines[-1]
    last_point = last_line[-1]  # 取出当前连接段的终点

    # 找到最接近的line
    nearest_line, nearest_index, reverse = find_nearest_line(last_point, lines, tolerance_z)

    if nearest_line is not None:
        # 如果找到最近的line，判断是否需要翻转
        if reverse:
            nearest_line = nearest_line[::-1]  # 翻转line方向

        # 将最近的line加入已连接的lines
        connected_lines.append(nearest_line)

        # 将已连接的line从剩余的lines中移除
        lines.pop(nearest_index)
        print('nearest_line is not None')
    else:
        print("无法找到接近的线段，结束连接")
        break

# 输出最终连接的lines（每个连接的线段是 x, y, z 格式的数组）
final_lines = np.concatenate(connected_lines)

# 将所有点合并到一个点集
all_points = np.array([point for line in lines for point in final_lines])

# # 使用 alphashape 构造凹多边形
# alpha = 0.01  # 控制多边形的凹度，值越小越凹
# alpha_shape = alphashape.alphashape(all_points, alpha)

# 绘制所有点
# plt.scatter(all_points[:, 0], all_points[:, 1], s=1)
plt.plot(all_points[:, 0], all_points[:, 1],markersize='1',marker='o')
# 检查 alpha_shape 的类型
# print(f"alpha_shape type: {type(alpha_shape)}")
# print(alpha_shape)
# 处理 alpha_shape，判断它是 Polygon 还是 MultiPolygon
# if isinstance(alpha_shape, Polygon):
#     # 单个 Polygon，绘制边界
#     if not alpha_shape.exterior.is_closed:
#         coords = list(alpha_shape.exterior.coords)
#         coords.append(coords[0])  # 闭合多边形
#         alpha_shape = Polygon(coords)
    
#     x, y = alpha_shape.exterior.xy
#     plt.fill(x, y, alpha=0.5, fc='lightgray', ec='blue')

# elif isinstance(alpha_shape, MultiPolygon):
#     # 遍历 MultiPolygon 中的每个 Polygon 并绘制
#     for poly in alpha_shape.geoms:

#         if not poly.exterior.is_closed:
#             coords = list(poly.exterior.coords)
#             coords.append(coords[0])  # 闭合多边形
#             poly = Polygon(coords)
        
#         x, y = poly.exterior.xy
#         plt.fill(x, y, alpha=0.5, fc='lightgray', ec='blue')

# else:
#     print("Alpha shape is neither Polygon nor MultiPolygon.")

# 找到面积最大的 Polygon
# if isinstance(alpha_shape, Polygon):
#     largest_polygon = alpha_shape  # 只有一个 Polygon
# elif isinstance(alpha_shape, MultiPolygon):
#     # 遍历所有的 Polygon，找到面积最大的
#     largest_polygon = max(alpha_shape.geoms, key=lambda poly: poly.area)
# else:
#     raise ValueError("Alpha shape is neither Polygon nor MultiPolygon.")

# # 确保多边形是闭合的
# if not largest_polygon.exterior.is_closed:
#     coords = list(largest_polygon.exterior.coords)
#     coords.append(coords[0])
#     largest_polygon = Polygon(coords)

# # 绘制面积最大的 Polygon
# x, y = largest_polygon.exterior.xy
# plt.fill(x, y, alpha=0.5, fc='lightgray', ec='blue')
# # 设置坐标轴范围
# plt.xlim(-200, 200)
# plt.ylim(-200, 200)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.grid(True)
save_path = '/home/ubuntu/junhaoge/real_world_simulation/data_utils/map_boundary/map_boundary_test.png'
plt.savefig(save_path)
