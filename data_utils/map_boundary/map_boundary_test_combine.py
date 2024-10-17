import numpy as np
from shapely.geometry import LineString
import matplotlib.pyplot as plt
# 存储所有的lines
lines = []

# 假设 tolerance_z 是允许的 z 轴变化幅度
tolerance_z = 1.0
tolerance_xy = 20.0  # xy 平面允许的最大距离

# 计算两点的欧氏距离，并且判断z轴差异是否在容忍范围内
def distance_with_tolerance(p1, p2, tolerance_z, tolerance_xy):
    xy_dist = np.linalg.norm(p1[:2] - p2[:2])  # 只计算 x, y 轴的距离
    z_diff = abs(p1[2] - p2[2])  # 计算 z 轴的高度差
    if z_diff <= tolerance_z and xy_dist <= tolerance_xy:
        return xy_dist
    else:
        return float('inf')  # 如果z或xy差异过大，返回无穷大

# 合并两条线，头尾连接或翻转后连接
def merge_lines(line1, line2, reverse=False):
    if reverse:
        line2 = line2[::-1]  # 翻转line2
    return np.concatenate((line1, line2), axis=0)  # 合并两条线

# 找到最近的曲线
def find_nearest_line(end_point, remaining_lines, tolerance_z, tolerance_xy):
    min_distance = float('inf')
    nearest_line = None
    nearest_index = -1
    reverse = False  # 用来记录是否需要翻转

    # 遍历剩余的lines，找到距离end_point最近的line
    for i, line in enumerate(remaining_lines):
        start_point = line[0]  # line的起点
        end_point_of_line = line[-1]  # line的终点

        # 比较终点和下一个line的起点
        dist_to_start = distance_with_tolerance(end_point, start_point, tolerance_z, tolerance_xy)
        dist_to_end = distance_with_tolerance(end_point, end_point_of_line, tolerance_z, tolerance_xy)

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

# 第一步：初步合并基本相近的曲线
def initial_merge(lines, tolerance_z, tolerance_xy):
    merged_lines = [lines.pop(0)]  # 将第一条曲线作为已合并曲线的初始值

    while lines:
        # 取出当前已合并线段的最后一个终点
        last_line = merged_lines[-1]
        last_point = last_line[-1]

        # 找到最接近的line
        nearest_line, nearest_index, reverse = find_nearest_line(last_point, lines, tolerance_z, tolerance_xy)

        if nearest_line is not None and nearest_index != -1:
            # 合并最近的line
            new_line = merge_lines(last_line, nearest_line, reverse)
            merged_lines[-1] = new_line  # 更新已合并的最后一条线段
            lines.pop(nearest_index)  # 移除已合并的线段
        else:
            # 如果没有合适的曲线，则开始新的一条线段
            merged_lines.append(lines.pop(0))

    return merged_lines

# 第二步：递推式合并非闭合的曲线
def iterative_merge_non_closed(merged_lines, tolerance_z, tolerance_xy):
    updated = True

    while updated:  # 递推式地合并非闭合曲线，直到无法再合并
        updated = False
        for i, line1 in enumerate(merged_lines):
            if i >= len(merged_lines) - 1:
                continue  # 最后一条线不用比

            # 找出当前线的尾部
            end_point = line1[-1]

            # 尝试寻找其他非闭合曲线的起点或终点
            for j in range(i + 1, len(merged_lines)):
                line2 = merged_lines[j]
                start_point = line2[0]
                end_point_of_line2 = line2[-1]

                # 判断是否能连接
                dist_to_start = distance_with_tolerance(end_point, start_point, tolerance_z, tolerance_xy)
                dist_to_end = distance_with_tolerance(end_point, end_point_of_line2, tolerance_z, tolerance_xy)

                if dist_to_start != float('inf') or dist_to_end != float('inf'):
                    # 合并最近的线段
                    reverse = dist_to_end < dist_to_start  # 选择合适的方向
                    new_line = merge_lines(line1, line2, reverse)
                    merged_lines[i] = new_line
                    merged_lines.pop(j)  # 移除已合并的线段
                    updated = True
                    break

    return merged_lines

map_feature_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/003/map_feature.json'
import json
with open(map_feature_path, 'r') as f:
    map_feature = json.load(f)

# 示例数据填充
for lane_id, lane_info in map_feature.items():
    if 'road_edge_type' not in lane_info:
        continue
    if lane_info['feature_type'] == 'road_edge' and lane_info['road_edge_type'] == 1:
        lines.append(np.array(lane_info['polyline']))

# 第一步：初步合并基本可以接上的曲线
initial_merged_lines = initial_merge(lines, tolerance_z, tolerance_xy)

# 第二步：递推式合并非闭合的曲线
final_merged_lines = iterative_merge_non_closed(initial_merged_lines, tolerance_z, tolerance_xy)

# 打印或绘制最终合并的结果
for line in final_merged_lines:
    plt.plot(line[:, 0], line[:, 1], markersize='1', marker='o')

save_path = '/home/ubuntu/junhaoge/real_world_simulation/data_utils/map_boundary/map_boundary_test.png'
plt.savefig(save_path)
