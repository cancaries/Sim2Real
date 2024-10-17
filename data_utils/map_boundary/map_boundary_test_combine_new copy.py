import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# 假设 tolerance_z 是允许的 z 轴变化幅度
tolerance_z = 1.0
tolerance_xy = 2.0  # xy 平面允许的最大距离
closure_threshold = 2.0  # 闭合判断的阈值
tolerance_z_connection = 5.0
tolerance_xy_connection = 60.0  # 连接两个线段时，xy 平面允许的最大距离


# 计算两点的欧氏距离，并且判断z轴差异是否在容忍范围内
def distance_with_tolerance(p1, p2, tolerance_z, tolerance_xy):
    xy_dist = np.linalg.norm(p1[:2] - p2[:2])  # 只计算 x, y 轴的距离
    z_diff = abs(p1[2] - p2[2])  # 计算 z 轴的高度差
    if z_diff <= tolerance_z and xy_dist <= tolerance_xy:
        return xy_dist
    else:
        return float('inf')  # 如果z或xy差异过大，返回无穷大

# 插值函数：在 p1 和 p2 之间生成插值点
def interpolate_between_points(p1, p2, num_points=20):
    t = np.linspace(0, 1, num_points)  # 生成参数 t
    interpolated_points = (1 - t)[:, None] * p1 + t[:, None] * p2  # 线性插值
    return interpolated_points

# 样条插值函数：在 p1 和 p2 之间生成平滑曲线
def spline_interpolation_between_points(p1, p2, num_points=20):
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]

    t = [0, 1]  # 参数化的 t
    t_new = np.linspace(0, 1, num_points)  # 新的 t 值，用于插值
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)

    interpolated_points = np.vstack((cs_x(t_new), cs_y(t_new), cs_z(t_new))).T
    return interpolated_points

# 合并两条线，头尾连接并进行插值
def merge_lines_with_interpolation(line1, line2, reverse_line1=False, reverse_line2=False, interpolation='spline', num_points=20):
    if reverse_line1:
        line1 = line1[::-1]  # 翻转 line1
    if reverse_line2:
        line2 = line2[::-1]  # 翻转 line2

    p1 = line1[-1]
    p2 = line2[0]

    # 插值处理
    if interpolation == 'spline':
        interp_points = spline_interpolation_between_points(p1, p2, num_points)
    else:
        interp_points = interpolate_between_points(p1, p2, num_points)

    return np.concatenate((line1, interp_points, line2), axis=0)

# 判断线段是否闭合，使用闭合阈值
def is_closed(line, closure_threshold):
    start_point = line[0]
    end_point = line[-1]
    closure_distance = np.linalg.norm(start_point[:2] - end_point[:2])  # 只考虑 x, y
    return closure_distance <= closure_threshold

# 找到最近的线段并考虑所有四种可能的连接方式，跳过已闭合线段
def find_nearest_line(connected_line, remaining_lines, tolerance_z, tolerance_xy):
    last_point_of_connected = connected_line[-1]
    first_point_of_connected = connected_line[0]

    min_distance = float('inf')
    nearest_line = None
    nearest_index = -1
    best_reverse_connected = False
    best_reverse_next = False

    for i, line in enumerate(remaining_lines):
        if is_closed(line, closure_threshold):
            continue  # 跳过已闭合线段

        start_point = line[0]
        end_point = line[-1]

        # 尝试四种连接方式
        dist1 = distance_with_tolerance(last_point_of_connected, start_point, tolerance_z, tolerance_xy)
        dist2 = distance_with_tolerance(last_point_of_connected, end_point, tolerance_z, tolerance_xy)
        dist3 = distance_with_tolerance(first_point_of_connected, start_point, tolerance_z, tolerance_xy)
        dist4 = distance_with_tolerance(first_point_of_connected, end_point, tolerance_z, tolerance_xy)

        # 选择最小的距离
        if dist1 < min_distance:
            min_distance = dist1
            nearest_line = line
            nearest_index = i
            best_reverse_connected = False
            best_reverse_next = False
        if dist2 < min_distance:
            min_distance = dist2
            nearest_line = line
            nearest_index = i
            best_reverse_connected = False
            best_reverse_next = True
        if dist3 < min_distance:
            min_distance = dist3
            nearest_line = line
            nearest_index = i
            best_reverse_connected = True
            best_reverse_next = False
        if dist4 < min_distance:
            min_distance = dist4
            nearest_line = line
            nearest_index = i
            best_reverse_connected = True
            best_reverse_next = True

    return nearest_line, nearest_index, best_reverse_connected, best_reverse_next
lines = []
map_feature_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/021/map_feature.json'
scene_name = map_feature_path.split('/')[-2]
import json
with open(map_feature_path, 'r') as f:
    map_feature = json.load(f)
    # 示例数据填充
for lane_id, lane_info in map_feature.items():
    if 'road_edge_type' not in lane_info:
        continue
    if lane_info['feature_type'] == 'road_edge' and lane_info['road_edge_type'] == 1:
        lines.append(np.array(lane_info['polyline']))

# 初始合并的线段
merged_lines = [lines.pop(0)]  # 将第一条曲线作为初始曲线

# 初步连接，包含插值并考虑四种连接方式
while lines:
    # 取出当前已合并线段的第一个和最后一个终点
    current_line = merged_lines[-1]

    # 找到最接近的line
    nearest_line, nearest_index, reverse_connected, reverse_next = find_nearest_line(current_line, lines, tolerance_z, tolerance_xy)

    if nearest_line is not None and nearest_index != -1:
        # 合并最近的line，并进行插值
        new_line = merge_lines_with_interpolation(current_line, nearest_line, reverse_line1=reverse_connected, reverse_line2=reverse_next)
        merged_lines[-1] = new_line  # 更新已合并的最后一条线段
        lines.pop(nearest_index)  # 移除已合并的线段
    else:
        # 如果没有合适的曲线，则开始新的一条线段
        merged_lines.append(lines.pop(0))

lines = merged_lines.copy()
merged_lines_new = [lines.pop(0)]
while lines:
    # 取出当前已合并线段的第一个和最后一个终点
    current_line = merged_lines_new[-1]

    # 找到最接近的line
    nearest_line, nearest_index, reverse_connected, reverse_next = find_nearest_line(current_line, lines, tolerance_z_connection, tolerance_xy_connection)

    if nearest_line is not None and nearest_index != -1:
        # 合并最近的line，并进行插值
        new_line = merge_lines_with_interpolation(current_line, nearest_line, reverse_line1=reverse_connected, reverse_line2=reverse_next)
        merged_lines_new[-1] = new_line  # 更新已合并的最后一条线段
        lines.pop(nearest_index)  # 移除已合并的线段
    else:
        # 如果没有合适的曲线，则开始新的一条线段
        merged_lines_new.append(lines.pop(0))

# 绘制最终合并后的曲线
for line in merged_lines_new:
    plt.plot(line[:, 0], line[:, 1])
    plt.text(line[0, 0], line[0, 1], 'start', fontsize=12)
    plt.text(line[-1, 0], line[-1, 1], 'end', fontsize=12)

save_path = f'/home/ubuntu/junhaoge/real_world_simulation/data_utils/map_boundary/map_boundary_{scene_name}.png'
plt.savefig(save_path)