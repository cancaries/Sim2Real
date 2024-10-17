import numpy as np
import rpds
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, Point
import os
# 假设 tolerance_z 是允许的 z 轴变化幅度
tolerance_z = 1.0
tolerance_xy = 2.0  # xy 平面允许的最大距离
closure_threshold = 2.0  # 闭合判断的阈值
tolerance_z_connection = 5.0
tolerance_xy_connection = 60.0  # 连接两个线段时，xy 平面允许的最大距离


def detect_route_interaction(test_path, reference_path, interaction_range=75.0):
    """
    使用 Shapely 库检测路径交互。
    test_path: 测试路径 (列表形式，每个元素是 (x, y) 坐标)
    reference_path: 参考路径 (列表形式，每个元素是 (x, y) 坐标)
    interaction_range: 交互范围，默认5米
    返回 True 表示有交互，False 表示没有交互
    """
    # 将参考路径转换为 LineString
    reference_line = LineString(reference_path)
    
    # 为参考路径创建一个缓冲区范围（例如 5 米）
    reference_buffer = reference_line.buffer(interaction_range,cap_style='square')
    
    # 检查测试路径的每个点是否与参考路径的缓冲区有交集
    for point in test_path:
        point_geometry = Point(point)
        if reference_buffer.intersects(point_geometry):
            return True  # 如果有点与缓冲区相交，则存在交互
    
    return False

# 计算两点的欧氏距离，并且判断z轴差异是否在容忍范围内
def distance_with_tolerance(p1, p2, tolerance_z, tolerance_xy):
    xy_dist = np.linalg.norm(p1[:2] - p2[:2])  # 只计算 x, y 轴的距离
    z_diff = abs(p1[2] - p2[2]) 
    if z_diff <= tolerance_z and xy_dist <= tolerance_xy:
        return xy_dist
    else:
        return float('inf')

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
        # 跳过已闭合线段
        if is_closed(line, closure_threshold):
            continue

        start_point = line[0]
        end_point = line[-1]

        dist1 = distance_with_tolerance(last_point_of_connected, start_point, tolerance_z, tolerance_xy)
        dist2 = distance_with_tolerance(last_point_of_connected, end_point, tolerance_z, tolerance_xy)
        dist3 = distance_with_tolerance(first_point_of_connected, start_point, tolerance_z, tolerance_xy)
        dist4 = distance_with_tolerance(first_point_of_connected, end_point, tolerance_z, tolerance_xy)

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
root_path = '/home/ubuntu/junhaoge/real_world_simulation/data/waymo_multi_view/'
for path in os.listdir(root_path):
    lines = []

    scene_path = os.path.join(root_path,path)
    if not os.path.isdir(scene_path):
        continue
    map_feature_path = os.path.join(scene_path,'map_feature.json')
    ego_pose_path = os.path.join(scene_path,'ego_pose.json')
    scene_name = map_feature_path.split('/')[-2]
    import json
    with open(map_feature_path, 'r') as f:
        map_feature = json.load(f)
    with open(ego_pose_path, 'r') as f:
        ego_pose = json.load(f)
    ego_z = []
    ego_xy = []
    for ego_pose_tmp in ego_pose:
        ego_xy.append(ego_pose_tmp['location'][:2])
        ego_z.append(ego_pose_tmp['location'][2])
    # min_ego_z = min(ego_z) - 5.0
    # max_ego_z = max(ego_z) + 5.0
    min_ego_z = min(ego_z) - 3.0
    max_ego_z = max(ego_z) + 3.0
    patch_size = [110,110]
    # 以ego车辆的轨迹为参照，以patch_size为大小，生成patch
    
    
    for lane_id, lane_info in map_feature.items():
        if 'road_edge_type' not in lane_info:
            continue
        lane_new = []
        if lane_info['feature_type'] == 'road_edge' and lane_info['road_edge_type'] == 1:
            for point in lane_info['polyline']:
                if point[2] < min_ego_z or point[2] > max_ego_z:
                    continue
                lane_new.append(point)
            # lines.append(np.array(lane_info['polyline']))
            if detect_route_interaction(lane_new, ego_xy[:2], interaction_range=51.0):
                if len(lane_new) > 0:
                    lines.append(np.array(lane_new))
    print("#"*10)
    print(scene_name,'  ',ego_xy[0])
    for line in lines:
        plt.plot(line[:, 0], line[:, 1],color='red')
        # plt.text(line[0, 0], line[0, 1], 'start', fontsize=12)
        # plt.text(line[-1, 0], line[-1, 1], 'end', fontsize=12)
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
    merged_lines_new = [lines.pop(0)]  # 将第一条曲线作为初始曲线
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

    lines = merged_lines_new.copy()
    merged_lines_new = [lines.pop(0)]  # 将第一条曲线作为初始曲线
    while lines:
        # 取出当前已合并线段的第一个和最后一个终点
        current_line = merged_lines_new[-1]

        # 找到最接近的line
        nearest_line, nearest_index, reverse_connected, reverse_next = find_nearest_line(current_line, lines, tolerance_z_connection*1.5, tolerance_xy_connection*1.5)

        if nearest_line is not None and nearest_index != -1:
            # 合并最近的line，并进行插值
            new_line = merge_lines_with_interpolation(current_line, nearest_line, reverse_line1=reverse_connected, reverse_line2=reverse_next)
            merged_lines_new[-1] = new_line  # 更新已合并的最后一条线段
            lines.pop(nearest_index)  # 移除已合并的线段
        else:
            # 如果没有合适的曲线，则开始新的一条线段
            merged_lines_new.append(lines.pop(0))

    final_lines = []

    for line in merged_lines_new:
        start = line[0]
        end = line[-1]
        # 如果起点和终点足够近，则闭合并进行插值
        if distance_with_tolerance(start, end, tolerance_z_connection, tolerance_xy_connection) <= tolerance_xy_connection:
            interp_points = interpolate_between_points(start, end, 10)
            line = np.concatenate((line, interp_points), axis=0)  # 闭合
            final_lines.append(line)
        else:
            final_lines.append(line)


    # 绘制最终合并后的曲线
    # for line in final_lines:
    #     plt.plot(line[:, 0], line[:, 1],color='black')
        # plt.text(line[0, 0], line[0, 1], 'start', fontsize=12)
        # plt.text(line[-1, 0], line[-1, 1], 'end', fontsize=12)


    # contours = []
    # for line in final_lines:
    #     # 使用 ConvexHull 来找到曲线的最外层点
    #     # hull = ConvexHull(line)

    #     # # 获取凸包顶点的坐标
    #     # hull_points = line[hull.vertices]
    #     contours.append(line[:, :2])
    # # 画出所有的contours
    
    # # 画出面积最大的contour
    # max_area = 0
    # max_contour = None
    # contours_max_idx = 0
    # for idx, contour in enumerate(contours):
    #     area = np.abs(np.sum(contour[:-1, 0] * contour[1:, 1] - contour[1:, 0] * contour[:-1, 1])) / 2
    #     if area > max_area:
    #         max_area = area
    #         max_contour = contour
    #         contours_max_idx = idx
    
    # for idx, contour in enumerate(contours):
    #     if idx != contours_max_idx:
    #         plt.fill(contour[:, 0], contour[:, 1], color='lightgrey')
    #     else:
    #         plt.fill(max_contour[:, 0], max_contour[:, 1], color='lightblue')


        
    save_path = f'/home/ubuntu/junhaoge/real_world_simulation/data_utils/map_boundary/map_boundary_{scene_name}.png'
    plt.savefig(save_path)
    plt.close()