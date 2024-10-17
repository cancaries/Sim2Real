from turtle import color
from shapely.geometry import Polygon, LineString, Point
import numpy as np

def get_bbox_corners(location, heading, size):
    """
    根据车辆的位置、航向角、尺寸计算四个顶点的坐标
            
            :param location: 车辆的位置[x,y]
            :param heading: 航向角yaw
            :param size: 车辆的尺寸[w,l]
            
            :return: 四个顶点的坐标
    """
    x, y = location[:2]
    width, length = size[1], size[0] # 但这边size的顺序是[l,w]

    # 计算四个顶点的相对位置
    corners = np.array([
        [-width/2, -length/2],
        [width/2, -length/2],
        [width/2, length/2],
        [-width/2, length/2]
    ])
    
    # 旋转矩阵根据航向角进行旋转
    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])
    
    # 应用旋转矩阵和位置偏移
    rotated_corners = np.dot(rotation_matrix,corners.T).T
    translated_corners = rotated_corners + np.array([x, y])
    
    return translated_corners

def is_collision(target_vehicle_loc,
                     target_vehicle_yaw,
                     target_vehicle_bbox,
                     reference_vehicle_loc,
                     reference_vehicle_yaw,
                     reference_vehicle_bbox):
    # 计算两个车辆的四个顶点
    corners1 = get_bbox_corners(target_vehicle_loc, target_vehicle_yaw, target_vehicle_bbox)
    corners2 = get_bbox_corners(reference_vehicle_loc, reference_vehicle_yaw, reference_vehicle_bbox)
    
    # 使用shapely库判断两个多边形是否有重叠区域
    polygon1 = Polygon(corners1)
    polygon2 = Polygon(corners2)
    # if polygon1.intersects(polygon2):
    import matplotlib.pyplot as plt
    plt.plot(*polygon1.exterior.xy, label='target_vehicle',color='red')
    plt.plot(*polygon2.exterior.xy, label='reference_vehicle')
    plt.savefig('/home/ubuntu/junhaoge/real_world_simulation/agent_control/agents/navigation/tools/collision.png')
    plt.close()
    return polygon1.intersects(polygon2)
target_vehicle_loc = [10,10,0]
target_vehicle_yaw = np.pi / 4
target_vehicle_bbox = [2,4]
reference_vehicle_loc = [9,8,0]
reference_vehicle_yaw = np.pi / 4
reference_vehicle_bbox = [2,4]
print(is_collision(target_vehicle_loc,
                     target_vehicle_yaw,
                     target_vehicle_bbox,
                     reference_vehicle_loc,
                     reference_vehicle_yaw,
                     reference_vehicle_bbox))