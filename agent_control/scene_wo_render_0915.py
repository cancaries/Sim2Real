from __future__ import print_function
from calendar import c
import json
import random
random.seed(303)
import time
import math
from weakref import ref
from agents.navigation.tools.misc import is_within_distance,calculate_distance,\
                                        get_bbox_corners,calculate_movement_vector,\
                                        calculate_angle_between_vectors,is_collision,calculate_relative_vector,\
                                        detect_route_interaction,build_transform_path_from_ego_pose_data
import numpy as np
import datetime
import copy
def KinematicModel(x, y, yaw, v, a, delta, f_len, r_len, dt):
    beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
    x = x + v * math.cos(yaw + beta) * dt
    y = y + v * math.sin(yaw + beta) * dt
    yaw = yaw + (v / f_len) * math.sin(beta) * dt 
    v = v + a * dt
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi # normalize the angle range from [-pi, pi]
    v = max(0, v)
    return x, y, yaw, v

class Scene_wo_render(object):
    """
    场景类，用于管理场景中的所有实体
    """
    _data_root = None
    _scene_name = None
    _map = None
    _ego_vehicle = None
    _ego_vehicle_control = None
    _static_agent_loc_dict = dict()
    _agent_dict = dict()
    _agent_control_dict = dict()
    _agent_del_dict = dict()
    _all_actor_dict = dict()
    _FPS = 10
    _start_time = None
    _car_dict_sequence = None
    _debug_mode = False
    _ego_time_step = 0
    _skip_ego = False
    @staticmethod
    def initialize_scene(config):
        """
        初始化场景

        Args:
            config (dict): 场景配置
        """

        Scene_wo_render._data_root = config['_data_root']
        Scene_wo_render._scene_name = config['_scene_name']
        if '_FPS' in config.keys():
            Scene_wo_render._FPS = config['_FPS']
        Scene_wo_render.create_map(config['_map_config_path'])
        Scene_wo_render.initilize_static_agents(config['_static_actor_config_path'])
        Scene_wo_render.initialize_ego_vehicle(config['_ego_pose_path'])
        transform_path = build_transform_path_from_ego_pose_data(ego_pose_data)
        Scene_wo_render.initialize_ego_vehicle_w_transform_path(transform_path)
        Scene_wo_render._map.refine_spawn_points(ego_init_point=transform_path[0].location)
        Scene_wo_render.generate_background_agents(spawn_num=10) 
        # Scene_wo_render.generate_background_agents(random_agents=True,ego_forward_clear=True) 
        Scene_wo_render.set_debug_mode(True)
        # Scene_wo_render.refine_spawn_points_w_static()

        # Scene_wo_render.generate_background_agents()
        Scene_wo_render._start_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        
    @staticmethod
    def create_map(map_config_path):
        # 加载JSON
        map_config_path = map_config_path
        with open(map_config_path, 'r') as file:
            map_data = json.load(file)
        from agents.map import Map
        # 创建地图
        Scene_wo_render._map = Map(map_data)
    
    @staticmethod
    def initialize_ego_vehicle_w_transform_path(original_path,behavior_type='normal'):
        """
        初始化ego车辆

        Args:
            original_path (list): 起始路径

        Returns:
            bool: 是否初始化成功
        """
        from agents.navigation.behavior_agent_test import BehaviorAgent
        agent_config = dict()
        agent_config['name'] = 'ego_vehicle'
        agent_config['vehicle_type'] = 'car'
        agent_config['vehicle_bbox'] = (1.5,3.0,1.5)
        agent_config['f_len'] = 1.3
        agent_config['r_len'] = 1.3
        agent_config['control_angel'] = 50.0
        waypoint_path = Scene_wo_render._map.generate_waypoint_path_from_transform_path(original_path)
        agent_config['initial_path'] = waypoint_path
        agent_config['behavior'] = behavior_type
        Scene_wo_render._ego_vehicle = BehaviorAgent(agent_config)
        Scene_wo_render._agent_dict['ego_vehicle'] = BehaviorAgent(agent_config)
        return True

    @staticmethod
    def initialize_ego_vehicle(ego_vehicle_pose_path,behavior_type='normal'):
        """
        初始化ego车辆

        Args:
            original_path (list): 起始路径

        Returns:
            bool: 是否初始化成功
        """
        from agents.navigation.behavior_agent_test import BehaviorAgent
        agent_config = dict()
        agent_config['name'] = 'ego_vehicle'
        agent_config['vehicle_type'] = 'car'
        agent_config['vehicle_bbox'] = (1.5,3.0,1.5)
        agent_config['f_len'] = 1.3
        agent_config['r_len'] = 1.3
        agent_config['control_angel'] = 50.0
        agent_config['initial_path'] = original_path
        agent_config['behavior'] = behavior_type
        Scene_wo_render._ego_vehicle = BehaviorAgent(agent_config)
        Scene_wo_render._agent_dict['ego_vehicle'] = BehaviorAgent(agent_config)
        return True

    @staticmethod
    def initilize_static_agents(static_agent_config_path):
        from agents.navigation.behavior_agent_static import BehaviorAgent
        with open(static_agent_config_path, 'r') as file:
            static_agent_config = json.load(file)
        for agent_idx, (agent_id, static_agent_config) in enumerate(static_agent_config.items()):
            agent_config = dict()
            agent_config['name'] = 'static_agent_' + str(agent_idx)
            agent_config['vehicle_type'] = None
            agent_config['vehicle_bbox'] = static_agent_config['size']
            location = static_agent_config['location']
            rotation = static_agent_config['rotation']
            waypoint_config = Scene_wo_render._map.build_waypoint_config(location, rotation)
            waypoint = Scene_wo_render._map.build_waypoint(waypoint_config)
            agent_config['static_waypoint'] = waypoint
            agent = BehaviorAgent(agent_config)
            Scene_wo_render._agent_dict[agent_config['name']] = agent
            Scene_wo_render._static_agent_loc_dict[agent_config['name']] = dict()
            Scene_wo_render._static_agent_loc_dict[agent_config['name']]['location'] = waypoint.transform.location[:2]
            Scene_wo_render._static_agent_loc_dict[agent_config['name']]['yaw'] = waypoint.transform.rotation[2]
            Scene_wo_render._static_agent_loc_dict[agent_config['name']]['bbox'] = agent.bounding_box

    @staticmethod
    def refine_spawn_points_w_static():
        """
        移除静态车辆周围的可部署点
        """
        for agent_name, agent in Scene_wo_render._agent_dict.items():
            if 'static' in agent_name:
                Scene_wo_render._map.refine_spawn_points_w_location(agent.cur_waypoint.transform.location)

    @staticmethod
    def refine_route_w_static(route,self_size,distance_threshold=5):
        """
        修改与静态车辆冲突的随机路径

        Args:
            route (list): 随机路径

        Returns:
            list: 修改后的随机路径
        """
        # for agent_name, agent in Scene_wo_render._static_agent_loc_dict.items():
        #     if 'static' in agent_name:
        #         static_loc = agent['location']
        #         static_yaw = agent['yaw']
        #         static_bbox = agent['bbox']
        #         if calculate_distance(route[0].transform.location[:2],static_loc) > distance_threshold:
        #             continue
        #         influenced_num = False
        #         for idx, path_waypoint in enumerate(route):
        #             loc = path_waypoint.transform.location[:2]
        #             yaw = path_waypoint.transform.rotation[2]
        #             offset = 0.3
        #             collision_flag = False
        #             while is_collision(loc, yaw, self_size, static_loc, static_yaw, static_bbox):
        #                 offset += 0.2
        #                 waypoint_new = copy.deepcopy(path_waypoint)
        #                 waypoint_new = Scene_wo_render._map.get_waypoint_w_offset(waypoint_new, offset)
        #                 route[idx] = waypoint_new
        #                 loc = waypoint_new.transform.location[:2]
        #                 collision_flag = True
        #             if collision_flag:
        #                 offset += 0.2
        #                 waypoint_new = Scene_wo_render._map.get_waypoint_w_offset(waypoint_new, offset)
        #                 route[idx] = waypoint_new
        # return route
        
        for idx, path_waypoint in enumerate(route):
            loc = path_waypoint.transform.location[:2]
            yaw = path_waypoint.transform.rotation[2]
            offset = 0.3
            collision_flag = False
            for agent_name, agent in Scene_wo_render._static_agent_loc_dict.items():
                if 'static' in agent_name:
                    static_loc = agent['location']
                    static_yaw = agent['yaw']
                    static_bbox = agent['bbox']
                    if calculate_distance(route[0].transform.location[:2],static_loc) > distance_threshold:
                        continue
                while is_collision(loc, yaw, self_size, static_loc, static_yaw, static_bbox):
                    offset += 0.2
                    waypoint_new = copy.deepcopy(path_waypoint)
                    waypoint_new = Scene_wo_render._map.get_waypoint_w_offset(waypoint_new, offset)
                    route[idx] = waypoint_new
                    loc = waypoint_new.transform.location[:2]
                    collision_flag = True
                if collision_flag:
                    offset += 0.2
                    waypoint_new = Scene_wo_render._map.get_waypoint_w_offset(waypoint_new, offset)
                    route[idx] = waypoint_new
                    break
        return route

    @staticmethod
    def check_route_valid_w_ego_route(route,width_threshold=10):
        """
        检查路径是否与在ego route的width_threshold范围内,左右两侧10m，前后5m的范围内,如果是则返回False

        Args:
            route (list): 随机路径

        Returns:
            bool: 路径是否有效
        """
        if Scene_wo_render._ego_vehicle is not None:
            ego_route = [x.transform.location[:2] for x in Scene_wo_render._ego_vehicle.plan_path]
        else:
            print('Ego vehicle is not initialized!')
            return True
        
        if route is not None:
            test_route = [x.transform.location[:2] for x in route]
        else:
            print('Route is None!')
            return True
        return detect_route_interaction(test_route,ego_route,width_threshold)

    @staticmethod
    def generate_background_agents(spawn_num=10, 
                                   random_agents=False,
                                   close_to_ego=False,
                                   close_threshold=30,
                                   ego_forward_clear=False,
                                   forward_threshold=10,
                                   same_lane=False):
        """
        生成带有随机路径的背景车辆并添加到场景中

        Args:
            random (bool, optional): _description_. Defaults to True.
            spawn_num (int, optional): _description_. Defaults to 10.
        """
        from agents.navigation.behavior_agent_test import BehaviorAgent
        spawn_points = Scene_wo_render._map.get_spawn_points()
        spawn_points_num = len(spawn_points)
        # vehicle_type_list = ['car','SUV','truck','bus']
        # vehicle_type_list = ['car']
        # vehicle_type_list = ['car','SUV','truck']
        vehicle_type_list = ['car','SUV']
        weights=[0.8,0.2]
        vehicle_bbox_dict = {'car':(1.5,3.0,1.5),'SUV':(1.8,4.0,1.8),'truck':(2.0,6.0,2.0),'bus':(2.5,12.0,2.5)}
        vehicle_fr_len_dict = {'car':1.3,'SUV':1.5,'truck':2.5}
        vehicle_control_angle_dict = {'car':50.0,'SUV':50.0,'truck':50.0}
        # behavior_type_list = ['cautious','normal']
        # behavior_type_list = ['cautious']
        # behavior_type_list = ['aggressive']
        # behavior_type_list = ['extreme_aggressive']
        # behavior_type_list = ['normal']
        behavior_type_list = ['normal','aggressive']
        # behavior_type_list = ['cautious','normal','aggressive']
        random.seed(303)
        spawned_num = 0
        if random_agents:
            # 随机选取可部署点的十分之一数量的点进行部署
            spawn_points_shuffle = random.shuffle(spawn_points)
            spawn_points_filter = random.sample(spawn_points_shuffle, int(spawn_points_num/2))
            for idx, spawn_point in enumerate(spawn_points_filter):
                if ego_forward_clear:
                    if Scene_wo_render._ego_vehicle is not None:
                        ego_loc = Scene_wo_render._ego_vehicle.cur_waypoint.transform.location
                        ego_yaw = Scene_wo_render._ego_vehicle.cur_waypoint.transform.rotation[2]
                        ego_bbox = Scene_wo_render._ego_vehicle.bounding_box
                        if is_within_distance(spawn_point, ego_loc, ego_yaw,
                              forward_threshold, 45, 0):
                            continue
                
                agent_config = dict()
                agent_config['name'] = 'background_agent_' + str(idx)
                vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                agent_config['vehicle_type'] = vehicle_type
                agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                random_path = Scene_wo_render._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                random_path = Scene_wo_render.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
                if not Scene_wo_render.check_route_valid_w_ego_route(random_path):
                    continue
                agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                agent_config['initial_path'] = random_path
                agent_config['behavior'] = random.choice(behavior_type_list)
                # agent = None
                agent = BehaviorAgent(agent_config)
                Scene_wo_render._agent_dict[agent_config['name']] = agent
                spawned_num += 1
                if spawned_num >= spawn_num:
                    break
        elif close_to_ego and Scene_wo_render._ego_vehicle is not None:
            ego_loc = Scene_wo_render._ego_vehicle.cur_waypoint.transform.location
            ego_yaw = Scene_wo_render._ego_vehicle.cur_waypoint.transform.rotation[2]
            ego_bbox = Scene_wo_render._ego_vehicle.bounding_box
            spawn_points_shuffle = random.shuffle(spawn_points)
            for idx, spawn_point in enumerate(spawn_points_shuffle):
                if same_lane:
                    if Scene_wo_render._ego_vehicle is not None:
                        ego_lane_id = Scene_wo_render._ego_vehicle.cur_waypoint.lane_id
                        spawn_lane_id, _, _ = Scene_wo_render._map.find_nearest_lane_point(spawn_point)
                        if ego_lane_id != spawn_lane_id:
                            continue
                spawn_loc = spawn_point
                if ego_forward_clear:
                    if Scene_wo_render._ego_vehicle is not None:
                        ego_loc = Scene_wo_render._ego_vehicle.cur_waypoint.transform.location
                        ego_yaw = Scene_wo_render._ego_vehicle.cur_waypoint.transform.rotation[2]
                        ego_bbox = Scene_wo_render._ego_vehicle.bounding_box
                        if is_within_distance(spawn_point, ego_loc, ego_yaw,
                              forward_threshold, 45, 0):
                            continue
                spawn_bbox = vehicle_bbox_dict['car']
                if calculate_distance(spawn_loc, ego_loc) < close_threshold:
                    agent_config = dict()
                    agent_config['name'] = 'background_agent_' + str(spawned_num)
                    # vehicle_type = random.choice(vehicle_type_list)
                    vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                    agent_config['vehicle_type'] = vehicle_type
                    agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                    random_path = Scene_wo_render._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                    random_path = Scene_wo_render.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
                    if not Scene_wo_render.check_route_valid_w_ego_route(random_path):
                        continue
                    agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]
                    agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]
                    agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                    agent_config['initial_path'] = random_path
                    agent_config['behavior'] = random.choice(behavior_type_list)
                    # agent = None
                    agent = BehaviorAgent(agent_config)
                    Scene_wo_render._agent_dict[agent_config['name']] = agent
                    spawned_num += 1
                    if spawned_num >= spawn_num:
                        break
                
        else:
            spawn_points_filter = random.shuffle(spawn_points_filter)
            for idx, spawn_point in enumerate(spawn_points_filter):
                if ego_forward_clear:
                    if Scene_wo_render._ego_vehicle is not None:
                        ego_loc = Scene_wo_render._ego_vehicle.cur_waypoint.transform.location
                        ego_yaw = Scene_wo_render._ego_vehicle.cur_waypoint.transform.rotation[2]
                        ego_bbox = Scene_wo_render._ego_vehicle.bounding_box
                        if is_within_distance(spawn_point, ego_loc, ego_yaw,
                              forward_threshold, 45, 0):
                            continue
                agent_config = dict()
                agent_config['name'] = 'background_agent_' + str(spawned_num)
                # vehicle_type = random.choice(vehicle_type_list)
                vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                agent_config['vehicle_type'] = vehicle_type
                agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                random_path = Scene_wo_render._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                random_path = Scene_wo_render.refine_route_w_static(random_path,agent_config['vehicle_bbox'])
                if not Scene_wo_render.check_route_valid_w_ego_route(random_path):
                    continue
                agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                agent_config['initial_path'] = random_path
                agent_config['behavior'] = random.choice(behavior_type_list)
                # agent = None
                agent = BehaviorAgent(agent_config)
                Scene_wo_render._agent_dict[agent_config['name']] = agent
                spawned_num += 1
                if spawned_num >= spawn_num:
                    break

    @staticmethod
    def generate_background_agents_for_test(spawn_num=10, 
                                   random_agents=False,
                                   close_to_ego=False,
                                   close_threshold=30,
                                   ego_forward=False,
                                   forward_threshold=10,
                                   same_lane=False):
        """
        生成带有随机路径的背景车辆并添加到场景中

        Args:
            random (bool, optional): _description_. Defaults to True.
            spawn_num (int, optional): _description_. Defaults to 10.
        """
        from agents.navigation.behavior_agent_test import BehaviorAgent
        spawn_points = Scene_wo_render._map.get_spawn_points()
        spawn_points_num = len(spawn_points)
        # vehicle_type_list = ['car','SUV','truck','bus']
        # vehicle_type_list = ['car']
        # vehicle_type_list = ['car','SUV','truck']
        vehicle_type_list = ['car','SUV']
        weights=[0.6,0.4]
        vehicle_bbox_dict = {'car':(1.5,3.0,1.5),'SUV':(1.8,4.0,1.8),'truck':(2.0,6.0,2.0),'bus':(2.5,12.0,2.5)}
        vehicle_fr_len_dict = {'car':1.3,'SUV':1.5,'truck':2.5}
        vehicle_control_angle_dict = {'car':60.0,'SUV':60.0,'truck':60.0}
        # behavior_type_list = ['cautious','normal']
        # behavior_type_list = ['cautious']
        # behavior_type_list = ['aggressive']
        # behavior_type_list = ['extreme_aggressive']
        # behavior_type_list = ['normal']
        behavior_type_list = ['normal','aggressive']
        # behavior_type_list = ['cautious','normal','aggressive']
        random.seed(303)
        spawned_num = 0
        for idx, spawn_point in enumerate(spawn_points):
            if spawned_num >= spawn_num:
                break
            if ego_forward:
                if Scene_wo_render._ego_vehicle is not None:
                    ego_loc = Scene_wo_render._ego_vehicle.cur_waypoint.transform.location
                    ego_yaw = Scene_wo_render._ego_vehicle.cur_waypoint.transform.rotation[2]
                    ego_bbox = Scene_wo_render._ego_vehicle.bounding_box
                    if is_within_distance(spawn_point, ego_loc, ego_yaw,
                            forward_threshold, 30, 0):
                        random_path = Scene_wo_render._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                        agent_config = dict()
                        agent_config['name'] = 'background_agent_' + str(idx)
                        # vehicle_type = random.choice(vehicle_type_list)
                        vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                        agent_config['vehicle_type'] = vehicle_type
                        agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                        agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]
                        agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]
                        agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                        agent_config['initial_path'] = random_path
                        agent_config['behavior'] = random.choice(behavior_type_list)
                        # agent = None
                        agent = BehaviorAgent(agent_config)
                        Scene_wo_render._agent_dict[agent_config['name']] = agent
                        spawned_num += 1
            else:
                random_path = Scene_wo_render._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                agent_config = dict()
                agent_config['name'] = 'background_agent_' + str(idx)
                # vehicle_type = random.choice(vehicle_type_list)
                vehicle_type = random.choices(vehicle_type_list,weights=weights)[0]
                agent_config['vehicle_type'] = vehicle_type
                agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                agent_config['f_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['r_len'] = vehicle_fr_len_dict[vehicle_type]
                agent_config['control_angel'] = vehicle_control_angle_dict[vehicle_type]
                agent_config['initial_path'] = random_path
                agent_config['behavior'] = random.choice(behavior_type_list)
                # agent = None
                agent = BehaviorAgent(agent_config)
                Scene_wo_render._agent_dict[agent_config['name']] = agent
                spawned_num += 1


    @staticmethod
    def get_vehicle_speed(vehicle_name):
        vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
        return vehicle_agent.get_speed()

    @staticmethod
    def get_vehicle_speed_m(vehicle_name):
        vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
        return vehicle_agent.get_speed()/3.6

    @staticmethod
    def get_vehicle_bbox(vehicle_name):
        vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
        return vehicle_agent.bounding_box

    @staticmethod
    def get_vehicle_location(vehicle_name):
        vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
        return vehicle_agent.cur_waypoint.transform.location

    @staticmethod
    def get_vehicle_waypoint(vehicle_name):
        # print(Scene_wo_render._agent_dict.keys())
        if vehicle_name not in Scene_wo_render._agent_dict.keys():
            return None
        vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
        return vehicle_agent.cur_waypoint
    @staticmethod
    def get_vehicle_control(vehicle_name):
        vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
        return vehicle_agent.cur_control

    @staticmethod
    def generate_route_w_waypoints(start_waypoint, end_waypoint):
        """
        生成两个waypoint之间的路径

        Args:
            start_waypoint (Waypoint): 起始waypoint
            end_waypoint (Waypoint): 结束waypoint

        Returns:    
            list: 路径上的waypoints
        """
        return Scene_wo_render._map.plan_path_w_waypoints(start_waypoint, end_waypoint)

    @staticmethod
    def adjust_position(target_vehicle_agent, reference_vehicle_agent):
        # 获取当前和上一帧的位置
        current_position = target_vehicle_agent.cur_waypoint.transform.location[:2]
        previous_position = target_vehicle_agent.last_waypoint.transform.location[:2]
        target_vehicle_yaw = target_vehicle_agent.last_waypoint.transform.rotation[2]
        target_vehicle_agent.cur_waypoint.transform.set_rotation([0, 0, target_vehicle_yaw])
        # 计算车辆运动方向的单位向量
        movement_direction = calculate_relative_vector(previous_position, current_position)
        movement_distance = np.linalg.norm(movement_direction)
        
        if movement_distance > 0:
            movement_direction /= movement_distance  # 单位化

        # 二分查找最接近的非碰撞位置
        low, high = 0, movement_distance
        best_position = current_position
        reference_vehicle_loc = reference_vehicle_agent.cur_waypoint.transform.location[:2]
        reference_vehicle_yaw = reference_vehicle_agent.cur_waypoint.transform.rotation[2]
        reference_vehicle_bbox = reference_vehicle_agent.bounding_box
        while high - low > 1e-1:  # 精度控制
            mid = (low + high) / 2
            test_position = previous_position + movement_direction * mid
            target_vehicle_yaw = np.arctan2(test_position[1] - previous_position[1], test_position[0] - previous_position[0])
            target_vehicle_agent.cur_waypoint.transform.set_location(test_position)
            target_vehicle_agent.cur_waypoint.transform.set_rotation([0, 0, target_vehicle_yaw])
            target_vehicle_loc = target_vehicle_agent.cur_waypoint.transform.location[:2]
            target_vehicle_bbox = target_vehicle_agent.bounding_box
            if is_collision(target_vehicle_loc, 
                            target_vehicle_yaw, 
                            target_vehicle_bbox, 
                            reference_vehicle_loc, 
                            reference_vehicle_yaw, 
                            reference_vehicle_bbox):
                high = mid  # 缩小范围，靠近上一帧位置
            else:
                best_position = test_position
                low = mid  # 缩小范围，靠近当前帧位置
        if 'ego' in target_vehicle_agent.vehicle_name:
            Scene_wo_render._ego_time_step = Scene_wo_render._ego_time_step - 1 if Scene_wo_render._ego_time_step > 0 else 0
        # 调整到最佳位置
        target_vehicle_agent.cur_waypoint.transform.set_location(best_position)
        # target_vehicle_agent.set_speed(0.0)  # 停车


    @staticmethod
    def check_collision(angle_threshold=np.pi/4):
        """通过朝向和相对位置的夹角来判断哪个车辆是造成碰撞的车辆"""
        agent_dict = Scene_wo_render._agent_dict  # 提取agent字典，避免每次都访问Scene_wo_render._agent_dict
        agent_ids = list(agent_dict.keys())  # 获取所有车辆的ID列表

        for i, id1 in enumerate(agent_ids):
            target_vehicle = agent_dict[id1]
            target_vehicle_loc = target_vehicle.cur_waypoint.transform.location[:2]
            target_vehicle_yaw = target_vehicle.cur_waypoint.transform.rotation[2]
            target_vehicle_bbox = target_vehicle.bounding_box
            target_vehicle_bbox = [target_vehicle_bbox[0]+0.1, target_vehicle_bbox[1]+0.1, target_vehicle_bbox[2]]
            target_vehicle_last_loc = target_vehicle.last_waypoint.transform.location[:2]

            # 内循环从外循环的下一个元素开始，避免重复检测
            for id2 in agent_ids[i + 1:]:
                if Scene_wo_render._skip_ego and ('ego' in id1 or 'ego' in id2):
                    continue
                reference_vehicle = agent_dict[id2]
                reference_vehicle_loc = reference_vehicle.cur_waypoint.transform.location[:2]

                reference_vehicle_yaw = reference_vehicle.cur_waypoint.transform.rotation[2]
                reference_vehicle_bbox = reference_vehicle.bounding_box
                reference_vehicle_last_loc = reference_vehicle.last_waypoint.transform.location[:2]

                if is_collision(target_vehicle_loc,
                                target_vehicle_yaw,
                                target_vehicle_bbox,
                                reference_vehicle_loc,
                                reference_vehicle_yaw,
                                reference_vehicle_bbox):
                    if 'static' in id1:
                        Scene_wo_render.adjust_position(reference_vehicle, target_vehicle)
                        continue
                    elif 'static' in id2:
                        Scene_wo_render.adjust_position(target_vehicle, reference_vehicle)
                        continue

                    # 计算两辆车的相对位置向量
                    relative_vector = calculate_relative_vector(target_vehicle_loc, reference_vehicle_loc)

                    # 计算两辆车的朝向向量
                    heading_vector1 = np.array([np.cos(target_vehicle_yaw), np.sin(target_vehicle_yaw)])
                    heading_vector2 = np.array([np.cos(reference_vehicle_yaw), np.sin(reference_vehicle_yaw)])

                    # 计算运动方向与相对位置向量的夹角
                    angle1 = calculate_angle_between_vectors(heading_vector1, relative_vector)
                    angle2 = calculate_angle_between_vectors(heading_vector2, -relative_vector)

                    # 判断哪个车辆的朝向更可能是造成碰撞的
                    if angle1 < angle_threshold:
                        # print(f'Collision between {id1} and {id2}, caused by {id1}')
                        Scene_wo_render.adjust_position(target_vehicle, reference_vehicle)  # 调整主动车辆的位置
                    elif angle2 < angle_threshold:
                        # print(f'Collision between {id1} and {id2}, caused by {id2}')
                        Scene_wo_render.adjust_position(reference_vehicle, target_vehicle)  # 调整主动车辆的位置
                    else:
                        # 如果两辆车都没有明显朝向对方的迹象，可以根据其他标准处理
                        if np.linalg.norm(calculate_movement_vector(target_vehicle_loc, target_vehicle_last_loc)) \
                                > np.linalg.norm(calculate_movement_vector(reference_vehicle_loc, reference_vehicle_last_loc)):
                            Scene_wo_render.adjust_position(target_vehicle, reference_vehicle)
                        else:
                            Scene_wo_render.adjust_position(reference_vehicle, target_vehicle)


    @staticmethod
    def set_debug_mode(debug_mode):
        """
        设置debug模式,degub模式下会记录每一步的车辆信息,用于最后信息的记录以及作图

        Args:
            debug_mode (bool): 是否开启debug模式
        """
        Scene_wo_render._debug_mode = debug_mode
    
    @staticmethod
    def run_step_w_ego(cur_time_step, ego_control=False):
        time_step = 1.0/Scene_wo_render._FPS # 1s/5 = 0.2s，作为每一个时间间隔
        # 基于_debug_mode的设置进行初始化
        if Scene_wo_render._debug_mode:
            if Scene_wo_render._car_dict_sequence is None:
                Scene_wo_render._car_dict_sequence = []
            car_dict = dict()
        # 遍历所有仍然存在的agents
        if not ego_control:
            Scene_wo_render._skip_ego = True
        else:
            Scene_wo_render._skip_ego = False
        for agent_name, agent in Scene_wo_render._agent_dict.items():
            if agent_name == 'ego_vehicle':
                if not ego_control:
                    # print(f"cur_time_step: {cur_time_step} cur _ego_time_step: {Scene_wo_render._ego_time_step}", )
                    agent.scenario_trigger = False
                    next_ego_waypoint = agent.plan_path[Scene_wo_render._ego_time_step]
                    agent.set_last_waypoint(agent.cur_waypoint)
                    next_ego_loc = next_ego_waypoint.transform.location[:2] + [0]
                    next_ego_rot = next_ego_waypoint.transform.rotation
                    agent.cur_waypoint.transform.set_rotation(next_ego_rot)
                    agent.cur_waypoint.transform.set_location(next_ego_loc)
                    agent.update_information(next_ego_waypoint.transform)
                    # print(f"next_ego_waypoint.transform {next_ego_waypoint.transform.location} ego_vehicle control: {agent.cur_control.throttle}, {agent.cur_control.brake}, {agent.cur_control.steer}")
                    # print(agent.if_overtake, agent.if_tailgate)
                    # agent.cur_control.throttle = 0.0
                    # agent.cur_control.brake = 0.0
                    # agent.cur_control.steer = 0.0
                    Scene_wo_render._ego_time_step += 1
                    continue
            if 'static' in agent_name:
                continue
            # 获取当前agent的控制信息
            agent_control = Scene_wo_render._agent_control_dict[agent_name]
            # if agent_name == 'ego_vehicle':
                # print(f"ego_vehicle cur target waypoint: {agent.cur_waypoint.transform.location} ego_vehicle control: {agent_control.throttle}, {agent_control.brake}, {agent_control.steer}")
                # print(agent.if_overtake, agent.if_tailgate)
            # 如果agent的结束标志为真，则将其添加到删除列表中
            if agent.end_route_flag:
                Scene_wo_render._agent_del_dict[agent_name] = agent
                continue
            # 暂时设置前后轴距为1.3m
            agent_f_len = agent.f_len # 前轴距 unit: m
            agent_r_len = agent.r_len # 后轴距 unit: m
            agent_control_angel = agent.control_angel
            # agent具体控制信息的获取，包括油门、刹车、方向盘
            throttle = agent_control.throttle # throttle range:[0, 1] 最大值在local planner里被设置为了0.75 unit: None
            brake = agent_control.brake # brake range:[0, 1] 最大值再local planner里被设置为了0.3 unit: None
            steer = agent_control.steer # steer range:[-1, 1] 最大值在local planner里被设置为了0.8 unit: None
            delta = steer*agent_control_angel*math.pi/180 # steer range:[-1, 1] -> delta range:[-40, 40] unit: deg
            cur_agent_loc = agent.cur_waypoint.transform.location # location [x, y, z] unit：m
            cur_agent_rot = agent.cur_waypoint.transform.rotation # rotation [roll, pitch, yaw] unit: rad 
            agent_x = cur_agent_loc[0] # x_loc unit：m
            agent_y = cur_agent_loc[1] # y_loc unit：m
            agent_yaw = cur_agent_rot[2] # yaw range:[-pi, pi] unit: rad
            agent_v = copy.deepcopy(agent.speed) # velocity unit:km/h
            if brake >= 0.001:
                agent_a = -8.0 * brake # brake deceleration
            else:
                agent_a = 10.0 * throttle # acceleration
            agent_x_update, agent_y_update, agent_yaw_update, agent_speed_update = \
                KinematicModel(x=agent_x, 
                               y=agent_y,
                               yaw=agent_yaw,
                               v=agent_v / 3.6, # velocity unit: m/s
                               a=agent_a, # acceleration unit: m/s^2
                               delta=delta,
                               f_len=agent_f_len,
                               r_len=agent_r_len,
                               dt=time_step)
            agent_next_loc = [agent_x_update ,agent_y_update , 0.0] # location [x, y, z] unit：m
            agent_next_rot = [0.0, 0.0, agent_yaw_update] # rotation [roll, pitch, yaw] unit: rad
            agent.set_last_waypoint(agent.cur_waypoint)
            agent.cur_waypoint.transform.set_rotation(agent_next_rot)
            agent.cur_waypoint.transform.set_location(agent_next_loc)
            updated_transform = agent.cur_waypoint.transform
            agent.update_information(updated_transform)
            agent_speed_update_in_km = agent_speed_update * 3.6 # velocity unit: km/h
            agent.set_speed(agent_speed_update_in_km) # velocity unit: km/h
        Scene_wo_render.check_collision()
        if Scene_wo_render._debug_mode:
            for agent_name, agent in Scene_wo_render._agent_dict.items():
                car_dict[agent_name] = dict()
                car_dict[agent_name]['loc'] = agent.cur_waypoint.transform.location
                car_dict[agent_name]['rot'] = agent.cur_waypoint.transform.rotation
                car_dict[agent_name]['bbox'] = agent.bounding_box
                car_dict[agent_name]['if_overtake'] = agent.if_overtake
                car_dict[agent_name]['if_tailgate'] = agent.if_tailgate
                car_dict[agent_name]['if_static'] = agent.if_static
        
        for agent_name, agent in Scene_wo_render._agent_del_dict.items():
            del Scene_wo_render._agent_dict[agent_name]
            del Scene_wo_render._agent_control_dict[agent_name]
        if Scene_wo_render._debug_mode:
            Scene_wo_render._car_dict_sequence.append(car_dict)
        Scene_wo_render._agent_del_dict.clear()