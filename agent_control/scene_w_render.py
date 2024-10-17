from __future__ import print_function
import json
import random
import time
import math
from .agents.navigation.tools.misc import get_vehicle_bbox

def KinematicModel(x, y, yaw, v, a, delta, f_len, r_len, dt):
    beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
    x = x + v * math.cos(yaw + beta) * dt
    y = y + v * math.sin(yaw + beta) * dt
    yaw = yaw + (v / f_len) * math.sin(beta) * dt 
    v = v + a * dt
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
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
    _agent_dict = dict()
    _agent_control_dict = dict()
    _agent_del_dict = dict()
    _all_actor_dict = dict()
    _FPS = 10
    _start_time = None
    _car_dict_sequence = None
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
        # Scene_wo_render.generate_background_agents()
        Scene_wo_render._start_time = time.time()
        
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
    def check_collision():
        pass
    
    @staticmethod
    def initialize_ego_vehicle(original_path):
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
        waypoint_path = Scene_wo_render._map.generate_waypoint_path_from_transform_path(original_path)
        agent_config['initial_path'] = waypoint_path
        agent_config['behavior'] = 'normal'
        Scene_wo_render._ego_vehicle = BehaviorAgent(agent_config)
        Scene_wo_render._agent_dict['ego_vehicle'] = Scene_wo_render._ego_vehicle
        return True

    @staticmethod
    def generate_background_agents(random_agents=True,spawn_num=10):
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
        vehicle_type_list = ['car']
        vehicle_bbox_dict = {'car':(1.5,3.0,1.5),'SUV':(1.8,4.0,1.8),'truck':(2.5,10.0,2.5),'bus':(2.5,12.0,2.5)}
        # behavior_type_list = ['cautious','normal','aggressive']
        behavior_type_list = ['aggressive']

        if random_agents:
            # 随机选取可部署点的十分之一数量的点进行部署
            spawn_points_filter = random.sample(spawn_points, int(spawn_points_num/2))
            for idx, spawn_point in enumerate(spawn_points_filter):
                random_path = Scene_wo_render._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                agent_config = dict()
                agent_config['name'] = 'background_agent_' + str(idx)
                vehicle_type = random.choice(vehicle_type_list)
                agent_config['vehicle_type'] = vehicle_type
                agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                agent_config['initial_path'] = random_path
                agent_config['behavior'] = random.choice(behavior_type_list)
                # agent = None
                agent = BehaviorAgent(agent_config)
                Scene_wo_render._agent_dict[agent_config['name']] = agent
        else:
            if spawn_num > len(spawn_points):
                spawn_num = len(spawn_points)
            spawn_points_filter = random.sample(spawn_points, spawn_num)
            for idx, spawn_point in enumerate(spawn_points_filter):
                random_path = Scene_wo_render._map.generate_overall_plan_waypoints_w_refine(spawn_point)
                agent_config = dict()
                agent_config['name'] = 'background_agent_' + str(idx)
                vehicle_type = random.choice(vehicle_type_list)
                agent_config['vehicle_type'] = vehicle_type
                agent_config['vehicle_bbox'] = vehicle_bbox_dict[vehicle_type]
                agent_config['initial_path'] = random_path
                agent_config['behavior'] = random.choice(behavior_type_list)
                # agent = None
                agent = BehaviorAgent(agent_config)
                Scene_wo_render._agent_dict[agent_config['name']] = agent
    
    @staticmethod
    def get_vehicle_speed(vehicle_name):
        vehicle_agent = Scene_wo_render._agent_dict[vehicle_name]
        return vehicle_agent.get_speed()*3.6

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
    def set_debug_mode(debug_mode):
        """
        设置debug模式,degub模式下会记录每一步的车辆信息,用于最后信息的记录以及作图

        Args:
            debug_mode (bool): 是否开启debug模式
        """
        Scene_wo_render._debug_mode = debug_mode

    @staticmethod
    def check_collision():
        """
        根据车辆位置、航向角、bbox等信息检查是否发生碰撞,若碰撞则将车辆置于紧贴的位置
        """
        pass
                                    
    

    @staticmethod
    def run_step():
        time_step = 1.0/Scene_wo_render._FPS # 1s/5 = 0.2s，作为每一个时间间隔
        # 基于_debug_mode的设置进行初始化
        if Scene_wo_render._debug_mode:
            if Scene_wo_render._car_dict_sequence is None:
                Scene_wo_render._car_dict_sequence = []
            car_dict = dict()
        # 遍历所有仍然存在的agents
        for agent_name, agent in Scene_wo_render._agent_dict.items():
            if agent_name == 'ego_vehicle':
                continue
            # 获取当前agent的控制信息
            agent_control = Scene_wo_render._agent_control_dict[agent_name]
            # 如果agent的结束标志为真，则将其添加到删除列表中
            if agent.end_route_flag:
                Scene_wo_render._agent_del_dict[agent_name] = agent
                continue
            # 暂时设置前后轴距为1.3m
            agent_f_len = 1.3 # 前轴距 unit: m
            agent_r_len = 1.3 # 后轴距 unit: m
            # agent具体控制信息的获取，包括油门、刹车、方向盘
            throttle = agent_control.throttle # throttle range:[0, 1] 最大值在local planner里被设置为了0.75 unit: None
            brake = agent_control.brake # brake range:[0, 1] 最大值再local planner里被设置为了0.3 unit: None
            steer = agent_control.steer # steer range:[-1, 1] 最大值在local planner里被设置为了0.8 unit: None
            delta = steer*40*math.pi/180 # steer range:[-1, 1] -> delta range:[-40, 40] unit: deg
            cur_agent_loc = agent.cur_waypoint.transform.location # location [x, y, z] unit：m
            cur_agent_rot = agent.cur_waypoint.transform.rotation # rotation [roll, pitch, yaw] unit: rad 
            agent_x = cur_agent_loc[0] # x_loc unit：m
            agent_y = cur_agent_loc[1] # y_loc unit：m
            agent_yaw = cur_agent_rot[2] # yaw range:[-pi, pi] unit: rad
            agent_v = agent.speed # velocity unit:km/h
            if brake >= 0.001:
                agent_a = -1.0 * brake # brake deceleration
            else:
                agent_a = 2 * throttle # acceleration
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
            # if agent_name == 'background_agent_0':
            #     print(f'agent_x_update: {agent_x_update}, agent_y_update: {agent_y_update}, agent_yaw_update: {agent_yaw_update}, agent_speed_update: {agent_speed_update}')
            # update agent's velocity and location
            agent_next_loc = [agent_x_update ,agent_y_update , 0.0] # location [x, y, z] unit：m
            agent_next_rot = [0.0, 0.0, agent_yaw_update] # rotation [roll, pitch, yaw] unit: rad
            agent.cur_waypoint.transform.set_rotation(agent_next_rot)
            agent.cur_waypoint.transform.set_location(agent_next_loc)
            updated_transform = agent.cur_waypoint.transform
            agent.update_information(updated_transform)
            agent.set_speed(agent_speed_update * 3.6) # velocity unit: km/h
            if Scene_wo_render._debug_mode:
                car_dict[agent_name] = dict()
                car_dict[agent_name]['loc'] = agent_next_loc
                car_dict[agent_name]['rot'] = agent_next_rot
                car_dict[agent_name]['bbox'] = agent.bounding_box
        for agent_name, agent in Scene_wo_render._agent_del_dict.items():
            del Scene_wo_render._agent_dict[agent_name]
            del Scene_wo_render._agent_control_dict[agent_name]
        if Scene_wo_render._debug_mode:
            Scene_wo_render._car_dict_sequence.append(car_dict)
        Scene_wo_render._agent_del_dict.clear()
    
    @staticmethod
    def run_step_w_ego(cur_time_step, ego_control=False):
        time_step = 1.0/Scene_wo_render._FPS # 1s/5 = 0.2s，作为每一个时间间隔
        # 基于_debug_mode的设置进行初始化
        if Scene_wo_render._debug_mode:
            if Scene_wo_render._car_dict_sequence is None:
                Scene_wo_render._car_dict_sequence = []
            car_dict = dict()
        # 遍历所有仍然存在的agents
        for agent_name, agent in Scene_wo_render._agent_dict.items():
            if agent_name == 'ego_vehicle':
                if not ego_control:
                    next_ego_waypoint = agent.plan_path[cur_time_step]
                    next_ego_loc = next_ego_waypoint.transform.location
                    next_ego_rot = next_ego_waypoint.transform.rotation
                    agent.cur_waypoint.transform.set_rotation(next_ego_rot)
                    agent.cur_waypoint.transform.set_location(next_ego_loc)
                    agent.update_information(next_ego_waypoint.transform)
                    agent.set_speed(agent.speed)
                    if Scene_wo_render._debug_mode:
                        car_dict[agent_name] = dict()
                        car_dict[agent_name]['loc'] = next_ego_loc
                        car_dict[agent_name]['rot'] = next_ego_rot
                        car_dict[agent_name]['bbox'] = agent.bounding_box
            # 获取当前agent的控制信息
            agent_control = Scene_wo_render._agent_control_dict[agent_name]
            # 如果agent的结束标志为真，则将其添加到删除列表中
            if agent.end_route_flag:
                Scene_wo_render._agent_del_dict[agent_name] = agent
                continue
            # 暂时设置前后轴距为1.3m
            agent_f_len = 1.3 # 前轴距 unit: m
            agent_r_len = 1.3 # 后轴距 unit: m
            # agent具体控制信息的获取，包括油门、刹车、方向盘
            throttle = agent_control.throttle # throttle range:[0, 1] 最大值在local planner里被设置为了0.75 unit: None
            brake = agent_control.brake # brake range:[0, 1] 最大值再local planner里被设置为了0.3 unit: None
            steer = agent_control.steer # steer range:[-1, 1] 最大值在local planner里被设置为了0.8 unit: None
            delta = steer*40*math.pi/180 # steer range:[-1, 1] -> delta range:[-40, 40] unit: deg
            cur_agent_loc = agent.cur_waypoint.transform.location # location [x, y, z] unit：m
            cur_agent_rot = agent.cur_waypoint.transform.rotation # rotation [roll, pitch, yaw] unit: rad 
            agent_x = cur_agent_loc[0] # x_loc unit：m
            agent_y = cur_agent_loc[1] # y_loc unit：m
            agent_yaw = cur_agent_rot[2] # yaw range:[-pi, pi] unit: rad
            agent_v = agent.speed # velocity unit:km/h
            if brake >= 0.001:
                agent_a = -6.0 * brake # brake deceleration
            else:
                agent_a = 10 * throttle # acceleration
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
            # if agent_name == 'background_agent_0':
            #     print(f'agent_x_update: {agent_x_update}, agent_y_update: {agent_y_update}, agent_yaw_update: {agent_yaw_update}, agent_speed_update: {agent_speed_update}')
            # update agent's velocity and location
            agent_next_loc = [agent_x_update ,agent_y_update , 0.0] # location [x, y, z] unit：m
            agent_next_rot = [0.0, 0.0, agent_yaw_update] # rotation [roll, pitch, yaw] unit: rad
            agent.cur_waypoint.transform.set_rotation(agent_next_rot)
            agent.cur_waypoint.transform.set_location(agent_next_loc)
            updated_transform = agent.cur_waypoint.transform
            agent.update_information(updated_transform)
            agent.set_speed(agent_speed_update * 3.6) # velocity unit: km/h
            if Scene_wo_render._debug_mode:
                car_dict[agent_name] = dict()
                car_dict[agent_name]['loc'] = agent_next_loc
                car_dict[agent_name]['rot'] = agent_next_rot
                car_dict[agent_name]['bbox'] = agent.bounding_box
        for agent_name, agent in Scene_wo_render._agent_del_dict.items():
            del Scene_wo_render._agent_dict[agent_name]
            del Scene_wo_render._agent_control_dict[agent_name]
        if Scene_wo_render._debug_mode:
            Scene_wo_render._car_dict_sequence.append(car_dict)
        Scene_wo_render._agent_del_dict.clear()