from map import Map
import random
import json
if __name__ == '__main__':
    test_car_list = []
    test_car_name_list = ['car1', 'car2']
    test_driving_mode_list = ['normal', 'aggressive', 'defensive']
    test_driving_option_list = ['LaneFollow', 'ChangeLaneLeft', 'ChangeLaneRight', 'Left', 'Right', 'Straight']
    # random.seed(0)
    map_json_path = '/home/ubuntu/junhaoge/ChatSim/data/end2end_map_data/test_map.json'
    # 加载JSON数据
    with open(map_json_path, 'r') as file:
        map_data = json.load(file)
    test_map = Map(map_data)

    # 初始化test_car_list
    for i in range(len(test_car_name_list)):
        vehicle_dict = {}
        vehicle_dict['name'] = test_car_name_list[i]
        # vehicle_dict['type'] = 'car'
        # vehicle_dict['driving_mode'] = random.choice(test_driving_mode_list)
        # vehicle_dict['driving_option'] = random.choice(test_driving_option_list)
        vehicle_dict['start_loc'] = None
        vehicle_dict['end_loc'] = None
        test_car_list.append(vehicle_dict)

    # 随机选择不重复的车辆出生点
    # for i in range(len(test_car_name_list)):
    #     while True:
    #         test_location = random.choice(test_map.get_spawn_points())
    #         if test_location not in [car['start_loc'] for car in test_car_list]:
    #             break
    #     test_car_list[i]['start_loc'] = test_location
    #     min_distance = 50
    #     while True:
    #         test_location = random.choice(test_map.get_spawn_points())
    #         if test_map.calculate_distance(test_car_list[i]['start_loc'], test_location) < min_distance:
    #             continue
    #         if test_location not in [car['end_loc'] for car in test_car_list]:
    #             break
    #     test_car_list[i]['end_loc'] = test_location
    # # # plan path
    # save_path = '/home/ubuntu/junhaoge/ChatSim/data/test_map/pic_save'
    # map_png_path = '/home/ubuntu/junhaoge/ChatSim/data/test_map/test_map.png'

    test_car_list[0]['start_loc'] = test_map.features['97'].polyline[0]
    test_car_list[0]['end_loc'] = test_map.features['50'].polyline[-1]
    test_car_list[1]['start_loc'] = test_map.features['97'].polyline[20]
    test_car_list[1]['end_loc'] = test_map.features['74'].polyline[-1]
    for i in range(len(test_car_name_list)):
        test_car_list[i]['path'] = test_map.plan_path(test_car_list[i]['start_loc'], test_car_list[i]['end_loc'])

    
    min_path_len = 10000
    for i in range(len(test_car_name_list)):
        if len(test_car_list[i]['path']) < min_path_len:
            min_path_len = len(test_car_list[i]['path'])

    for i in range(len(test_car_name_list)):
        test_car_list[i]['path'] = test_car_list[i]['path'][:min_path_len]

    for i in range(min_path_len):
        car_dict = []
        for j in range(len(test_car_list)):
            car_info = {}
            # car_dict['name'] = test_car_list[j]['name']
            point = test_car_list[j]['path'][i]
            point_loc = test_map.get_location_by_lane_point(point)
            car_info['loc'] = point_loc
            # print(car_info['loc'])
            car_dict.append(car_info)
        test_map.draw_map_w_traffic_flow(car_dict)

    
