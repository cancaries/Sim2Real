import subprocess
import os

input_path = '/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/test/036_0'
frame_list = [int(x.split('_')[0]) for x in os.listdir(input_path) if x.endswith('.png') and 'concat' not in x and 'relight' not in x]
frame_list = list(set(frame_list))
frame_list.sort()

for frame in frame_list:
    # 使用 iclight 环境运行 iclight_process.py
    subprocess.run(["conda", "run", "-n", "iclight", "python", "/home/ubuntu/junhaoge/real_world_simulation/data_utils/post_process_iclight.py", input_path, str(frame)])
    
    # 使用 Libcom 环境运行 libcom_process.py
    # subprocess.run(["conda", "run", "-n", "Libcom", "python", "/home/ubuntu/junhaoge/real_world_simulation/data_utils/post_process_libcom.py", input_path, str(frame)])
