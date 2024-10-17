import os
import sys

if __name__ == '__main__':
    dataset_path = '/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp'
    delete_name_list = [96,382,402,427]
    for name in delete_name_list:
        scene_path = os.path.join(dataset_path, f'waymo_train_{str(name).zfill(3)}')
        trajectory_path = os.path.join(scene_path, 'trajectory')
        for trajectory in os.listdir(trajectory_path):
            trajectory = os.path.join(trajectory_path, trajectory)
            if '50000' or '100000' in trajectory:
                os.system(f'rm -r {trajectory}')