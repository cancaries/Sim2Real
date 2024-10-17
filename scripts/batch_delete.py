import os
import sys

path_str = """
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_0_rgb_bkgd.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_0_rgb_obj.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_0_rgb.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_1_depth.npy
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_1_rgb_bkgd.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_1_rgb_obj_mask.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_1_rgb_obj.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_1_rgb.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_3_depth.npy
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_3_rgb_bkgd.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_3_rgb_obj.png
/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_036/trajectory/036_0/65_3_rgb.png
"""
for path in path_str.split('\n'):
    try:
        if path:
            os.remove(path)
    except Exception as e:
        print(e)
        pass