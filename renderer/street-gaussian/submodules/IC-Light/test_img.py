import cv2
import numpy as np
img_path = 'E:/Download/IC-Light独立整合包/IC-Light/20_0_rgb_obj_mask.png'
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
print(img.shape)
# 透明度
