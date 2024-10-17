from enum import Enum
from process_fbc import process_relight2,change_state2
import cv2
class BGSource_1(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

class BGSource_2(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"

input_fg2 = cv2.imread('/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/submodules/IC-Light/20_0_rgb_obj_mask.png', cv2.IMREAD_UNCHANGED)
input_bg2 = cv2.imread('/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/submodules/IC-Light/20_0_rgb_bkgd.png', cv2.IMREAD_UNCHANGED)
# 将两张图片的width和height都保证为8的倍数
input_fg2 = cv2.resize(input_fg2, (1600, 1064), interpolation=cv2.INTER_NEAREST)
input_bg2 = cv2.resize(input_bg2, (1600, 1064), interpolation=cv2.INTER_NEAREST)
prompt = ''
image_width = 1600
image_height = 1064
num_samples = 1
seed = 12345
steps = 25
a_prompt = 'best quality'
n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
cfg = 2
highres_scale = 1.5
highres_denoise = 0.5
bg2_source = BGSource_2.UPLOAD.value
ips2 = [input_fg2, input_bg2, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg2_source]
change_state2(True)
img_output = process_relight2(*ips2)
cv2.imwrite('/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/submodules/IC-Light/20_0_rgb_obj_mask_relight.png', img_output)