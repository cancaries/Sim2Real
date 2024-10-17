from enum import Enum
from process_fbc_custom_weather import process_relight2
from PIL import Image
import cv2
import numpy as np
import os
from joblib import Parallel, delayed

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

def iclight_single_image(input, output):
    input_img = Image.open(input)
    size = input_img.size
    img_ori_height = size[1]
    img_ori_width = size[0]

    image_height = img_ori_height - 8 + img_ori_height % 8
    image_width = img_ori_width - 8 + img_ori_width % 8

    input_img.resize((image_width,image_height))
    input_img2 = np.array(input_img)
    num_samples = 1
    seed = 0
    steps = 10
    prompt = 'daylight,blue sky,vehicle,city,vehicle shadow'
    a_prompt = 'best quality,vehicle shadow'
    n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
    cfg = 3
    highres_scale = 1.5
    highres_denoise = 0.75
    lowres_denoise = 0.5
    bg2_source = BGSource_2.UPLOAD.value
    ips2 = [input_img2, None, None, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg2_source]
    output_img = process_relight2(*ips2)
    new_img = Image.fromarray(output_img[0])
    new_img.save(output)

if __name__ == '__main__':
    import sys
    folder = sys.argv[1]
    frame = int(sys.argv[2])
    # iclight
    Parallel(n_jobs=1)([delayed(iclight_single_image)(
    os.path.join(folder, f"{frame}_{cam_id}_rgb.png"),
    os.path.join(folder, f"{frame}_{cam_id}_rgb_iclight.png")
) for cam_id in range(3)])