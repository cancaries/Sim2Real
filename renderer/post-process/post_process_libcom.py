from PIL import Image
import numpy as np
import os
from joblib import Parallel, delayed
import cv2
from libcom import ImageHarmonizationModel

# 计算 H 值的循环差值
def circular_hue_difference(h1, h2):
    diff1 = np.abs(h1 - h2)
    diff2 = np.abs(180 - diff1)
    return np.minimum(diff1, diff2)

def get_foreground_mask(foreground_img_path,conbined_img,output_path):
    foreground_img = cv2.imread(foreground_img_path)
    size = foreground_img.shape
    # print(size)
    composition_img = cv2.imread(conbined_img)
    # 确保图像是 BGR 格式
    if len(foreground_img.shape) == 3 and foreground_img.shape[2] == 4:
        foreground_img = cv2.cvtColor(foreground_img, cv2.COLOR_BGRA2BGR)
    if len(composition_img.shape) == 3 and composition_img.shape[2] == 4:
        composition_img = cv2.cvtColor(composition_img, cv2.COLOR_BGRA2BGR)
    foreground_img_hsv = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HSV)
    composition_img_hsv = cv2.cvtColor(composition_img, cv2.COLOR_BGR2HSV)
    if np.all(foreground_img == 255):
        mask = np.zeros((foreground_img.shape[0], foreground_img.shape[1], 4), dtype=np.uint8)
        cv2.imwrite(output_path, mask)
        return
    # 使用numpy简化上述操作并且加速，初次之外纯白的背景也认为是背景
    mask = np.zeros((foreground_img.shape[0], foreground_img.shape[1], 1), dtype=np.uint8)
    # diff计算的时候考虑循环，最大值为255，最小值为0
    h_diff = circular_hue_difference(foreground_img_hsv[:, :, 0], composition_img_hsv[:, :, 0])
    s_diff = np.abs(foreground_img_hsv[:, :, 1].astype(np.int16) - composition_img_hsv[:, :, 1].astype(np.int16))
    v_diff = np.abs(foreground_img_hsv[:, :, 2].astype(np.int16) - composition_img_hsv[:, :, 2].astype(np.int16))
    region = (h_diff < 50) & (s_diff<100) &(v_diff<20)
    mask[region] = 255
    mask[np.all(foreground_img == 255, axis=2)] = 0
    cv2.imwrite(output_path, mask)

def get_and_scale_bounding_box(image_path, scale_factor=100):
    # 读取图像
    image = Image.open(image_path).convert("L")
    binary_image = image.point(lambda p: p > 127 and 255)
    
    # 转换为 NumPy 数组
    binary_array = np.array(binary_image)
    
    # 获取图像的尺寸
    image_width, image_height = image.size
    
    # 找到非零（白色）像素的坐标
    white_pixels = np.argwhere(binary_array == 255)

    # 如果没有找到白色像素，返回None
    if len(white_pixels) == 0:
        print("没有找到任何白色像素")
        return None

    # 获取最小外接矩形
    min_y, min_x = np.min(white_pixels, axis=0)
    max_y, max_x = np.max(white_pixels, axis=0)

    # 计算矩形的左上角和宽高
    x, y = min_x, min_y
    width, height = max_x - min_x, max_y - min_y

    # 计算外接矩形的中心点
    center_x = x + width / 2
    center_y = y + height / 2

    # 按比例放大矩形
    new_width = width * scale_factor
    new_height = height * scale_factor

    # 计算放大后的新左上角和右下角坐标
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)

    # 确保坐标不超出图像边界
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    # 最终的放大并限制后的边界框
    scaled_bounding_box = (new_x1, new_y1, new_x2, new_y2)
    print("放大并调整后的最小外接矩形的坐标: ", scaled_bounding_box)
    
    return scaled_bounding_box

def libcom_single_image(img_path1,img_path2,output_path): #obj_mask,bkgd,output
    input_fg = Image.open(img_path1)
    input_bg = Image.open(img_path2)
    input_fg = input_fg.convert("RGBA")
    
    bbox = get_and_scale_bounding_box(img_path1)

    input_fg_crop = input_fg.crop(bbox)
    input_fg_crop_path = img_path1.replace('_concat_rgb_obj_mask.png', '_concat_rgb_obj_mask_cropped.png')
    input_fg_crop.save(input_fg_crop_path)

    input_bg_crop = input_bg.crop(bbox)
    input_bg_crop_path = img_path1.replace('_concat_rgb.png', '_concat_rgb_cropped.png')
    input_bg_crop.save(input_bg_crop_path)

    PCTNet = ImageHarmonizationModel(device=0, model_type='PCTNet')
    PCT_result1 = PCTNet(input_bg_crop_path, input_fg_crop_path)

    # 确保类型和范围
    PCT_result1 = np.clip(PCT_result1, 0, 255).astype(np.uint8)

    # BGR 转 RGB
    PCT_result1_rgb = PCT_result1[:, :, ::-1]

    # 转换为 PIL 图像
    new_foreground = Image.fromarray(PCT_result1_rgb)
    new_foreground = new_foreground.convert(input_bg.mode)

    # 将 new_foreground 粘贴到原始图像中
    original_rgb = input_bg.copy()
    original_rgb.paste(new_foreground, (bbox[0], bbox[1]))

    # 转换为 NumPy 数组并调整颜色格式
    original_rgb_np = np.array(original_rgb)
    original_rgb_bgr = original_rgb_np[:, :, ::-1]  # 转换为 BGR 格式

    cv2.imwrite(output_path, original_rgb_bgr)
    # net = ShadowGenerationModel(device=0, model_type='ShadowGeneration')
    # preds = net(output_path, img_path1, number=1)
    # cv2.imwrite(output_path.replace('relight','shadow'), preds[0])
    # 删除model释放显存
    del PCTNet
    # del net

if __name__ == '__main__':
    import sys
    folder = sys.argv[1]
    frame = int(sys.argv[2])
    #concat obj
    if os.path.exists(os.path.join(folder,str(frame)+'_concat_rgb_obj.png')):
        pass
    else:
        image_concat_list = []
        for cam_id in [1,0,2]:
            img = cv2.imread(os.path.join(folder,str(frame)+'_'+str(cam_id)+'_rgb_obj.png'),cv2.IMREAD_UNCHANGED)
            image_concat_list.append(img)
        concat_img = np.concatenate(image_concat_list, axis=1)
        cv2.imwrite(os.path.join(folder,str(frame)+'_concat_rgb_obj.png'),concat_img)

    #concat rgb
    if os.path.exists(os.path.join(folder,str(frame)+'_concat_rgb.png')):
        pass
    else:
        image_concat_list = []
        for cam_id in [1,0,2]:
            img = cv2.imread(os.path.join(folder,str(frame)+'_'+str(cam_id)+'_rgb.png'))
            image_concat_list.append(img)
        concat_img = np.concatenate(image_concat_list, axis=1)
        cv2.imwrite(os.path.join(folder,str(frame)+'_concat_rgb.png'),concat_img)

    #get mask
    if os.path.exists(os.path.join(folder,str(frame)+'_concat_rgb_obj_mask.png')):
        pass
    else:
        get_foreground_mask(os.path.join(folder,str(frame)+'_concat_rgb_obj.png'),\
                            os.path.join(folder,str(frame)+'_concat_rgb.png'),\
                                os.path.join(folder,str(frame)+'_concat_rgb_obj_mask.png'))

    #libcom
    Parallel(n_jobs=1)(delayed(libcom_single_image)(os.path.join(folder,str(frame)+'_concat_rgb_obj_mask.png'),\
                                                        os.path.join(folder,str(frame)+'_concat_rgb.png'),\
                                                        os.path.join(folder,str(frame)+'_rgb_iclight_libcom.png')))