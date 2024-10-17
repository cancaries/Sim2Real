import os
import sys
import numpy
import cv2
import numpy as np

def get_foreground_mask(foreground_img_path,conbined_img,output_path):
    foreground_img = cv2.imread(foreground_img_path)
    composition_img = cv2.imread(conbined_img)
    # mask = numpy.zeros((foreground_img.shape[0], foreground_img.shape[1], 4), dtype=numpy.uint8)
    # foreground_img与composition_img值完全相等的像素点，认为是最终的前景，其余的像素点认为是背景
    # 将foreground_img的像素值转换为hsv
    foreground_img_hsv = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HSV)
    composition_img_hsv = cv2.cvtColor(composition_img, cv2.COLOR_BGR2HSV)
    # for i in range(foreground_img.shape[0]):
    #     for j in range(foreground_img.shape[1]):
    #         # 如果前景图像与合成图像的像素值基本相等，认为是前景
    #         if np.linalg.norm(foreground_img_hsv[i][j] - composition_img_hsv[i][j]) < 10:
    #             # print(foreground_img[i][j])
    #             # 将原本的图像拓展为4通道，第四个通道为透明度通道
    #             new_foreground = numpy.zeros(4, dtype=numpy.uint8)
    #             new_foreground[0] = foreground_img[i][j][0]
    #             new_foreground[1] = foreground_img[i][j][1]
    #             new_foreground[2] = foreground_img[i][j][2]
    #             new_foreground[3] = 255
    #             mask[i][j] = new_foreground
    #         else:
    #             # 设施为透明的像素点
    #             mask[i][j] = [0, 0, 0, 0]
    # 如果foreground是纯白，则直接返回空的mask
    if np.all(foreground_img == 255):
        mask = np.zeros((foreground_img.shape[0], foreground_img.shape[1], 4), dtype=np.uint8)
        cv2.imwrite(output_path, mask)
        return
    # 使用numpy简化上述操作并且加速，初次之外纯白的背景也认为是背景
    mask = np.zeros((foreground_img.shape[0], foreground_img.shape[1], 4), dtype=np.uint8)
    mask[np.linalg.norm(foreground_img_hsv - composition_img_hsv, axis=2) < 10] = np.concatenate((foreground_img, np.ones((foreground_img.shape[0], foreground_img.shape[1], 1), dtype=np.uint8) * 255), axis=2)[np.linalg.norm(foreground_img_hsv - composition_img_hsv, axis=2) < 10]
    mask[np.linalg.norm(foreground_img_hsv - composition_img_hsv, axis=2) >= 10] = [0, 0, 0, 0]
    mask[np.all(foreground_img == 255, axis=2)] = [0, 0, 0, 0]
    # print('save mask to: ', output_path)
    # print('mask shape: ', mask.shape)
    cv2.imwrite(output_path, mask)

def get_foreground_mask_in_folder(folder):
    frame_list = [x.split('_')[0] for x in os.listdir(folder) if x.endswith('.png')]
    cam_id_list = [x.split('_')[1] for x in os.listdir(folder) if x.endswith('.png')]
    # 删除重复值
    frame_list = list(set(frame_list))
    cam_id_list = list(set(cam_id_list))
    # for frame in frame_list:
    #     for cam_id in cam_id_list:
    #         print('Processing frame: ', frame, ' camera: ', cam_id)
    #         foreground_img_path = os.path.join(folder, frame + '_' + cam_id + '_rgb_obj.png')
    #         conbined_img = os.path.join(folder, frame + '_' + cam_id + '_rgb.png')
    #         output_path = os.path.join(folder, frame + '_' + cam_id + '_rgb_obj_mask.png')
    #         get_foreground_mask(foreground_img_path, conbined_img, output_path)
    from joblib import Parallel, delayed
    Parallel(n_jobs=4)(delayed(get_foreground_mask)(os.path.join(folder, frame + '_' + cam_id + '_rgb_obj.png'), os.path.join(folder, frame + '_' + cam_id + '_rgb.png'), os.path.join(folder, frame + '_' + cam_id + '_rgb_obj_mask.png')) for frame in frame_list for cam_id in cam_id_list)


if __name__ == '__main__':
    folder_path = '/home/ubuntu/junhaoge/real_world_simulation/renderer/street-gaussian/output/waymo_full_exp/waymo_train_019/trajectory/019_0'
    get_foreground_mask_in_folder(folder_path)

