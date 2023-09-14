import os
import cv2
from scipy.io import loadmat
import json
import random

def read_screen_mat(screen_path):
    # 换算到屏幕的cm单位上
    mat_data = loadmat(screen_path)
    height_ratio = float(mat_data['height_mm'][0][0])/int(mat_data['height_pixel'][0][0])*0.1  
    width_ratio = float(mat_data['width_mm'][0][0])/int(mat_data['width_pixel'][0][0])*0.1
    return width_ratio, height_ratio

def process_txt_and_images(src_dir, label, prefix="p"):
    for dir_name in os.listdir(src_dir):
        if dir_name.startswith(prefix):
            txt_file = os.path.join(src_dir, dir_name, dir_name + '.txt')
            screen_mat = os.path.join(src_dir, dir_name, 'Calibration/screenSize.mat')
            width_ratio, height_ratio=read_screen_mat(screen_mat)
            
            if os.path.isfile(txt_file):
                with open(txt_file, 'r') as file:
                    for line in file:
                        info1 = line.split()
                        imgname = info1[0].replace('/', '_')  # 将 '/' 替换为 '_'
                        imgname = dir_name + '_' + imgname  # 加上以 'p' 开头的文件名
                        gaze_point_pixel = [int(info1[1]), int(info1[2])]
                        gaze_point_cm = [round(int(info1[1])*width_ratio, 5), round(int(info1[2])*height_ratio, 5)]
                        label.append([imgname, gaze_point_pixel, gaze_point_cm])


if __name__ == "__main__":
    root_dir = r'/home/snowwhite/eye_tracking/MPIIFaceGaze' # 根文件夹路径，例如 '/path/to/your/rootdir'
    label = []
    process_txt_and_images(root_dir, label)
    with open('data.json', 'w') as json_file:
        json.dump(label, json_file)
    split_label = {"Train":[], "Val":[], "Test":[]}
    #random.shuffle(label)  # 打乱label列表

    # 根据8:1:1的比例划分数据集
    num_train = int(len(label) * (11/15))
    num_val = int(len(label) * (2/15))

    split_label["Train"] = label[:num_train]
    split_label["Val"] = label[num_train:num_train+num_val]
    split_label["Test"] = label[num_train+num_val:]

    # 将划分好的数据集写入到JSON文件中
    with open('all_data_person_specific.json', 'w') as json_file:
        json.dump(split_label, json_file)
    print(len(label))
    print(len(split_label["Train"]))
    print(len(split_label["Val"]))
    print(len(split_label["Test"]))
    
    print("Finished")
    