import os
import shutil
from tqdm import tqdm 

def copy_and_rename_files(src_dir, dst_dir, dir_name, prefix="day"):
    #dst_dir = os.path.join(src_dir, dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for root, dirs, files in os.walk(src_dir):
        for dir in dirs:
            if dir.startswith(prefix):
                img_dir = os.path.join(root, dir)
                for filename in os.listdir(img_dir):
                    if filename.endswith('.jpg'):
                        src_file = os.path.join(img_dir, filename)
                        dst_file = os.path.join(dst_dir, dir_name + '_' + dir + '_' + filename)
                        shutil.copy(src_file, dst_file)


if __name__ == "__main__":
    root_dir = r'D:\MPIIFaceGaze\MPIIFaceGaze'  # 根文件夹路径，例如 '/path/to/your/rootdir'
    dst_dir = 'images'  # 目标文件夹名

    # 在根文件夹下遍历所有以 'p' 开头的主文件夹
    for dir_name in tqdm(os.listdir(root_dir)):
        if dir_name.startswith('p'):
            src_dir = os.path.join(root_dir, dir_name)
            copy_and_rename_files(src_dir, dst_dir, dir_name)
