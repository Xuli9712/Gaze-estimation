import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image 
import json
import cv2
from tqdm import tqdm 
            
class MPIIFaceGazeData(Dataset):
    def __init__(self, dataset_path, split, right_eye_flip, face_size, eye_size, grid_size, landmark_mask, meta_file, onlyRawImage):
        self.dataset_path = dataset_path
        self.split=split
        self.face_size = face_size
        self.eye_size = eye_size
        self.grid_size = grid_size
        self.right_eye_flip = right_eye_flip
        # 选择使用lanmark_mask还是original grid做输入
        self.landmark_mask = landmark_mask
        self.meta_file = meta_file
        self.onlyRawImage = onlyRawImage
        
        if self.split == 'train':
            with open(os.path.join(dataset_path, meta_file), 'r') as f:
                self.meta = json.load(f)
            self.indices = list(range(len(self.meta["Train"])))
            self.meta = self.meta["Train"]
        if self.split == 'val':
            with open(os.path.join(dataset_path, meta_file), 'r') as f:
                self.meta = json.load(f)
            self.indices = list(range(len(self.meta["Val"])))
            self.meta = self.meta["Val"]
        if self.split == 'test':
            with open(os.path.join(dataset_path, meta_file), 'r') as f:
                self.meta = json.load(f)
            self.indices = list(range(len(self.meta["Test"])))
            self.meta = self.meta["Test"]

        self.transformFace = transforms.Compose([
            transforms.Resize(self.face_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.462, 0.350, 0.322], std=[0.270, 0.239, 0.232])
        ])
        self.transformLEye = transforms.Compose([
            transforms.Resize(self.eye_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.441, 0.306, 0.278], std=[0.243, 0.205, 0.194])
        ])

        self.transformGrid = transforms.Compose([
            transforms.Resize(self.grid_size),
            transforms.ToTensor(),
        ])

        
        self.transformREye = transforms.Compose([
            transforms.Resize(self.eye_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.413, 0.278, 0.253], std=[0.238, 0.196, 0.185])
            ])
        # Test all data items and remove invalid ones

    # load 3-channel image
    def load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError:
            raise RuntimeError('Read Image Error' + image_path)
        return image
    
    def load_one_channel_image(self, image_path):
        try:
            image = Image.open(image_path)
        except OSError:
            raise RuntimeError('Read Image Error' + image_path)
        return image
    
    def convert_RGB(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_image(self, index):
        img_name = self.meta[index][0]
        image_path = os.path.join(self.dataset_path, "images/{}".format(img_name))
        face_path = os.path.join(self.dataset_path, "Face/{}".format(img_name))
        Leye_path = os.path.join(self.dataset_path, "LeftEye/{}".format(img_name))
        Reye_path = os.path.join(self.dataset_path, "RightEye/{}".format(img_name))
        original_grid_path = os.path.join(self.dataset_path, "Original_Grid/{}".format(img_name))
        landmark_mask_path = os.path.join(self.dataset_path, "LandMarkMask/{}".format(img_name))
        img = Image.open(image_path)
        face = Image.open(face_path)
        Leye = Image.open(Leye_path)
        Reye = Image.open(Reye_path)
        if self.right_eye_flip == True:
            Reye = Reye.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            Reye = Reye

        if self.landmark_mask == True:
            Grid = Image.open(landmark_mask_path)
        else:
            Grid = Image.open(original_grid_path)
        GazePointCM = torch.FloatTensor(np.array(self.meta[index][2]))
        GazePointPixel = torch.FloatTensor(np.array(self.meta[index][1]))
        return img, Leye, Reye, face, Grid, GazePointCM, GazePointPixel
    
    def get_only_raw_image(self, index):
        img_file = self.meta[index][0]
        image_path = os.path.join(self.dataset_path, "images/{}".format(img_file))
        img = Image.open(image_path)
        GazePoint = torch.FloatTensor(np.array(self.meta[index][2]))
        return img, GazePoint
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        try:
            if self.onlyRawImage == True:
                index = self.indices[index]
                img, GazePoint = self.get_only_raw_image(index)
                return img, GazePoint
            else:
                index = self.indices[index]
                # image in Tensor (C*H*W) 
                img, Leye, Reye, face, Grid, GazePointCM, GazePointPixel = self.get_image(index)
                face = self.transformFace(face)
                Leye = self.transformLEye(Leye)
                Reye = self.transformREye(Reye)
                img = self.transformFace(img)
                Grid = self.transformGrid(Grid)

                return img, face, Leye, Reye, Grid, GazePointCM, GazePointPixel
        except Exception as e:
            if self.onlyRawImage == True:
                index = self.indices[index-1]
                img, GazePoint = self.get_only_raw_image(index)
                return img, GazePoint
            else:
                index = self.indices[index-1]
                # image in Tensor (C*H*W) 
                img, Leye, Reye, face, Grid, GazePointCM, GazePointPixel = self.get_image(index)
                face = self.transformFace(face)
                Leye = self.transformLEye(Leye)
                Reye = self.transformREye(Reye)
                img = self.transformFace(img)
                Grid = self.transformGrid(Grid)

        return img, face, Leye, Reye, Grid, GazePointCM, GazePointPixel

if __name__ == '__main__':
    dataset_path = r'C:\Users\Xuli\Desktop\MPIIGazeCode'
    meta_file = r'C:\Users\Xuli\Desktop\MPIIGazeCode\all_data.json'
    onlyRawImage=False
    train_dataset = MPIIFaceGazeData(dataset_path, split='train',right_eye_flip=True, face_size=(224,224), eye_size=(112, 112), grid_size=(25,25), landmark_mask=False, meta_file=meta_file, onlyRawImage=False)
    print(len(train_dataset))
    if onlyRawImage == False:
        Img, Face, LEye, REye, Grid, GazePointCM, GazePointPixel = train_dataset[60]
        print(Face.shape)
        print(LEye.shape)
        print(Grid.shape)
        print(GazePointCM)
        print(GazePointPixel)
        fig, axs = plt.subplots(1, 5)


        axs[0].imshow(Face.permute(1, 2, 0))
        axs[0].axis('off')


        axs[1].imshow(LEye.permute(1, 2, 0))
        axs[1].axis('off')

        axs[2].imshow(REye.permute(1, 2, 0))
        axs[2].axis('off')


        axs[3].imshow(Grid.permute(1, 2, 0))
        axs[3].axis('off')

        axs[4].imshow(Img.permute(1, 2, 0))
        axs[4].axis('off')

        plt.subplots_adjust(wspace=0.1)

        plt.show()
    elif onlyRawImage == True:
        Img, Grid, GazePoint = train_dataset[60]
        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(Grid.permute(1, 2, 0))
        axs[0].axis('off')

        axs[1].imshow(Img.permute(1, 2, 0))
        axs[1].axis('off')

        plt.subplots_adjust(wspace=0.1)

        plt.show()

