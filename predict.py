import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from models.UNetITrackerModel import ITrackerModel
import torchvision.transforms as transforms
from tools.mediapipe_eye_detector import MPFacialLandmarkDetector
from PIL import Image
import argparse, sys, os
import torch.backends.cudnn as cudnn
import time

parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', default=r'results/BRL4200-UnetEncoderfinetune/best.pth')
parser.add_argument('--video_path', default=r'/home/snowwhite/eye_tracking/test_videos/GX011069.MP4')
args = parser.parse_args()

def visulaize_result_distribution(predicted_gazes):
    img = np.ones((540, 960, 3), np.float32)*255

    for gaze in predicted_gazes:
        gaze_x = int(gaze[0] * (1920 / 34.4))
        gaze_y = int(gaze[1] * (1080 / 19.35))
        img = cv2.circle(img, (gaze_x, gaze_y), 3, (0, 0, 255), -1)  # Red color in BGR format

    cv2.imwrite('visualize_video1.png', img)  # Overwrite the original image with the modified one


class Preprocessor:
    def __init__(self, face_size, eye_size, grid_size, landmark_detector, right_eye_flip=True, use_landmark_mask=False, onlyRawImage=False):
        self.face_size = face_size
        self.eye_size = eye_size
        self.grid_size = grid_size
        self.landmark_detector = landmark_detector
        self.right_eye_flip = right_eye_flip
        self.use_landmark_mask = use_landmark_mask
        self.onlyRawImage = onlyRawImage
        self.landmark = None
        self.transformFace = transforms.Compose([
            transforms.Resize(self.face_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.378, 0.252, 0.193], std=[0.207, 0.142, 0.118])
        ])
        self.transformLEye = transforms.Compose([
            transforms.Resize(self.eye_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.433, 0.263, 0.189], std=[0.133, 0.0939, 0.0739])
        ])

        self.transformGrid = transforms.Compose([
            transforms.Resize(self.grid_size),
            transforms.ToTensor(),
        ])

        
        self.transformREye = transforms.Compose([
            transforms.Resize(self.eye_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.447, 0.274, 0.196], std=[0.138, 0.0997, 0.0783])
            ])

    def preprocess_frame_to_tensor(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_show = img.copy()
            h, w = img.shape[:2]
            if onlyRawImage == False:
                landmarks = self.landmark_detector.get_landmarks(img)

                if landmarks is not None:
                    for landmark in landmarks:
                        self.landmark = landmark
                        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157]
                        right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]
                        

                        left_eye_points = [landmark[i] for i in left_eye_indices]
                        right_eye_points = [landmark[i] for i in right_eye_indices]

                        #face = landmark_detector.get_square_roi(img, landmark, scale=1.2)
                        face, original_grid, face_bbox = landmark_detector.get_square_roi_and_grid(img, landmark, scale=1.2)
                        left_eye, left_eye_bbox = landmark_detector.get_square_roi(img, left_eye_points, scale=1.5)
                        right_eye, right_eye_bbox = landmark_detector.get_square_roi(img, right_eye_points, scale=1.5)

                        
                        Leye_img = Image.fromarray(left_eye)
                        Reye_img = Image.fromarray(right_eye)
                        face_img = Image.fromarray(face)
                        original_grid = Image.fromarray(original_grid)
                        if self.right_eye_flip == True:
                            Reye_img = Reye_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        else:
                            Reye_img = Reye_img
                        
                        Face = self.transformFace(face_img).unsqueeze(0)
                        Leye = self.transformLEye(Leye_img).unsqueeze(0)
                        Reye = self.transformREye(Reye_img).unsqueeze(0)
                        img = self.transformFace(Image.fromarray(img)).unsqueeze(0)
                        Grid = self.transformGrid(original_grid).unsqueeze(0)
                        
                        
                return img_show, Face, Leye, Reye, Grid
            elif onlyRawImage == True:
                img = self.transformFace(Image.fromarray(img)).unsqueeze(0)
                return img
        except Exception as e:
             print("Preprocess Error: ", e)
                    
face_size=(224,224)
eye_size=(112, 112)
grid_size=(25, 25)
use_landmark_mask = False
onlyRawImage = False  
right_eye_flip = True

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# facail landmark 检测器
landmark_detector = MPFacialLandmarkDetector()
# 预处理
preprocessor = Preprocessor(face_size, eye_size, grid_size, landmark_detector, right_eye_flip, use_landmark_mask, onlyRawImage)
# 设置模型类型
model = ITrackerModel()
model.eval()
model = torch.nn.DataParallel(model)
# 加载模型权重
if args.weight_path is None:
        print("No weights path provided!")
try:
    pretrained_dict = torch.load(args.weight_path)

    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("Weights {} loaded successfully.".format(os.path.basename(args.weight_path)))
except FileNotFoundError:
    print("Weights file not found!")
except Exception as e:
    print("Error loading pretrained weights:", e)


model.cuda()
cudnn.benchmark = True  

# 使用OpenCV打开摄像头
cap = cv2.VideoCapture(args.video_path)
gaze_points = []
start_frame = 170
# 设定显示窗口
plt.ion()  # 打开交互模式
fig = plt.figure()
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_counter =0
skip_frame = 10
# 循环获取摄像头的画面并进行推理
image_counter = 0
while True:
    # 读取一帧图像
    ret, frame = cap.read()

    if not ret:
        break

    frame_counter +=1
    if frame_counter % skip_frame !=0:
        continue

    start = time.time()
    # 将图像转化为模型需要的输入格式，比如转为Tensor，做归一化等操作
    if onlyRawImage == False:
        img, face, Leye, Reye, Grid = preprocessor.preprocess_frame_to_tensor(frame)
        # 使用模型进行推理
        with torch.no_grad():
            face = face.to(device)  # 将数据移动到和模型相同的设备上
            Leye = Leye.to(device)
            Reye = Reye.to(device)
            Grid = Grid.to(device)
            output = model(face, Leye, Reye, Grid)

    elif onlyRawImage == True:
        img = preprocessor.preprocess_frame_to_tensor(frame)
        with torch.no_grad():
            output = model(face, Leye, Reye, Grid)
    
    #for (x, y) in preprocessor.landmark:
        #cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1) 
    gaze_point = output.squeeze().cpu().numpy()
    print("Process Time(s) Per Image", time.time()-start)
    gaze_points.append(gaze_point)
    print(gaze_point)
    visulaize_result_distribution(gaze_points)

    #cv2.imshow('Original Image', frame)

    plt.clf()
    plt.plot(*zip(*gaze_points), marker='o', markersize=6, linewidth=1, color='blue', markerfacecolor='red', markeredgecolor='red')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.title('Screen Coordinate')
    plt.gca().invert_yaxis()  # invert y-axis to change origin to top-left
    
    # Save the figure
    plt.savefig(f'plot/gaze_distribution_{image_counter}.png', dpi=500)
    image_counter += 1


    plt.pause(0.01)
    cv2.waitKey(int(10))

cap.release()
cv2.destroyAllWindows()
plt.close(fig)

