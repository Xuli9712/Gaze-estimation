import math, os, time, argparse, sys
from tqdm import tqdm 
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np

from dataset.BRL4200Gaze import BRL4200GazeData

#from models.ITrackerModel import ITrackerModel
#from models.ITrackerModelWithLandmark import ITrackerModel

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='UnetEncoder') 
parser.add_argument('--weight_path', default=r'results/BRL4200-UnetEncoderfinetune/best.pth')
parser.add_argument('--batch_size', default=1)
parser.add_argument('--num_workers', default=8)
args = parser.parse_args()  

workers = args.num_workers
batch_size = args.batch_size

faceSize=(224,224)
eyeSize=(112, 112)
gridSize=(25,25)
landmark_mask = False
onlyRawImage = False  #using only the raw image, no croping
right_eye_flip = True
# 若使用原始grid作为输入，则使用参数 gridSize=(25, 25), landmark_mask=False
# faceSize和eyeSize均可调整


def main(model):

    if args.weight_path is None:
        print("No weights path provided!")
        sys.exit(1)  # exit the program when no weights path is provided

    try:
        pretrained_dict = torch.load(args.weight_path)

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        

        print("Weights {} loaded successfully.".format(os.path.basename(args.weight_path)))
    except FileNotFoundError:
        print("Weights file not found!")
        sys.exit(1)  # exit the program when weights file not found
    except Exception as e:
        print("Error loading pretrained weights:", e)
        sys.exit(1)  # exit the program when other error occurs
    model.cuda()
    cudnn.benchmark = True 
    
    dataTest = BRL4200GazeData(args.data_path, split='test', right_eye_flip=right_eye_flip, face_size=faceSize, eye_size=eyeSize, grid_size=gridSize, landmark_mask=landmark_mask, meta_file=args.meta_file, onlyRawImage=onlyRawImage)
    

    test_loader = torch.utils.data.DataLoader(
        dataTest,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
   
    criterion = nn.SmoothL1Loss().cuda()
    test(model, test_loader, criterion)

def test(model, test_loader, criterion):
    if len(test_loader) == 0:
        print("NO TEST SET!")
        return 0, 0
    elif onlyRawImage == False:
        test_loss=0
        inference_times=[]
        test_error = 0
        gaze_list = []
        with tqdm(total=len(test_loader), desc="Test") as vbar:
            for i, (Img, Face, LEye, REye, Grid, GazePoint, GazePointPixel) in enumerate(test_loader):
                Face = Face.cuda()
                LEye = LEye.cuda()
                REye = REye.cuda()
                Grid = Grid.cuda()
                GazePoint = GazePoint.cuda()
                with torch.no_grad():
                    start_time = time.time() 
                    pred_gaze = model(Face, LEye, REye, Grid)
                    
                    loss = criterion(pred_gaze, GazePoint)
                    end_time = time.time() 
                    inference_times.append(end_time - start_time) 
                    error = torch.norm(pred_gaze-GazePoint, dim=1).mean()

                    test_loss += loss.item()
                    test_error += error.item()

                    vbar.set_postfix(**{'loss per batch': loss.item()})
                    vbar.update()
                    pred_gaze = pred_gaze.squeeze().cpu().numpy()
                    gaze_list.append(pred_gaze)

            visulaize_result_distribution(gaze_list)
            mean_test_loss = round(test_loss / len(test_loader), 5)
            mean_test_error = round(test_error/ (len(test_loader)), 5)
            mean_inference_time = round(sum(inference_times) / len(inference_times), 5)
            mean_inference_time_per_second = round(mean_inference_time/int(batch_size), 5)
        print("Test MSE: ", mean_test_loss, "Test Error:", mean_test_error, "Inference time(s) per batch({}): ".format(batch_size), mean_inference_time, "Mean Inference time(s) per image", mean_inference_time_per_second)
        return mean_test_loss, mean_test_error
    elif onlyRawImage == True:
        test_loss=0
        inference_times=[]
        test_error = 0
        with tqdm(total=len(test_loader), desc="Test") as vbar:
            for i, (Img, Face, LEye, REye, Grid, GazePoint) in enumerate(test_loader):
                Img = Img.cuda()
                GazePoint = GazePoint.cuda()
                with torch.no_grad():
                    start_time = time.time() 
                    pred_gaze = model(Img)
                    loss = criterion(pred_gaze, GazePoint)
                    end_time = time.time() 
                    error = torch.norm(pred_gaze-GazePoint, dim=1).mean()
                    inference_times.append(end_time - start_time) 

                    test_loss += loss.item()

                    vbar.set_postfix(**{'loss per batch': loss.item()})
                    vbar.update()

            mean_test_loss = round(test_loss / len(test_loader), 5)
            mean_test_error = round(test_error/ (len(test_loader)), 5)
            mean_inference_time = round(sum(inference_times) / len(inference_times), 5)
            mean_inference_time_per_second = round(mean_inference_time/int(batch_size), 5)
        print("Test Loss: ", mean_test_loss, "Test Error: ",mean_test_error, "Mean inference time(s) per batch({}): ".format(batch_size), mean_inference_time, "Mean inference time(s) per image", mean_inference_time_per_second)
        return mean_test_loss, mean_test_error
    
def visulaize_result_distribution(predicted_gazes):
    #background_path = r'tools/points_visualization_3.png'
    img = np.ones((540, 960))

    for gaze in predicted_gazes:
        gaze_x = int(gaze[0] * (1920 / 34.4))
        gaze_y = int(gaze[1] * (1080 / 19.35))
        img = cv2.circle(img, (gaze_x, gaze_y), 3, (0, 0, 255), -1)  # Red color in BGR format

    cv2.imwrite('visualize_result_train.png', img)  # Overwrite the original image with the modified one


        
if __name__ == "__main__":
    if args.model == 'ITrackerOriAffine':
        from models.ITrackerModel_ori_affine import ITrackerModel
    elif args.model == 'ITrackerOri':
        from models.ITrackerModel_ori import ITrackerModel
    elif args.model == 'UnetEncoder':
        from models.UNetITrackerModel import ITrackerModel
    elif args.model == 'UnetEncoderAffine':
        from models.UNetITrackerModel_affine import ITrackerModel
    model = ITrackerModel()
    model.eval()
    model = torch.nn.DataParallel(model)
    main(model)
