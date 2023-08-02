import math, os, time, argparse, sys
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from dataset.MPIIFaceGaze import MPIIFaceGazeData

#from models.ITrackerModel import ITrackerModel
#from models.ITrackerModel_ori import ITrackerModel
from models.ITrackerModel_ori_affine import ITrackerModel
#from models.ITrackerModelWithLandmark import ITrackerModel

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'C:\Users\Xuli\Desktop\MPIIGazeCode')  #cut images in this name plus Crop
parser.add_argument('--weight_path', default=r'results\affine2\best.pth')
parser.add_argument('--meta_file', default=r'C:\Users\Xuli\Desktop\MPIIGazeCode\all_data.json')
parser.add_argument('--batch_size', default=64)
parser.add_argument('--num_workers', default=4)
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


def test_main(model, weight_path):
    model.eval()
    model = torch.nn.DataParallel(model)
    model = load_weight(weight_path)
    model.cuda()
    cudnn.benchmark = True 
    dataTest = MPIIFaceGazeData(args.data_path, split='test', right_eye_flip=right_eye_flip, face_size=faceSize, eye_size=eyeSize, grid_size=gridSize, landmark_mask=landmark_mask, meta_file=args.meta_file, onlyRawImage=onlyRawImage)
    
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

def load_weight(weight_path):
    if weight_path is None:
        print("No weights path provided!")
        sys.exit(1)  # exit the program when no weights path is provided
    try:
        pretrained_dict = torch.load(weight_path)

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Weights {} loaded successfully.".format(os.path.basename(weight_path)))
    except FileNotFoundError:
        print("Weights file not found!")
        sys.exit(1)  # exit the program when weights file not found
    except Exception as e:
        print("Error loading pretrained weights:", e)
        sys.exit(1)  # exit the program when other error occurs
    return model


if __name__ == "__main__":
    model = ITrackerModel()
    
    test_main(model, args.weight_path)
