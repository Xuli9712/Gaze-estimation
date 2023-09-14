import math, shutil, os, time, argparse, sys
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv

#from dataset.MPIIFaceGaze import MPIIFaceGazeData
from dataset.BRL4200Gaze import BRL4200GazeData

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',default=r'/home/snowwhite/eye_tracking/brlgaze4200')
parser.add_argument('--meta_file', default=r'/home/snowwhite/eye_tracking/brlgaze4200/brl4200_split_data.json') 
parser.add_argument('--model', default='UnetEncoderAffine')
parser.add_argument('--pretrained_path', default='results/UnetEncoderfromscratch/best.pth')
parser.add_argument('--epochs', default=30)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--num_workers', default=16)
parser.add_argument('--save_per_epoch', default=1)
parser.add_argument('--result_name', default="BRL4200-UnetEncoder-fc")

args = parser.parse_args()

save_cp_and_plot = True
save_per_epoch = args.save_per_epoch

workers = args.num_workers
epochs = args.epochs
batch_size = args.batch_size

faceSize=(224,224)
eyeSize=(112, 112)
gridSize=(25,25)
landmark_mask = False
onlyRawImage = False  #using only the raw image, no croping
right_eye_flip = True
# 若使用原始grid作为输入，则使用参数 gridSize=(25, 25), landmark_mask=False
# faceSize和eyeSize均可调整
# train settings
lr = 0.001  
momentum = 0.9  
weight_decay = 1e-4  
lr_decay_milestones = []  
lr_decay_gamma = 0.1  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(model):
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    print(f"Number of parameters to update when training from scratch: {count_trainable_params(model)}")
    dataTrain = BRL4200GazeData(args.data_path, split='train', right_eye_flip=right_eye_flip, face_size=faceSize, eye_size=eyeSize, grid_size=gridSize, landmark_mask=landmark_mask, meta_file=args.meta_file, onlyRawImage=onlyRawImage)
    dataVal = BRL4200GazeData(args.data_path, split='val', right_eye_flip=right_eye_flip, face_size=faceSize, eye_size=eyeSize, grid_size=gridSize, landmark_mask=landmark_mask, meta_file=args.meta_file, onlyRawImage=onlyRawImage)
    train(model, dataTrain, dataVal)


def train(model, dataTrain, dataVal):
    start_time = datetime.now().strftime("%m%d_%H%M")
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, prefetch_factor=2)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, prefetch_factor=2)
    print('Sucessfully load trainset:', len(train_loader))
    print('Sucessfully load valset:', len(val_loader))

    if args.pretrained_path:
        try:
            pretrained_dict = torch.load(args.pretrained_path)

            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            print("Pretrained weights {} loaded successfully.".format(os.path.basename(args.pretrained_path)))
        except FileNotFoundError:
            print("Pretrained weights file not found! Training from scratch.")
        except Exception as e:
            print("Error loading pretrained weights:", e)
            print("Training from scratch.")
    
    # 冻结所有层
    for param in model.module.parameters():
        param.requires_grad = False

    # 重新启用 AffineLayer 的梯度
    for param in model.module.affine_layer.parameters():
        param.requires_grad = True

    # 重新启用 fc 的梯度
    for param in model.module.fc.parameters():
        param.requires_grad = False

    print(f"Number of parameters to update when frozen most layers: {count_trainable_params(model)}")
    
    #criterion = nn.MSELoss().cuda()
    criterion = nn.SmoothL1Loss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr,
                                weight_decay=weight_decay)
    
    # 只对affine_layer进行优化
    #optimizer = torch.optim.Adam(model.module.affine_layer.parameters(), lr, weight_decay=weight_decay)
    
    # 对fc 和 affine——layer优化
    optimizer = torch.optim.Adam(
    [
        {'params': model.module.affine_layer.parameters()},
        #{'params': model.module.fc.parameters()},  
    ],
    lr=lr,
    weight_decay=weight_decay
    )
    '''
    criterion = nn.SmoothL1Loss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr,
                                weight_decay=weight_decay)

    '''
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                               milestones=lr_decay_milestones, 
                                               gamma=lr_decay_gamma)
    
    global_step = 0
    e = []
    train_loss = []
    train_error = []
    val_loss = []
    val_error = []
    best_val_loss = float('inf')
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_error = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as bar:
            for i, (Img, Face, LEye, REye, Grid, GazePoint, GazePointPixel) in enumerate(train_loader):

                Face = Face.to(device)
                LEye = LEye.to(device)
                REye = REye.to(device)
                Grid = Grid.to(device)
                GazePoint = GazePoint.to(device)

                # compute output
                pred_gaze = model(Face, LEye, REye, Grid)

                loss = criterion(pred_gaze, GazePoint)
                error = torch.norm(pred_gaze-GazePoint, dim=1).mean()

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # increment the total epoch loss
                epoch_loss += loss.item()
                epoch_error += error.item()

                # update the progress bar with the intermediate loss value
                bar.set_postfix(**{'loss per batch': loss.item()})
                bar.update()

                global_step += 1

            mean_epoch_loss = round(epoch_loss / len(train_loader), 5)
            mean_epoch_error = round(epoch_error / len(train_loader), 5)
            print('\n', f"Epoch {epoch+1} Train loss:", mean_epoch_loss, "Train Error(cm):", mean_epoch_error)
            train_loss.append(mean_epoch_loss)
            train_error.append(mean_epoch_error)
            e.append(epoch+1)

        scheduler.step()
    
        mean_val_loss, mean_val_error = validate(model, val_loader, criterion)
        print('\n', f"Epoch {epoch+1} Val loss:", mean_val_loss, "Val Error(cm):", mean_val_error)
        val_loss.append(mean_val_loss)
        val_error.append(mean_val_error)

        if save_cp_and_plot:
            checkpoints_dir = f"results/{args.result_name}"
            os.makedirs(checkpoints_dir, exist_ok=True)
            save_plot_jpg(e, train_loss, train_error, val_loss, val_error, checkpoints_dir)
            if (epoch+1) % save_per_epoch == 0:
                cp_name = 'Epoch' + str(epoch + 1) + '.pth'
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, cp_name))
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best.pth'))
    end_time = time.time()
    elapsed_time = end_time - start_time
    mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_frame=test_main(model, os.path.join(checkpoints_dir, 'best.pth'))
    total_elapsed_time = print_train_time(elapsed_time)
    output_dict = {
    'settings': checkpoints_dir,
    'total_train_time': total_elapsed_time,
    'mean_test_loss': mean_test_loss,
    'mean_test__error': mean_test_error,
    'mean_inference_time': mean_inference_time,
    'mean_inference_time_per_frame': mean_inference_time_per_frame,
    }

    filename = 'test_result.csv'
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_dict.keys())
            writer.writeheader()
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_dict.keys())
        writer.writerow(output_dict)
        
def validate(model, val_loader, criterion):
    if len(val_loader) == 0:
        print("NO VAL SET!")
        return 0, 0
    else:
        model.eval()
        val_loss=0
        val_error=0
        with tqdm(total=len(val_loader), desc="Validation") as vbar:
            for i, (Img, Face, LEye, REye, Grid, GazePoint, GazePointPixel) in enumerate(val_loader):
                Face = Face.to(device)
                LEye = LEye.to(device)
                REye = REye.to(device)
                Grid = Grid.to(device)
                GazePoint = GazePoint.to(device)
                with torch.no_grad():
                    pred_gaze = model(Face, LEye, REye, Grid)
                    loss = criterion(pred_gaze, GazePoint)
                    error = torch.norm(pred_gaze-GazePoint, dim=1).mean()

                    val_loss += loss.item()
                    val_error += error.item()

                    vbar.set_postfix(**{'loss per batch': loss.item()})
                    vbar.update()

            mean_val_loss = round(val_loss / len(val_loader), 5)
            mean_val_error = round(val_error / len(val_loader), 5)
        return mean_val_loss, mean_val_error

def test_main(model, weight_path):
    model.eval()
    model = torch.nn.DataParallel(model)
    model = load_weight(weight_path)
    model.to(device)
    cudnn.benchmark = True 
    dataTest = BRL4200GazeData(args.data_path, split='test', right_eye_flip=right_eye_flip, face_size=faceSize, eye_size=eyeSize, grid_size=gridSize, landmark_mask=landmark_mask, meta_file=args.meta_file, onlyRawImage=onlyRawImage)
    test_loader = torch.utils.data.DataLoader(
        dataTest,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    criterion = nn.SmoothL1Loss().to(device)
    mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_second = test(model, test_loader, criterion)
    return mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_second

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
                Face = Face.to(device)
                LEye = LEye.to(device)
                REye = REye.to(device)
                Grid = Grid.to(device)
                GazePoint = GazePoint.to(device)
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
        return mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_second
    elif onlyRawImage == True:
        test_loss=0
        inference_times=[]
        test_error = 0
        with tqdm(total=len(test_loader), desc="Test") as vbar:
            for i, (Img, Face, LEye, REye, Grid, GazePoint) in enumerate(test_loader):
                Img = Img.to(device)
                GazePoint = GazePoint.to(device)
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
        return mean_test_loss, mean_test_error, mean_inference_time, mean_inference_time_per_second

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


def save_plot_jpg(e, train_loss, train_error, val_loss, val_error, checkpoints_dir):

    plt.figure(figsize=(10, 8))

    # 绘制训练和验证的损失函数
    plt.subplot(2, 1, 1)
    plt.plot(e, train_loss, linestyle="-", color="red", label='train_loss')
    plt.plot(e, val_loss, linestyle="-", color="green", label='val_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制训练和验证的error(cm)
    plt.subplot(2, 1, 2)
    plt.plot(e, train_error, linestyle="-", color="red", label='train_error(cm)')
    plt.plot(e, val_error, linestyle="-", color="green", label='val_error(cm)')
    plt.xlabel("Epochs")
    plt.ylabel("Error(cm)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoints_dir, "plot.jpg"), dpi=600)

def count_trainable_params(model):
    return sum(p.numel() for p in model.module.parameters() if p.requires_grad)

def print_train_time(elapsed_time):
    elapsed_hours = elapsed_time // 3600
    elapsed_minutes = (elapsed_time % 3600) // 60
    elapsed_seconds = (elapsed_time % 3600) % 60
    print(f"Training took: {elapsed_hours:.0f}h {elapsed_minutes:.0f}m {elapsed_seconds:.0f}s")
    return "{elapsed_hours:.0f}h {elapsed_minutes:.0f}m {elapsed_seconds:.0f}s"

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
    main(model)