import torch
import torch.nn as nn
import torch.nn.parallel
from .UNet import *

class ItrackerImageModel(nn.Module):
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.encoder = UNetEncoder()
        
        self.conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        encoder_output, inter_outputs = self.encoder(x)
        x = self.conv(encoder_output)
        x = x.view(x.size(0), -1)
        return x, encoder_output, inter_outputs


class FaceImageModel(ItrackerImageModel):
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x, _, _ = super().forward(x)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ITrackerModel(nn.Module):
    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        self.eyeDecoder = UNetDecoder()
        #self.eyesFC = nn.Sequential(
            #nn.Linear(128, 128),
            #nn.ReLU(inplace=True),
        #)
        self.fc = nn.Sequential(
            nn.Linear(128+64+128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        xEyeL, L_encoder_output, L_inter_outputs = self.eyeModel(eyesLeft)
        xEyeR, R_encoder_output, R_inter_outputs = self.eyeModel(eyesRight)
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        #xEyes = self.eyesFC(xEyes)
        xEyeL_hm = self.eyeDecoder(L_encoder_output, L_inter_outputs)
        xEyeR_hm = self.eyeDecoder(R_encoder_output, R_inter_outputs)
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)
        
        return x, xEyeL_hm, xEyeR_hm


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ITrackerModel().to(device)
    
    face = torch.randn(1, 3, 224, 224).to(device)
    lefteye = torch.randn(1, 3, 112, 112).to(device)
    righteye = torch.randn(1, 3, 112, 112).to(device)
    facegrid = torch.randn(1, 1, 25, 25).to(device)

    gazepoint, xEyeL_hm, xEyeR_hm = model(face, lefteye, righteye, facegrid)
    print("output shape: ", gazepoint.shape, xEyeL_hm.shape, xEyeR_hm.shape)
