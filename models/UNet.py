import torch
from torch import nn
from .unet_base import *


class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_conv = Double_conv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, [x4, x3, x2, x1]


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out_conv = Out_conv(64, 3)

    def forward(self, x, encoder_outputs):
        x4, x3, x2, x1 = encoder_outputs
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits


if __name__ == "__main__":
    encoder = UNetEncoder()
    decoder = UNetDecoder()

    # 创建一个随机张量作为输入
    x = torch.randn(1, 3, 224, 224)

    # 测试encoder的输出
    encoder_output, inter_outputs = encoder(x)
    print("Encoder output shape: ", encoder_output.shape)
    # 输出Encoder output shape:  torch.Size([1, 512, 7, 7])

    # 测试decoder的输出
    decoder_output = decoder(encoder_output, inter_outputs)
    print("Decoder output shape: ", decoder_output.shape)
    # 输出Decoder output shape:  torch.Size([1, 3, 224, 224])
