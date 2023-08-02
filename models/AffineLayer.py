import torch
import torch.nn as nn
import math

class AffineLayer(nn.Module):
    def __init__(self):
        super(AffineLayer, self).__init__()
        self.affine_matrix = nn.Parameter(torch.Tensor(2, 3))  # 参数化的仿射变换矩阵
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.affine_matrix, a=math.sqrt(5))  # Kaiming 初始化

    def forward(self, x):
        x = torch.cat((x, torch.ones(x.size(0), 1).to(x.device)), dim=1)  # 添加一个维度并填充为1，以形成[x, y, 1]的列向量
        x = torch.mm(x, self.affine_matrix.t())  # 矩阵相乘
        return x
    
if __name__ == "__main__":
    affine_layer = AffineLayer()

    # 验证 AffineLayer
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)  # 2x2 tensor
    print("输入Tensor: \n", input_tensor)
    
    output_tensor = affine_layer(input_tensor)
    print("输出Tensor: \n", output_tensor)