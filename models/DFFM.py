import torch
from torch import nn
from models.DMSAF import dmsaf
from models.DFFGA_1 import FGA




class DFFM(nn.Module):
    def __init__(self, in_channels, out_channels, sample, up=True, kernel_list=[3, 9]):
        super().__init__()

        # Deformable Multi-Scale Attention Fusion
        self.msf = dmsaf(in_channels,kernel_list=kernel_list)
        
        # Dual Fourier-Feature Guided Attention
        self.mpf = FGA(in_channels)
        
        # 多尺度多频率融合后的特征处理
        self.mlp = nn.Sequential(
                nn.BatchNorm2d(in_channels),  # 输入通道数是两者的合并
                nn.Conv2d(in_channels, out_channels, 1),  # 1x1卷积进行特征降维
                
                nn.GELU(),                                        # 使用GELU激活函数
                nn.Conv2d(out_channels, out_channels, 1),  # 第二次卷积进行特征处理
                nn.BatchNorm2d(out_channels)
            )
        
        # 上采样或池化选择
        if sample:
            if up:
                self.sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.sample = nn.MaxPool2d(2, stride=2)
        else:
            self.sample = None

        
        # 使用 1x1 卷积来调整通道数减半
        self.reduce_channels = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
 
        # x_msf = self.msf(x)
        # print("x_msf:",x_msf.shape)
        # x_mpf = self.mpf(x)
        # print("x_mpf:",x_mpf.shape)
        # x_cat = torch.cat([x_msf, x_mpf], dim=1)    
        # print("x_cat:",x_cat.shape)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        # x = self.mlp(x_cat)  
        # # 进行上采样或池化
        # if self.sample is not None:
        #     x = self.sample(x)
        
        # return x

        # # DFFGA消融实验
        x_msf = self.msf(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        x = self.mlp(x_msf)  # 突出显著特征，弱化非显著特征

        # DMSAF消融实验
        # x_mpf = self.mpf(x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        # x = self.mlp(x_mpf)  # 突出显著特征，弱化非显著特征
        
        if self.sample is not None:
            x = self.sample(x)
        
        return x
        
