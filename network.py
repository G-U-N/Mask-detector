import datetime
from data import Data
import numpy as np
import io
import torch
from tqdm import tqdm
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Conv2d,BatchNorm2d,ReLU,MaxPool2d,Sequential,Linear,Sigmoid,Dropout
from PIL import Image
from torch import imag
from torch.nn.modules import loss
from torchvision import transforms
import torchvision.models as models

class Network_simple_cnn(nn.Module):
    def __init__(self):
        super(Network_simple_cnn,self).__init__()

        self.cnn_layers = Sequential(
            # 定义2D卷积层
            Conv2d(3, 4, kernel_size=3, stride=3, padding=0),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=3),
            
        )

        self.linear_layers = Sequential(
            Linear(2500,256),
            Dropout(0.3),
            Linear(256,1),
            Sigmoid()    
        )

    # 前项传播
    def forward(self, x):
        x = self.cnn_layers(x)
        ## 相当于keras当中的flatten
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        