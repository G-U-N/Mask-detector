import datetime
from network import Network_simple_cnn
from data import Data
import numpy as np
import io
import torch
from tqdm import tqdm
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torch import imag
from torch.nn.modules import loss
from torchvision import transforms
import torchvision.models as models


class Models():
    def __init__(self, network, loss_fn, dataloader, optimizer):
        self.network = network
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.optim = optimizer

    def trainloop(self, n_epochs):
        for epoch in range(1, n_epochs+1):
            self.evaluate(mask_data.data,mask_data.label)
            loss_train = 0.0
            for input, realout in self.dataloader:
                predictout = self.network(input)

                loss = self.loss_fn(predictout, realout)

                self.optim.zero_grad()

                loss.backward()
                self.optim.step()
                loss_train += loss.item()
            #if epoch == 1 or epoch % 100 == 0:
            print(
                    f'{datetime.datetime.now()} epoch {epoch} training loss {loss_train/len(self.dataloader)}')
            #self.evaluate(mask_data.data,mask_data.label)
    def evaluate(self,valdata,vallabel):
        ans=0.;
        with torch.no_grad():
            prelabel=self.network(valdata)
            for i in range(len(prelabel)):
                if prelabel[i][0]>=0.5 and int(vallabel[i][0])==1:
                    ans+=1
                if prelabel[i][0]<0.5 and int(vallabel[i][0])==0:
                    ans+=1
        print("the accuracy is",ans/float(len(prelabel)))   
    def model_save(self,path):
        torch.save(self.network.state_dict(),path)
    def model_load(self,path):
        self.network.load_state_dict(torch.load(path))

            
                    


if __name__ == '__main__':
    #torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_tensor_type(torch.FloatTensor)
    pospath = "D:/QQ文件/Models/口罩识别/人脸口罩数据集，正样本加负样本/mask/have_mask"
    negpath = "D:/QQ文件/Models/口罩识别/人脸口罩数据集，正样本加负样本/mask/no_mask"
    mask_data = Data(pospath, negpath)
    mask_data_loader = mask_data.get_dataloader()

    #network = models.resnet18(num_classes=1)
    network=Network_simple_cnn()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    mymodel = Models(network, loss_fn, mask_data_loader, optimizer)

    mymodel.trainloop(50)

    #mymodel.model_save("simple-CNN.pkl")

#    mymodel.model_load("simple-CNN.pkl")
