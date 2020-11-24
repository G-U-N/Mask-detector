from PIL import Image
from torch import imag
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import os
from tqdm import tqdm
import torch
import io
import numpy as np

class Data():
    def __init__(self,positive_path,negative_path):
        self.normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        self.preprocess=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),self.normalize])
        self.posdata=self.get_data(positive_path)
        self.poslabel=torch.ones(size=(len(self.posdata),1))
        #print(self.posdata.shape,self.poslabel.shape)
        self.negdata=self.get_data(negative_path)
        self.neglabel=torch.zeros(size=(len(self.negdata),1))
        self.data=torch.cat([self.posdata,self.negdata])
        self.label=torch.cat([self.poslabel,self.neglabel])
        self.dataset=TensorDataset(self.data,self.label)
        
    def get_data(self,path):
        ans=[]
        for i in tqdm(os.listdir(path)):
            img=Image.open(path+"/"+i)
            ans.append(torch.unsqueeze(self.preprocess(img),0))
        return torch.cat(ans)

    def get_dataloader(self,batchsize=300,shuffle=True):
        return DataLoader(self.dataset,batch_size=batchsize,shuffle=shuffle)
if __name__=='__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    pospath="D:/QQ文件/Models/口罩识别/人脸口罩数据集，正样本加负样本/mask/have_mask"
    negpath="D:/QQ文件/Models/口罩识别/人脸口罩数据集，正样本加负样本/mask/no_mask"
    mask_data=Data(pospath,negpath)
    mask_data_loader=mask_data.get_dataloader()
    print(mask_data.dataset)
    #print(mask_data_loader)



            





