import cv2
from train import Models
from network import Network_simple_cnn
from data import Data
import datetime
import numpy as np
import io
import torch
from tqdm import tqdm
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, Linear, Sigmoid
from PIL import Image
from torch import imag
from torch.nn.modules import loss
from torchvision import transforms
import torchvision.models as models
import dlib

if __name__ == "__main__":
    network = Network_simple_cnn()
    mymodel = Models(network, None, None, None)
    mymodel.model_load("simple-CNN.pkl")

    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize(
        256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

    while cap.isOpened():
        flag,frame=cap.read()

        if flag==False:
            break

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        face_zone=detector.detectMultiScale(gray)
        #print(np.array(face_zone))
        if len(face_zone)!=0:

            for face in face_zone:
                left=face[0]
                top=face[1]
                h=face[2]
                cv2.rectangle(frame, (left, top),
                (left+h, top+h), (0, 255, 0), 2)
                img1 = frame[top:top+h,left:left+h]
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img1 = preprocess(Image.fromarray(img1))
                img1=img1.unsqueeze(0)

                prediction = mymodel.network(img1)

                if prediction[0][0] >= 0.5:
                    result = "mask"
                else:
                    result = "nomask"

            #print(result)

                cv2.putText(frame, result, (left, top), font,
                         2, (0, 255, 0), 2, cv2.LINE_AA)

        

        #print(face_zone)

        cv2.imshow("mask",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
   
    
