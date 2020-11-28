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


def rec(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(gray, 1)

    if dets is not None:
        for face in dets:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(img, (left, top),
                          (right, bottom), (0, 255, 0), 2)
            img1 = cv2.resize(
                img[top:bottom, left:right], dsize=(256,256))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = preprocess(Image.fromarray(img1))
            img1=img1.unsqueeze(0)

            prediction = mymodel.network(img1)

            if prediction[0][0] >= 0.5:
                result = "mask"
            else:
                result = "nomask"

            #print(result)

            cv2.putText(img, result, (left, top), font,
                         2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("mask detector", imgrd)



if __name__ == "__main__":
    network = Network_simple_cnn()
    mymodel = Models(network, None, None, None)
    mymodel.model_load("simple-CNN.pkl")

    detector = dlib.get_frontal_face_detector()
    video = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize(
        256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])


    while video.isOpened():
        res, imgrd = video.read()
        #print(res)
        if not res:
            break

        rec(imgrd)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
