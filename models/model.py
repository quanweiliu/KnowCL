# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:52:59 2020

@author: 86186
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import resnet50
from torchvision.models import wide_resnet50_2

from torchvision.models import resnext50_32x4d
from torchvision.models import densenet161
#from cbam_resnext import cbam_resnext50_16x64d

    
class Model_base(nn.Module):
    def __init__(self, input_dim):
        super(Model_base, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():      # 512
        # for name, module in resnet50().named_children():
        # for name, module in resnext50_32x4d().named_children():
        # for name, module in wide_resnet50_2().named_children():
            if name == 'conv1':
                module = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=True)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        
        
        # encoder
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature
    

class DINOHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))

    def forward(self, x):
        x = self.g(x)
        return x



class MLP_head(nn.Module):
    def __init__(self, in_dim=512, class_num=16):
        super(MLP_head, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num, bias=True))   #2048
    def forward(self, x):
        x = self.c(x)
        return x



class FDGC_head(nn.Module):
    def __init__(self, in_dim=128, class_num=17):
        super(FDGC_head, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 1024),
                               nn.Dropout(0.5),
                               nn.BatchNorm1d(1024),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(1024, 256),
                               nn.BatchNorm1d(256),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num)   
                               )   #2048
    def forward(self, x):
        x = self.c(x)
        return x



# print(model)
if __name__ == "__main__":
    input1 = torch.rand(128,32,28,28)
    model = Model_base(32)
    model2 = FDGC_head(in_dim=512)

    feature = model(input1)
    out = model2(feature)
    print ("feature1", feature.shape)
    print("output1", out.shape)



