import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn


def load_mnist_images(path):
    with open(path,"rb") as f:
        return np.frombuffer(f.read(),dtype=np.uint8,offset=16).reshape(-1,28,28)

def load_mnist_y(path):
    with open(path,"rb") as f:
        return np.frombuffer(f.read(),dtype=np.uint8,offset=8)

class my_dataset(Dataset):
    
    def __init__(self,x,y,transform):
        self.X=x
        self.Y=torch.tensor(y)
        self.transform=transform
    
    def __getitem__(self,idx):
        x=self.X[idx]
        y=self.Y[idx]
        if self.transform:
            x=self.transform(self.X[idx])
        return x, y
    
    def __len__(self):
        return len(self.Y)
    
class my_test_dataset(Dataset):
    
    def __init__(self,x,y):
        self.X=torch.tensor(x,dtype=torch.float32)
        self.Y=torch.tensor(y)
    
    def __getitem__(self,idx):
        x=self.X[idx]
        y=self.Y[idx]
        return x, y
    
    def __len__(self):
        return len(self.Y)    


class my_neural_net(nn.Module):
    def __init__(self,input_layer,hl_1,hl_2,output_layer):
        super().__init__()
        self.relu=nn.ReLU()
        self.l1=nn.Linear(input_layer,hl_1)
        self.l2=nn.Linear(hl_1,hl_2)
        self.l3=nn.Linear(hl_2,output_layer)
        
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        out=self.relu(out)
        out=self.l3(out)
        return out