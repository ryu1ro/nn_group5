import numpy as np 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt


#create dataset
class FER2013Dataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df_data = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        X = self.df_data.iloc[idx][' pixels']
        X = np.array(X).reshape(48,48)
        y = self.df_data.iloc[idx]['emotion']
      
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y

def preprocess(x, y):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device=dev, dtype=torch.float), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
        self.dataset = dl.dataset

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def get_dl(
    data='training', 
    bs=64, 
    shuffle=True, 
    transform=tt.Compose([tt.ToTensor(), tt.Normalize(0.5, 0.5)])):
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = '/content/drive/MyDrive/QMUL/NN/data/' 
    df = pd.read_pickle(data_path+data+'.pkl')
    ds = FER2013Dataset(df, transform=transform)
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    dl = WrappedDataLoader(dl, preprocess)
    return dl