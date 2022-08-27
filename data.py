import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
from EdgeDetection import Edge



class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path,'r+')  # NxCxHxW = 0x1x2x3=nx191x64x64   channel height width

        self.gt = data.get("GT")
        self.lms = data.get("LMS")
        self.ms = data.get("MS")
        self.pan = data.get("PAN")

        self.edge_pan = Edge(self.pan)



    def __getitem__(self, index):
        return torch.from_numpy(self.gt[index, :, :, :]).float(), \
                torch.from_numpy(self.lms[index, :, :, :]).float(), \
                torch.from_numpy(self.ms[index, :, :, :]).float(), \
                torch.from_numpy(self.pan[index, :, :, :]).float(), \
                torch.from_numpy(self.edge_pan[index, :, :, :]).float()


    def __len__(self):
        return self.gt.shape[0]
