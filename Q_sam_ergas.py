from scipy.io import loadmat
import math
import torch
import h5py
import numpy as np


def compute_index(img_base,img_out,ratio):
    h = img_out.shape[0]
    w = img_out.shape[1]
    chanel = img_out.shape[2]
    #SAM
    sum1 = torch.sum(img_base* img_out,2)
    sum2 = torch.sum(img_base* img_base,2)
    sum3 = torch.sum(img_out* img_out,2)
    t=(sum2*sum3)**0.5
    numlocal=torch.gt(t, 0)
    num=torch.sum(numlocal)
    t=sum1 / t
    angle = torch.acos(t)
    sumangle= torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num==0:
        averangle=sumangle
    else:
        averangle=sumangle/num
    SAM=averangle*180/3.14159256

    #ERGAS
    summ=0
    for i in range(chanel):
        a1 = torch.mean((img_base[:, :, i] - img_out[:, :, i])**2)
        m1=torch.mean(img_base[:, :, i])
        a2=m1*m1
        summ=summ+a1/a2
    ERGAS=100*(1/ratio)*((summ/chanel)**0.5)
    return SAM,ERGAS

def my_compute_sam_ergas(tensor_data,data_file='data/Test_WDC.h5',img_number = 4): # NHWC
    ref_data = h5py.File(data_file)
    ref_data = ref_data['GT'][:]
    tensor_ref_data = torch.from_numpy(ref_data).permute([0, 2, 3, 1])

    tensor_data = torch.clamp(tensor_data, min=0.0)

    ergas_value = np.zeros(img_number)
    sam_value = np.zeros(img_number)
    for i in range(img_number):  # i = 0123
        mynet = tensor_data[i, :, :, :]  # 128 128 191
        ref = tensor_ref_data[i, :, :, :]  # 128 128 191
        sam, ergas = compute_index(ref, mynet, 1)
        ergas_value[i] = ergas.float()
        sam_value[i] = sam.float()
    ERGAS = np.mean(ergas_value)
    SAM = np.mean(sam_value)
    return SAM, ERGAS
