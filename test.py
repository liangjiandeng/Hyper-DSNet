import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data import Dataset_Pro
from model import Hyper_DSNet
import h5py
import scipy.io as sio
import os
import datetime
from EdgeDetection import Edge
from Q_sam_ergas import my_compute_sam_ergas

version = '01'
version2 = 'a'
epoch_list = list(range(20, 51))
epoch_list = [100*x for x in epoch_list]
for epoch in epoch_list:

    load_weight = "Weights/"+ version +"/" + version2 + "{}.pth".format(epoch)  # chose model
    test_file_path = "data/Test_WDC.h5"

    num_testing = 4
    size = 128
    channel = 191

    test_set = Dataset_Pro(test_file_path)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=num_testing)

    model = Hyper_DSNet()
    model = model.cuda()  # fixed, important!

    weight = torch.load(load_weight)  # load Weights!
    model.load_state_dict(weight)  # fixed

    output1 = np.zeros([num_testing, size, size, channel])
    # output2 = np.zeros([num_testing, size, size, channel])

    starttime = datetime.datetime.now()

    for iteration, batch in enumerate(testing_data_loader, 1):  # just one
        gt, lms, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()

        edge_pan = torch.from_numpy(Edge(pan.cpu().numpy())).cuda().float()
        with torch.no_grad():
            outputi1 = model(pan, edge_pan, lms, ms)  # pan, edge_pan, lms, ms
            output1[:, :, :, :] = outputi1.permute([0, 2, 3, 1]).cpu().detach().numpy()   #output:numpy
            # output2[:, :, :, :] = outputi2.permute([0, 2, 3, 1]).cpu().detach().numpy()   #output:numpy

    sam, ergas = my_compute_sam_ergas(torch.from_numpy(output1), test_file_path, num_testing) #torch
    # sam2, ergas2 = my_compute_sam_ergas(torch.from_numpy(output2), test_file_path, num_testing) #torch

    endtime = datetime.datetime.now()
    # print("time:{}".format((endtime - starttime)))
    print("epoch:{}  sam:{}   ergas:{}".format(epoch,sam,ergas))

    save_name = "outputs/"+ version +"/"+ version2 +"{}.mat".format(epoch)  # chose model
    sio.savemat(save_name, {'output': output1})
