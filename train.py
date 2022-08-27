import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
from model import Hyper_DSNet
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
import datetime
from Q_sam_ergas import compute_index

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False

lr = 0.0001
epochs = 6000
ckpt = 100  # 每隔多少部存model
batch_size = 32
version = ''
version2 = ''
version1 = ''
start_epoch = 0
model_path = "Weights/"+ version +"/"+ version1 + str(start_epoch) +".pth"  # 模型参数存放地址
log_path = "train_log.txt"

# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#  预先定义模型，损失函数，正则优化器，学习率更新
model = Hyper_DSNet().cuda()  # pannet从model里读出来
if os.path.isfile(model_path):  # 如果有一些预训练的model可以直接调用
    model.load_state_dict(torch.load(model_path))  ## Load the pretrained Encoder
    print('Network is Successfully Loaded from %s' % model_path)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print('lr:{}'.format(lr))
print(get_parameter_number(model))
with open(log_path, mode='a') as filename:
    filename.write('\n')  # 换行
    filename.write('lr:{}  version:{}  start_epoch:{}  MSELoss  Model_2c  batch_size = {}  Adam_weight_decay=0  parameter_number:{}'.format(lr,version2,start_epoch,batch_size,get_parameter_number(model)))
    filename.write('\n')  # 换行

# summaries(model, grad=True)  ## Summary the Network   训练时把整个网络的结构、参数量打印出来
criterion = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function L1Loss
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)  # optimizer 1: Adam   优化器
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)   # learning-rate update  学习率的更新
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)

# writer = SummaryWriter('./train_logs/50')  ## Tensorboard_show: case 2

def save_checkpoint(model, epoch):  # save model function  保存模型
    model_out_path = 'Weights/' + version + "/"+ version2 +"{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


def ergas_loss_function(input, target, img_number):
    input = input.permute([0, 2, 3, 1])  # NHWC
    target = target.permute([0, 2, 3, 1])
    ergas_value = np.zeros(img_number)
    sam_value = np.zeros(img_number)
    # ERGAS = np.zeros(1) #biancheng ndarray not float
    # SAM = np.zeros(1)
    for i in range(img_number):  # i = 0123
        mynet = input[i, :, :, :]  # 128 128 191
        ref = target[i, :, :, :]  # 128 128 191
        sam, ergas = compute_index(ref, mynet, 1)
        ergas_value[i] = ergas.float()
        sam_value[i] = sam.float()

    ERGAS = torch.from_numpy(ergas_value).float()
    ERGAS = torch.mean(ERGAS).cuda()
    ERGAS.requires_grad_()

    # SAM[0] = np.mean(sam_value)
    # SAM = torch.from_numpy(SAM).float()
    return ERGAS


def train(training_data_loader, validate_data_loader, start_epoch=0):
    print('Start training...')

    starttime = datetime.datetime.now()
    print(starttime)

    for epoch in range(start_epoch, epochs, 1):  # epochs决定每个样本迭代的次数

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):  # iteration就是分批数，循环完以后每批数据都迭代了一次

            gt, lms, ms, pan, edge_pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[
                4].cuda()

            optimizer.zero_grad()

            output1 = model(pan, edge_pan, lms, ms)
            loss = criterion(output1, gt)  # compute loss

            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()  # fixed
            optimizer.step()  # fixed

            # for name, layer in model.named_parameters():
            #     writer.add_histogram('net/' + name + '_data_weight_decay', layer, epoch * iteration)

        # lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        # writer.add_scalar('L1_loss/t_loss', t_loss, epoch)  # write to tensorboard to check

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #  验证 测试valid
        model.eval()
        ergas_loss = []

        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms, pan, edge_pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[
                    4].cuda()

                output1 = model(pan, edge_pan, lms, ms)

                loss = criterion(output1, gt)

                ergas = ergas_loss_function(output1, gt, batch_size)
                ergas_loss.append(ergas.item())

                epoch_val_loss.append(loss.item())

        if epoch % 1 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            v_ergas = np.nanmean(np.array(ergas_loss))
            # writer.add_scalar('val/v_loss', v_loss, epoch)
            print('Epoch: {}/{} training loss:{}  validate loss:{}  ergas:{}'.format(epochs, epoch, t_loss, v_loss, v_ergas))  # print loss for each epoch
            with open(log_path, mode='a') as filename:
                endtime = datetime.datetime.now()
                filename.write('Epoch: {}/{} training loss:{}  validate loss:{}  ergas:{}  time:{}'.format(epochs, epoch, t_loss, v_loss, v_ergas, endtime - starttime))
                filename.write('\n')  # 换行
            endtime = datetime.datetime.now()
            # print(endtime)
            # print(endtime - starttime)

    endtime = datetime.datetime.now()
    print("time:{}".format(endtime - starttime))
    # writer.close()  # close tensorboard

    with open(log_path, mode='a') as filename:
        filename.write('\n')  # 换行


if __name__ == "__main__":
    train_set = Dataset_Pro('data/Train_WDC.h5')  # creat data for training   数据处理的函数，在data里面
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches  按batch_size个数据分批

    validate_set = Dataset_Pro('data/Valid_WDC.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training d ata to DataLoader for batches

    train(training_data_loader, validate_data_loader, start_epoch)  # call train function (call: Line 66)   分批以后的数据train
