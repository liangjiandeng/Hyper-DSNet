import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int


# -------------Initialization----------------------------------------  初始化 3种
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        init_weights(self.conv_du, self.avg_pool)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y



# -----------------------------------------------------  pannet网络
class Block2(nn.Module):
    def __init__(self, n_feature=16, channel=191+7):
        super(Block2, self).__init__()

        self.conv1 = nn.Sequential(
            conv(channel, n_feature, kernel_size=3, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv(channel, n_feature, kernel_size=5, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            conv(channel, n_feature, kernel_size=7, bias=True),
            nn.ReLU(inplace=True),
        )
        init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):  # x:output1
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat((out1, out2, out3), 1)
        return out

##########################################################################
class Block3(nn.Module):
    def __init__(self, channel=191, reduction=16):
        super(Block3, self).__init__()
        self.relu = nn.ReLU()
        self.conv01 = conv(48,32,3,True)
        self.conv02 = conv(32,16,3,True)
        self.conv03 = conv(16,8,3,True)
        self.conv04 = conv(8,8,3,True)

        init_weights(self.conv01, self.conv02, self.conv03, self.conv04)

    def forward(self, x):  # x:pan y:edge_pan z:lms ms:ms    1 6 191 = 198
        x1 = self.relu(self.conv01(x))
        x2 = self.relu(self.conv02(x1))
        x3 = self.relu(self.conv03(x2))
        x4 = self.relu(self.conv04(x3))
        out = torch.cat((x,x1,x2,x3,x4),1)    # 48+32+16+8+8
        return out

class Hyper_DSNet(nn.Module):
    def __init__(self):
        super(Hyper_DSNet, self).__init__()

        self.block2 = Block2()
        self.block3 = Block3()
        self.CA = CALayer(191, 4, bias=True)
        self.convlast = conv(112, 191, 1, True)

    def forward(self, x, y, z, ms):  # x:pan y:edge_pan z:lms ms:ms
        input1 = torch.cat((x,y,z),1)
        input2 = self.block2(input1)
        output1 = self.block3(input2)
        output1 = self.convlast(output1)
        res = output1 * self.CA(ms)
        output = res + z

        return output


# ----------------- End-Main-Part ------------------------------------
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def inspect_weight_decay():
    ...
