#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
import torch.nn.functional as F
import numpy as np
from mine.network_stage_2_mine_x2_resflow import Stage_2_generator
# from correlation import correlation

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def conv3X3(input_dim,output_dim,stride=1,padding=1):
    return nn.Conv2d(input_dim,output_dim,kernel_size=3,stride=stride,padding=padding,bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class ResnetBlock(nn.Module):
    def __init__(self,input_dim,output_dim,stride=1,padding=1,kernel_size=3,bias=True,norm_layer=nn.BatchNorm2d,downSample=False):
        super(ResnetBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim,out_channels=output_dim,kernel_size=kernel_size,padding=padding,stride=stride,bias=bias)
        self.bn1 = norm_layer(output_dim)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=output_dim,out_channels=output_dim,kernel_size=kernel_size,padding=padding,stride=1,bias=bias)
        self.bn2 = norm_layer(output_dim)
        self.relu2 = nn.ReLU(True)
        self.downSample = None
        if downSample == True:
            self.downSample = nn.Sequential(conv1x1(input_dim,output_dim,stride=stride))

    def forward(self, input):
        identity = input
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downSample != None:
            identity = self.downSample(input)
        output = identity + output
        output = self.relu2(output)
        return output

class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        return x
def conv_blck(in_channels, out_channels, kernel_size=3,
              stride=1, padding=1, dilation=1, bn=False):
    if bn:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.ReLU(inplace=True))

class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlock, self).__init__()

        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}

        conv1 = nn.Conv2d(input_nc,  output_nc,  **kwargs_down)
        conv2 = nn.Conv2d(output_nc, output_nc,  **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential( conv1,norm_layer(output_nc),  nonlinearity,
                                       conv2,norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out

class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc
        self.learnable_shortcut = True if input_nc != output_nc else learnable_shortcut

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': False}

        conv1 =nn.Conv2d(input_nc, hidden_nc,  **kwargs)
        conv2 =nn.Conv2d(hidden_nc, output_nc,  **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(conv1,norm_layer(hidden_nc), nonlinearity,
                                       conv2,norm_layer(output_nc), nonlinearity)

        if self.learnable_shortcut:
            bypass = nn.Conv2d(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
            self.shortcut = nn.Sequential(bypass,)


    def forward(self, x):
        if self.learnable_shortcut:
            out = self.model(x) + self.shortcut(x)
        else:
            out = self.model(x) + x
        return out

class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1,bias=False)
        conv2 = nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)
        bypass = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(conv1,norm_layer(hidden_nc), nonlinearity,conv2, norm_layer(output_nc), nonlinearity)

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

class DecokerBlock(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(DecokerBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1,bias=False)
        conv2 = nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(conv1,norm_layer(hidden_nc), nonlinearity,conv2, norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out

class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': False}

        self.conv1 = nn.Conv2d(input_nc, output_nc, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out
class Generator(nn.Module):
    def __init__(self,input_dim1=3,input_dim2=20,norm_layer=nn.BatchNorm2d):
        super(Generator, self).__init__()
        self.encode1 = [nn.Sequential(EncoderBlock(input_dim1,output_nc=64,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(64,output_nc=128,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(128,output_nc=256,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(256,output_nc=256,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(256,output_nc=256,norm_layer=nn.BatchNorm2d))
                        ]

        self.encode2 =  [nn.Sequential(EncoderBlock(input_dim2,output_nc=64,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(64,output_nc=128,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(128,output_nc=256,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(256,output_nc=256,norm_layer=nn.BatchNorm2d))
                       ,nn.Sequential(EncoderBlock(256,output_nc=256,norm_layer=nn.BatchNorm2d))
                        ]

        self.encode3 = [nn.Sequential(EncoderBlock(3, output_nc=64, norm_layer=nn.BatchNorm2d))
            , nn.Sequential(EncoderBlock(64, output_nc=128, norm_layer=nn.BatchNorm2d))
            , nn.Sequential(EncoderBlock(128, output_nc=256, norm_layer=nn.BatchNorm2d))
            , nn.Sequential(EncoderBlock(256, output_nc=256, norm_layer=nn.BatchNorm2d))
            , nn.Sequential(EncoderBlock(256, output_nc=256, norm_layer=nn.BatchNorm2d))
                        ]
        self.upSample = [nn.Sequential(ResBlock(768,768,norm_layer=nn.BatchNorm2d),ResBlockDecoder(768,256,norm_layer=nn.BatchNorm2d)),
                         nn.Sequential(ResBlock(768,768,norm_layer=nn.BatchNorm2d),ResBlockDecoder(768,256,norm_layer=nn.BatchNorm2d)),
                         nn.Sequential(ResBlock(768,768,norm_layer=nn.BatchNorm2d),ResBlockDecoder(768,128,norm_layer=nn.BatchNorm2d)),
                         nn.Sequential(ResBlock(384,384,norm_layer=nn.BatchNorm2d),ResBlockDecoder(384,64,norm_layer=nn.BatchNorm2d)),
                         nn.Sequential(ResBlock(192,192,norm_layer=nn.BatchNorm2d),ResBlockDecoder(192,64,norm_layer=nn.BatchNorm2d))]
        self.out = Output(64,3)

        self.flownet = Stage_2_generator(input_dim_1=20)
        self.encode1 = nn.Sequential(*self.encode1)
        init_weights(self.encode1,'normal')
        self.encode2 = nn.Sequential(*self.encode2)
        init_weights(self.encode2, 'normal')
        self.encode3 = nn.Sequential(*self.encode3)
        init_weights(self.encode3,'normal')
        self.upSample = nn.Sequential(*self.upSample)
        init_weights(self.upSample,'normal')
        init_weights(self.out,'normal')

    def warp(self,x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()
        # print(torch.max(flo), torch.min(flo))
        # print(flo[0])
        vgrid = grid + flo
        #
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        # print(torch.max(vgrid), torch.min(vgrid))
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        # mask = torch.ones(x.size()).cuda()
        # mask = nn.functional.grid_sample(mask, vgrid)
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        return output

    def load_flownet(self,checkpoint):
        self.flownet.load_state_dict(checkpoint['G'])
        self.flownet.eval()

    def forward(self, x1, x2, x3, t_parsing, s_parsing):
        with torch.no_grad():
            flow_list, r = self.flownet(t_parsing, s_parsing)
        E_feature_1 = []
        E_feature_2 = []
        E_feature_3 = []
        out_1 = x1
        out_2 = x2
        out_3 = x3
        for i in range(5):
            out_1 = self.encode1[i](out_1)
            out_2 = self.encode2[i](out_2)
            out_3 = self.encode3[i](out_3)
            E_feature_1.append(out_1)
            E_feature_2.append(out_2)
            E_feature_3.append(out_3)

        mix_feature = self.warp(E_feature_1[-1], flow_list[0].detach())
        up_feature = self.upSample[0](torch.cat((mix_feature,E_feature_2[-1],E_feature_3[-1]),dim=1))

        mix_feature = self.warp(E_feature_1[-2], flow_list[1].detach())
        up_feature = self.upSample[1](torch.cat((mix_feature,up_feature,E_feature_3[-2]),dim=1))

        mix_feature = self.warp(E_feature_1[-3], flow_list[2].detach())
        up_feature = self.upSample[2](torch.cat((mix_feature,up_feature,E_feature_3[-3]),dim=1))

        mix_feature = self.warp(E_feature_1[-4],  flow_list[3].detach())
        up_feature = self.upSample[3](torch.cat((mix_feature,up_feature,E_feature_3[-4]),dim=1))

        mix_feature = self.warp(E_feature_1[-5], flow_list[4].detach())
        up_feature = self.upSample[4](torch.cat((mix_feature,up_feature,E_feature_3[-5]),dim=1))
        img = self.out(up_feature)
        return img,flow_list



