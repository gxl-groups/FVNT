import torch
import torchvision
import torch.nn as nn
torchvision.models.ResNet
torch.set_printoptions(profile="full")
import torchvision.models as models
import collections
import functools
import numpy as np
import torch.nn.functional as F
from Deformable.modules import DeformConvPack
def weight_init(m):
    class_name = m.__class__.__name__
    if isinstance(m, nn.Conv2d) :
        # print(m,m.weight.data.shape)
        nn.init.kaiming_normal_(m.weight.data,a=1,mode='fan_in',nonlinearity='leaky_relu')
        # m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m,nn.ConvTranspose2d):
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)
# def weight_init(m):
#     if isinstance(m, nn.Conv2d):
#         #
#         m.weight.data.normal_(0.0, 0.02)
#         m.bias.data.zero_()
def get_norm_layer(norm_type='instance'):
    if norm_type=='batch':
        norm_layer = functools.partial(nn.BatchNorm2d,affine=True)
    elif norm_type=='instance':
        norm_layer = functools.partial(nn.InstanceNorm2d,affine=False)
    return norm_layer

def conv3X3(input_dim,output_dim,stride=1,padding=1):
    return nn.Conv2d(input_dim,output_dim,kernel_size=3,stride=stride,padding=padding,bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            nn.BatchNorm2d(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))



class ResnetBlock(nn.Module):
    def __init__(self,input_dim,output_dim,stride=1,padding=1,kernel_size=3,bias=True,norm_layer=nn.BatchNorm2d,downSample=False):
        super(ResnetBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim,out_channels=output_dim,kernel_size=kernel_size,padding=padding,stride=stride,bias=bias)
        self.bn1 = norm_layer(output_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels=output_dim,out_channels=output_dim,kernel_size=kernel_size,padding=padding,stride=1,bias=bias)
        self.bn2 = norm_layer(output_dim)
        self.relu2 = nn.LeakyReLU(0.1)
        self.downSample = None
        if downSample == True:
            self.downSample = nn.Sequential(conv1x1(input_dim,output_dim,stride=stride))

    def forward(self, input):
        identity = input
        output = self.conv1(input)
        # output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        # output = self.bn2(output)
        output = self.relu(output)
        if self.downSample != None:
            identity = self.downSample(input)
        output = identity + output

        return output

class Stage_2_generator(nn.Module):
    def __init__(self, input_dim_1,n_down=3,norm_layer=nn.BatchNorm2d):
        super(Stage_2_generator, self).__init__()
        self.encode = []
        self.n_down = n_down
        self.encode = [nn.Sequential(
            nn.Conv2d(in_channels=input_dim_1, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
             nn.LeakyReLU(0.1)
            , ResnetBlock(input_dim=64, output_dim=64))
            , nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
                             nn.LeakyReLU(0.1)
                            , ResnetBlock(input_dim=128, output_dim=128))
            , nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
                            nn.LeakyReLU(0.1)
                            , ResnetBlock(input_dim=256, output_dim=256))
            , nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
                            nn.LeakyReLU(0.1)
                            , ResnetBlock(input_dim=256, output_dim=256))
            , nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
                            nn.LeakyReLU(0.1)
                            , ResnetBlock(input_dim=256, output_dim=256))
                       ]
        self.M = []
        # self.M.append(ModulatedDeformConvPack(512, 2, 3, 1, 1, True,lr_mult=0.00005))
        self.M.append(DeformConvPack(512, 2, 3, 1, 1,True,lr_mult=0.00005))

        self.cg_channel = [
            nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            ]
        self.cg_channel2 = [
            nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))
            , nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True))]

        self.encode = nn.Sequential(*self.encode)
        self.encode.apply(weight_init)

        self.M = nn.Sequential(*self.M)
        self.M.apply(weight_init)

        self.cg_channel = nn.Sequential(*self.cg_channel)
        self.cg_channel.apply(weight_init)
        self.cg_channel2 = nn.Sequential(*self.cg_channel2)
        self.cg_channel2.apply(weight_init)




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
        mask = torch.ones(x.size()).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, mode='bilinear')
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output

    def forward_M(self, input_1, input_2, i, flow=0):
        input = torch.cat((input_1, input_2), dim=1)
        res_flow = self.M[0](input)
        return res_flow

    def forward(self, input_1, input_2):
        E_feature_1 = []
        E_feature_2 = []
        out_1 = input_1
        out_2 = input_2
        for i in range(5):
            out_1 = self.encode[i](out_1)
            out_2 = self.encode[i](out_2)
            E_feature_1.append(out_1)
            E_feature_2.append(out_2)

        l5_feature_1 = E_feature_1[-1]
        l5_feature_1 = self.cg_channel[0](l5_feature_1)
        l5_feature_2 = E_feature_2[-1]
        l5_feature_2 = self.cg_channel2[0](l5_feature_2)
        flow_l5 = self.forward_M(l5_feature_1, l5_feature_2, 0, 0)

        flow_l4 = F.interpolate(flow_l5, scale_factor=2, mode='bilinear')*2
        up_l4_feature_1 = F.interpolate(l5_feature_1, scale_factor=2, mode='bilinear')
        up_l4_feature_2 = F.interpolate(l5_feature_2, scale_factor=2, mode='bilinear')
        l4_feature_1 = self.cg_channel[1](E_feature_1[-2]) + up_l4_feature_1
        l4_feature_2 = self.cg_channel2[1](E_feature_2[-2]) + up_l4_feature_2
        res_flowl4 = self.forward_M(l4_feature_1, self.warp(l4_feature_2, flow_l4), 1)
        flow_l4 = flow_l4 + res_flowl4

        flow_l3 = F.interpolate(flow_l4, scale_factor=2, mode='bilinear')*2
        up_l3_feature_1 = F.interpolate(l4_feature_1, scale_factor=2, mode='bilinear')
        up_l3_feature_2 = F.interpolate(l4_feature_2, scale_factor=2, mode='bilinear')
        l3_feature_1 = self.cg_channel[2](E_feature_1[-3]) + up_l3_feature_1
        l3_feature_2 = self.cg_channel2[2](E_feature_2[-3]) + up_l3_feature_2
        res_flowl3 = self.forward_M(l3_feature_1, self.warp(l3_feature_2, flow_l3), 2)
        flow_l3 = flow_l3 + res_flowl3

        flow_l2 = F.interpolate(flow_l3, scale_factor=2, mode='bilinear')*2
        up_l2_feature_1 = F.interpolate(l3_feature_1, scale_factor=2, mode='bilinear')
        up_l2_feature_2 = F.interpolate(l3_feature_2, scale_factor=2, mode='bilinear')
        l2_feature_1 = self.cg_channel[3](E_feature_1[-4]) + up_l2_feature_1
        l2_feature_2 = self.cg_channel2[3](E_feature_2[-4]) + up_l2_feature_2
        res_flowl2 = self.forward_M(l2_feature_1, self.warp(l2_feature_2, flow_l2), 3)
        flow_l2 = flow_l2 + res_flowl2

        flow_l1 = F.interpolate(flow_l2, scale_factor=2, mode='bilinear')*2
        up_l1_feature_1 = F.interpolate(l2_feature_1, scale_factor=2, mode='bilinear')
        up_l1_feature_2 = F.interpolate(l2_feature_2, scale_factor=2, mode='bilinear')
        l1_feature_1 = self.cg_channel[4](E_feature_1[-5]) + up_l1_feature_1
        l1_feature_2 = self.cg_channel2[4](E_feature_2[-5]) + up_l1_feature_2
        res_flowl1 = self.forward_M(l1_feature_1, self.warp(l1_feature_2, flow_l1), 4)
        flow_l1 = flow_l1 + res_flowl1

        flow = F.interpolate(flow_l1, scale_factor=2, mode='bilinear')*2
        flow_list = [flow_l5, flow_l4, flow_l3, flow_l2, flow_l1, flow]
        res_flowlist = [flow_l5,res_flowl4,res_flowl3,res_flowl2,res_flowl1]
        return flow_list,res_flowlist

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class PerceptualCorrectness(nn.Module):
    """
    """
    def __init__(self, layer='relu3_1'):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer
        self.eps = 1e-8

    def warp(self,x,flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()

        vgrid = grid + flo
        # vgrid = flo - grid
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        # print(torch.max(vgrid),torch.min(vgrid))
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid,mode='bilinear')
        mask = torch.ones(x.size()).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output

    def __call__(self, gts, inputs, flow):
        gts_vgg, inputs_vgg = self.vgg(gts), self.vgg(inputs)
        gts_vgg = gts_vgg[self.layer]
        inputs_vgg = inputs_vgg[self.layer]
        [b, c, h, w] = gts_vgg.shape

        flow = F.adaptive_avg_pool2d(flow, [h, w])

        gts_all = gts_vgg.view(b, c, -1)  # [b C N2]
        inputs_all = inputs_vgg.view(b, c, -1).transpose(1, 2)  # [b N2 C]

        input_norm = inputs_all / (inputs_all.norm(dim=2, keepdim=True) + self.eps)
        gt_norm = gts_all / (gts_all.norm(dim=1, keepdim=True) + self.eps)
        correction = torch.bmm(input_norm, gt_norm)  # [b N2 N2]
        (correction_max, max_indices) = torch.max(correction, dim=1)

        # interple with gaussian sampling
        input_sample = self.warp(inputs_vgg, flow)
        input_sample = input_sample.view(b, c, -1)  # [b C N2]

        correction_sample = F.cosine_similarity(input_sample, gts_all)  # [b 1 N2]
        loss_map = torch.exp(-correction_sample / (correction_max + self.eps))
        loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))

        return loss
class AffineRegularizationLoss(nn.Module):
    """docstring for AffineRegularizationLoss"""
    # kernel_size: kz
    def __init__(self, kz):
        super(AffineRegularizationLoss, self).__init__()
        self.kz = kz
        self.criterion = torch.nn.L1Loss()
        self.extractor = BlockExtractor(kernel_size=kz)
        self.reshape = LocalAttnReshape()

        temp = np.arange(kz)
        A = np.ones([kz*kz, 3])
        A[:, 0] = temp.repeat(kz)
        A[:, 1] = temp.repeat(kz).reshape((kz,kz)).transpose().reshape(kz**2)
        AH = A.transpose()
        k = np.dot(A, np.dot(np.linalg.inv(np.dot(AH, A)), AH)) - np.identity(kz**2) #K = (A((AH A)^-1)AH - I)
        self.kernel = np.dot(k.transpose(), k)
        self.kernel = torch.from_numpy(self.kernel).unsqueeze(1).view(kz**2, kz, kz).unsqueeze(1)

    def __call__(self, flow_fields):
        grid = self.flow2grid(flow_fields)

        grid_x = grid[:,0,:,:].unsqueeze(1)
        grid_y = grid[:,1,:,:].unsqueeze(1)
        weights = self.kernel.type_as(flow_fields)
        loss_x = self.calculate_loss(grid_x, weights)
        loss_y = self.calculate_loss(grid_y, weights)
        return loss_x+loss_y


    def calculate_loss(self, grid, weights):
        results = nn.functional.conv2d(grid, weights)   # KH K B [b, kz*kz, w, h]
        b, c, h, w = results.size()
        kernels_new = self.reshape(results, self.kz)
        f = torch.zeros(b, 2, h, w).type_as(kernels_new) + float(int(self.kz/2))
        grid_H = self.extractor(grid, f)
        result = torch.nn.functional.avg_pool2d(grid_H*kernels_new, self.kz, self.kz)
        loss = torch.mean(result)*self.kz**2
        return loss

    def flow2grid(self, flow_field):
        b,c,h,w = flow_field.size()
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(flow_field).float()
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(flow_field).float()
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        return flow_field+grid


class MultiAffineRegularizationLoss(nn.Module):
    def __init__(self, kz_dic):
        super(MultiAffineRegularizationLoss, self).__init__()
        self.kz_dic = kz_dic
        self.method_dic = {}
        for key in kz_dic:
            instance = AffineRegularizationLoss(kz_dic[key])
            self.method_dic[key] = instance
        self.layers = sorted(kz_dic, reverse=True)

    def __call__(self, flow_fields):
        loss = 0
        for i in range(len(flow_fields)):
            method = self.method_dic[self.layers[i]]
            loss += method(flow_fields[i])
        return loss