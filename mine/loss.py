import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
# from block_extractor.block_extractor   import BlockExtractor
# from local_attn_reshape.local_attn_reshape   import LocalAttnReshape
# from resample2d_package.resample2d import Resample2d
class PerceptualAlign(nn.Module):
    r"""
    """

    def __init__(self, layer=['rel1_1','relu2_1','relu3_1','relu4_1','relu5_1']):
        super(PerceptualAlign, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer
        self.eps=1e-8
        self.resample = Resample2d(4, 1, sigma=2)
        self.criterion = nn.L1Loss()
    def __call__(self, target, source, flow_list, used_layers, mask=None, use_bilinear_sampling=False):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)



        return loss

    def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        # maps = F.interpolate(maps, [h,w]).view(b,-1)
        flow = F.interpolate(flow, [h,w])

        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow)
        else:
            input_sample = self.resample(source_vgg, flow)
        loss = self.criterion(target_vgg,input_sample)
        # loss=0


        # print(correction_sample[0,2076:2082])
        # print(correction_max[0,2076:2082])
        # coor_x = [32,32]
        # coor = max_indices[0,32+32*64]
        # coor_y = [int(coor%64), int(coor/64)]
        # source = F.interpolate(self.source, [64,64])
        # target = F.interpolate(self.target, [64,64])
        # source_i = source[0]
        # target_i = target[0]

        # source_i = source_i.view(3, -1)
        # source_i[:,coor]=-1
        # source_i[0,coor]=1
        # source_i = source_i.view(3,64,64)
        # target_i[:,32,32]=-1
        # target_i[0,32,32]=1
        # lists = str(int(torch.rand(1)*100))
        # img_numpy = util.tensor2im(source_i.data)
        # util.save_image(img_numpy, 'source'+lists+'.png')
        # img_numpy = util.tensor2im(target_i.data)
        # util.save_image(img_numpy, 'target'+lists+'.png')
        return loss

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample

class PerceptualCorrectness(nn.Module):
    r"""
    """
    def __init__(self, layer=['rel1_1','relu2_1','relu3_1','relu4_1','relu5_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer
        self.eps=1e-8
        self.resample = Resample2d(4, 1, sigma=2)

    def __call__(self, target, source, flow_list, used_layers, mask=None, use_bilinear_sampling=False):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)
        return loss

    def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        # maps = F.interpolate(maps, [h,w]).view(b,-1)
        flow = F.interpolate(flow, [h,w])

        target_all = target_vgg.view(b, c, -1)                      #[b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]


        source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps)
        target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
        correction = torch.bmm(source_norm, target_norm)
        # try:
        #     correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
        # except:
        #     print("An exception occurred")
        #     print(source_norm.shape)
        #     print(target_norm.shape)
        (correction_max,max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)
        correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
        loss_map = torch.exp(-correction_sample/(correction_max+self.eps))
        if mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            mask=F.interpolate(mask, size=(target_vgg.size(2), target_vgg.size(3)))
            mask=mask.view(-1, target_vgg.size(2)*target_vgg.size(3))
            loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
            loss = torch.sum(mask * loss_map)/(torch.sum(mask)+self.eps)

        # print(correction_sample[0,2076:2082])
        # print(correction_max[0,2076:2082])
        # coor_x = [32,32]
        # coor = max_indices[0,32+32*64]
        # coor_y = [int(coor%64), int(coor/64)]
        # source = F.interpolate(self.source, [64,64])
        # target = F.interpolate(self.target, [64,64])
        # source_i = source[0]
        # target_i = target[0]

        # source_i = source_i.view(3, -1)
        # source_i[:,coor]=-1
        # source_i[0,coor]=1
        # source_i = source_i.view(3,64,64)
        # target_i[:,32,32]=-1
        # target_i[0,32,32]=1
        # lists = str(int(torch.rand(1)*100))
        # img_numpy = util.tensor2im(source_i.data)
        # util.save_image(img_numpy, 'source'+lists+'.png')
        # img_numpy = util.tensor2im(target_i.data)
        # util.save_image(img_numpy, 'target'+lists+'.png')
        return loss

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

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


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss

backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            # self.loss = nn.BCELoss()
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                # print(pred)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)



class Regularization(torch.nn.Module):
    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.intUnfold = [ 3,3,5,5 ][intLevel]

        if intLevel >= 5:
            self.netFeat = torch.nn.Sequential()

        elif intLevel < 5:
            self.netFeat = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=[256,196,128,64 ][intLevel], out_channels=128, kernel_size=1,
                                stride=1, padding=0),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        # end

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=[ 131,131,131,131][intLevel], out_channels=128, kernel_size=3,
                            stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if intLevel >= 5:
            self.netDist = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=[ 9,9,25,25 ][intLevel],
                                kernel_size=[3,3,5,5][intLevel], stride=1,
                                padding=[ 1,1,2, 2][intLevel])
            )

        elif intLevel < 5:
            self.netDist = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=[ 9,9,25,25 ][intLevel],
                                kernel_size=([3,3,5,5][intLevel], 1), stride=1,
                                padding=([ 1,1,2, 2][intLevel], 0)),
            )

        # end

        self.netScaleX = torch.nn.Conv2d(in_channels=[ 9,9,25, 25 ][intLevel], out_channels=1, kernel_size=1,
                                         stride=1, padding=0)
        self.netScaleY = torch.nn.Conv2d(in_channels=[ 9,9,25, 25 ][intLevel], out_channels=1, kernel_size=1,
                                         stride=1, padding=0)

    # eny

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        tenDifference = (tenFirst - backwarp(tenInput=tenSecond, tenFlow=tenFlow)).pow(2.0).sum(1,
                                                                                         True).sqrt().detach()
        tenDist = self.netDist(self.netMain(torch.cat([tenDifference,
                                                       tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2,
                                                                                                            True).view(
                                                           tenFlow.shape[0], 2, 1, 1), self.netFeat(tenFeaturesFirst)],
                                                      1)))
        tenDist = tenDist.pow(2.0).neg()
        tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

        tenDivisor = tenDist.sum(1, True).reciprocal()

        tenScaleX = self.netScaleX(
            tenDist * torch.nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1,
                                                 padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
        tenScaleY = self.netScaleY(
            tenDist * torch.nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1,
                                                 padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
        return torch.cat([tenScaleX, tenScaleY], 1)

def get_row(coor, num):
    sec_dic = []
    for j in range(num):
        sum = 0
        buffer = 0
        flag = False
        max = -1
        for i in range(num - 1):
            differ = (coor[:, j * num + i + 1, :] - coor[:, j * num + i, :]) ** 2
            if not flag:
                second_dif = 0
                flag = True
            else:
                second_dif = torch.abs(differ - buffer)
                sec_dic.append(second_dif)

            buffer = differ
            sum += second_dif
    return torch.stack(sec_dic, dim=1)

def get_col(coor, num):
    sec_dic = []
    for i in range(num):
        sum = 0
        buffer = 0
        flag = False
        max = -1
        for j in range(num - 1):
            differ = (coor[:, (j + 1) * num + i, :] - coor[:, j * num + i, :]) ** 2
            if not flag:
                second_dif = 0
                flag = True
            else:
                second_dif = torch.abs(differ - buffer)
                sec_dic.append(second_dif)
            buffer = differ
            sum += second_dif
    return torch.stack(sec_dic, dim=1)

def grad_row(coor, num):
    sec_term = []
    for j in range(num):
        for i in range(1, num - 1):
            x0, y0 = coor[:, j * num + i - 1, :][0]
            x1, y1 = coor[:, j * num + i + 0, :][0]
            x2, y2 = coor[:, j * num + i + 1, :][0]
            grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
            sec_term.append(grad)
    return sec_term

def grad_col(coor, num):
    sec_term = []
    for i in range(num):
        for j in range(1, num - 1):
            x0, y0 = coor[:, (j - 1) * num + i, :][0]
            x1, y1 = coor[:, j * num + i, :][0]
            x2, y2 = coor[:, (j + 1) * num + i, :][0]
            grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
            sec_term.append(grad)
    return sec_term

def erjie_loss(flow):
    b,c,h,w = flow.size()
    flow = flow.permute(0,2,3,1)
    flow = flow.reshape(b,-1,c)
    row = get_row(flow,h)
    col = get_col(flow,w)
    rg_loss = sum(grad_row(flow, h))
    cg_loss = sum(grad_col(flow, w))
    rg_loss = torch.max(rg_loss, torch.tensor(0.02).cuda())
    cg_loss = torch.max(cg_loss, torch.tensor(0.02).cuda())
    rx, ry, cx, cy = torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda() \
        , torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda()
    row_x, row_y = row[:, :, 0], row[:, :, 1]
    col_x, col_y = col[:, :, 0], col[:, :, 1]
    rx_loss = torch.max(rx, row_x).mean()
    ry_loss = torch.max(ry, row_y).mean()
    cx_loss = torch.max(cx, col_x).mean()
    cy_loss = torch.max(cy, col_y).mean()

    return rx_loss + ry_loss + cx_loss + cy_loss + rg_loss + cg_loss

def get_A(bs, H, W):
    A = np.array([[[1,0,0],[0,1,0]]]).astype(np.float32)
    A = np.concatenate([A]*bs,0)
    A = torch.from_numpy(A)
    net = nn.functional.affine_grid(A,(bs,2,H,W)).cuda()
    net = net.transpose(2,3).transpose(1,2)
    return net

def loss_smt(mat):
    return (torch.sum(torch.abs(mat[:, :, :, 1:] - mat[:, :, :, :-1]))/mat.shape[2] + \
		    torch.sum(torch.abs(mat[:, :, 1:, :] - mat[:, :, :-1, :]))/mat.shape[3]) /mat.shape[0]

def tv_loss(flow_list):
    loss = 0
    for flow in flow_list:
        loss += loss_smt(flow-get_A(flow.shape[0], flow.shape[2], flow.shape[3]))
    return loss

def get_row(coor, num):
    sec_dic = []
    for j in range(num):
        sum = 0
        buffer = 0
        flag = False
        max = -1
        for i in range(num - 1):
            differ = (coor[:, j * num + i + 1, :] - coor[:, j * num + i, :]) ** 2
            if not flag:
                second_dif = 0
                flag = True
            else:
                second_dif = torch.abs(differ - buffer)
                sec_dic.append(second_dif)

            buffer = differ
            sum += second_dif
    return torch.stack(sec_dic, dim=1)

def get_col(coor, num):
    sec_dic = []
    for i in range(num):
        sum = 0
        buffer = 0
        flag = False
        max = -1
        for j in range(num - 1):
            differ = (coor[:, (j + 1) * num + i, :] - coor[:, j * num + i, :]) ** 2
            if not flag:
                second_dif = 0
                flag = True
            else:
                second_dif = torch.abs(differ - buffer)
                sec_dic.append(second_dif)
            buffer = differ
            sum += second_dif
    return torch.stack(sec_dic, dim=1)

def grad_row(coor, num):
    sec_term = []
    for j in range(num):
        for i in range(1, num - 1):
            x0, y0 = coor[:, j * num + i - 1, :][0]
            x1, y1 = coor[:, j * num + i + 0, :][0]
            x2, y2 = coor[:, j * num + i + 1, :][0]
            grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
            sec_term.append(grad)
    return sec_term

def grad_col(coor, num):
    sec_term = []
    for i in range(num):
        for j in range(1, num - 1):
            x0, y0 = coor[:, (j - 1) * num + i, :][0]
            x1, y1 = coor[:, j * num + i, :][0]
            x2, y2 = coor[:, (j + 1) * num + i, :][0]
            grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
            sec_term.append(grad)
    return sec_term

def patch_erjie_loss(flow,kernel_size=5,stride=2,padding=2):
    b,c,h,w = flow.size()
    patch_flow = torch.nn.functional.unfold(flow,kernel_size=kernel_size,stride=stride,padding=padding)
    patch_flow = patch_flow.reshape(b,c,kernel_size,kernel_size,-1)
    patch_flow = patch_flow.permute(0,4,1,2,3)
    loss = 0
    for flow in patch_flow:
        loss +=erjie_loss(flow)
    return loss/b

def erjie_loss(flow):
    b,c,h,w = flow.size()
    flow = flow.permute(0,2,3,1)
    flow = flow.reshape(b,-1,c)
    row = get_row(flow,h)
    col = get_col(flow,w)
    rg_loss = sum(grad_row(flow, h))
    cg_loss = sum(grad_col(flow, w))
    rg_loss = torch.max(rg_loss, torch.tensor(0.02).cuda())
    cg_loss = torch.max(cg_loss, torch.tensor(0.02).cuda())
    rx, ry, cx, cy = torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda() \
        , torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda()
    row_x, row_y = row[:, :, 0], row[:, :, 1]
    col_x, col_y = col[:, :, 0], col[:, :, 1]
    rx_loss = torch.max(rx, row_x).mean()
    ry_loss = torch.max(ry, row_y).mean()
    cx_loss = torch.max(cx, col_x).mean()
    cy_loss = torch.max(cy, col_y).mean()

    return (rx_loss + ry_loss + cx_loss + cy_loss + rg_loss + cg_loss)

class StyleLoss(nn.Module):
  r"""
  Perceptual loss, VGG-based
  https://arxiv.org/abs/1603.08155
  https://github.com/dxyang/StyleTransfer/blob/master/utils.py
  """

  def __init__(self):
    super(StyleLoss, self).__init__()
    self.add_module('vgg', VGG19())
    self.criterion = torch.nn.L1Loss()

  def compute_gram(self, x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)

    return G

  def __call__(self, x, y):
    # Compute features
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)

    # Compute loss
    style_loss = 0.0
    style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

    return style_loss


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