import argparse
from mine.dataset_viton import DataSet
import torch
import os
from mine.utils import flowtoimg
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from torch.utils.tensorboard import SummaryWriter
from mine.network_stage_2_mine_x2_resflow import Stage_2_generator
from mine.loss import VGGLoss
import torch.nn as nn
from mine.visualization import board_add_images
# 没有 target_posea'a
def get_opt():
    parser =argparse.ArgumentParser()
    parser.add_argument("--mode",default='train')
    parser.add_argument("--point_num",default=18)
    parser.add_argument("--data_root",default='./dataset/viton_resize')
    parser.add_argument("--file_path", default='train_pairs.txt')
    parser.add_argument("--batch_size",default=8)
    parser.add_argument('--epochs',default=100)
    parser.add_argument('--ndf',default=64)
    parser.add_argument('--n_layers',default=2)
    parser.add_argument('--num_D',default=3)
    parser.add_argument('--getIntermFeat',default=False)
    parser.add_argument('--tensorboard_dir',default='./tensorboard/stage2')
    parser.add_argument('--model_dir',default='./model/stage_2')
    parser.add_argument('--display_count',default=50)
    parser.add_argument('--save_count',default=1)
    parser.add_argument('--image_size', type=int, default=256, help='input image size')

    opt = parser.parse_args()
    return opt

def save_checkpoint(mode,model_g, save_path,epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path,mode+'_model_'+str(epoch))
    torch.save({'G':model_g.state_dict()}, save_file)

criterion_L1 = nn.L1Loss()
criterion_Vgg = VGGLoss().cuda()
def struct_loss(source_parsing,target_parsing,warp_parsing):
    B,C,H,W = source_parsing.size()
    loss = 0
    for i in range(B):
        for j in range(20):
            if torch.sum(source_parsing[i][j]) >0 and torch.sum(target_parsing[i][j]) >0:
                loss = loss + criterion_L1(warp_parsing[i][j].reshape(1,1,H,W),target_parsing[i][j].reshape(1,1,H,W))
    loss = loss/B
    return loss

def all_struct_loss(source_parsing,target_parsing,warp_parsing):
    B,C,H,W = source_parsing.size()
    loss = 0
    for i in range(B):
        for j in range(20):
            if torch.sum(source_parsing[i][j]) >0 and torch.sum(target_parsing[i][j]) >0:
                loss = loss + criterion_L1(warp_parsing[i][j].reshape(1,1,H,W),target_parsing[i][j].reshape(1,1,H,W))
    loss = loss/B
    return loss

def all_struct_loss(prev_images, next_images, flow_dict):
    loss = 0.
    loss_weight_sum = 0.

    for i in range(len(flow_dict)):
        height = flow_dict[i].shape[2]
        width  = flow_dict[i].shape[3]
        prev_images = F.interpolate(prev_images,size=(height,width))
        next_images = F.interpolate(next_images,size=(height,width))
        for image_num in range(prev_images.shape[0]):
            flow = flow_dict[i][image_num]

            prev_images_resize = prev_images[image_num]
            next_images_resize = next_images[image_num]
            next_images_warped = warp(next_images_resize.reshape(1,20,height,width), flow.reshape(1,2,height,width))
            for j in range(20):
                if torch.sum(next_images_warped[0][j]) > 0 and torch.sum(prev_images_resize[j]) > 0:

                    loss = loss + criterion_L1(next_images_warped[0][j].reshape(1, 1, height, width),
                                               prev_images_resize[j].reshape(1, 1, height, width))

            # distance = next_images_warped - prev_images_resize
            # photometric_loss = charbonnier_loss(distance)
            # total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    loss /= loss_weight_sum

    return loss
def roi_loss(source_parsing,target_parsing,warp_parsing,target_img,warp_img):
    B, C, H, W = source_parsing.size()
    loss = 0
    for i in range(B):
        for j in range(20):
            if torch.sum(source_parsing[i][j]) > 0 and torch.sum(target_parsing[i][j]) > 0:
                loss = loss + criterion_Vgg((warp_img[i]*warp_parsing[i][j]).reshape(1,3,H,W),(target_img[i]*target_parsing[i][j]).reshape(1,3,H,W))
                loss = loss + criterion_L1((warp_img[i]*warp_parsing[i][j]).reshape(1,3,H,W),(target_img[i]*target_parsing[i][j]).reshape(1,3,H,W))
    return loss/B
def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]
# def TVLoss(input):
#     loss = 0
#     for i,x in enumerate(input):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = _tensor_size(x[:, :, 1:, :])
#         count_w = _tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         loss += 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#     return loss
def TVLoss(input):
    loss = 0
    ahp = [2,2,2,2,2]
    for i,x in enumerate(input):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = _tensor_size(x[:, :, 1:, :])
        count_w = _tensor_size(x[:, :, :, 1:])
        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])).sum()
        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1])).sum()
        loss += (h_tv / count_h + w_tv / count_w) / batch_size
    return loss

def get_grid(x):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().cuda()
    # print(flo[0])
    vgrid = grid + x
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    return vgrid

def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss

def smooth_loss(flow_list):
    loss = 0
    for flow in flow_list:
        loss += compute_smoothness_loss(flow)
    return loss

def compute_photometric_loss(prev_images, next_images, flow_dict):
    """
    Multi-scale photometric loss, as defined in equation (3) of the paper.
    """
    total_photometric_loss = 0.
    loss_weight_sum = 0.
    for i in range(len(flow_dict)):
        for image_num in range(prev_images.shape[0]):
            flow = flow_dict[i][image_num]
            height = flow.shape[1]
            width = flow.shape[2]

            prev_images_resize = FF.to_tensor(FF.resize(FF.to_pil_image(prev_images[image_num].cpu()),
                                                      [height, width])).cuda()
            next_images_resize = FF.to_tensor(FF.resize(FF.to_pil_image(next_images[image_num].cpu()),
                                                      [height, width])).cuda()
            next_images_warped = warp(next_images_resize.reshape(1,3,height,width), flow.reshape(1,2,height,width))

            distance = next_images_warped - prev_images_resize
            photometric_loss = charbonnier_loss(distance)
            total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum

    return total_photometric_loss

def compute_smoothness_loss(flow):
    """
    Local smoothness loss, as defined in equation (5) of the paper.
    The neighborhood here is defined as the 8-connected region around each pixel.
    """
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]

    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                      charbonnier_loss(flow_ucrop - flow_dcrop) + \
                      charbonnier_loss(flow_ulcrop - flow_drcrop) + \
                      charbonnier_loss(flow_dlcrop - flow_urcrop)
    smoothness_loss /=flow.shape[0]

    return smoothness_loss

def warp(x, flo):
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

def sequence_loss(flow_list,s_parsing,t_parsing,t_img,s_img):
    loss_vgg = 0
    loss_struct = 0
    struct_theta = [10]
    for i in range(len(flow_list)):
        flow = flow_list[i]
        warp_parsing = warp(s_parsing,flow)
        warp_img = warp(s_img,flow)
        loss_vgg += roi_loss(s_parsing, t_parsing, warp_parsing, t_img, warp_img)
        loss_struct += struct_theta[i]*struct_loss(s_parsing, t_parsing, warp_parsing)
    return loss_vgg,loss_struct


opt = get_opt()
dataSet = DataSet(opt)
data_loader = torch.utils.data.DataLoader(dataSet, batch_size=opt.batch_size, pin_memory=True,shuffle=True,num_workers=8)

max_iter = 0
if not os.path.exists(os.path.dirname(opt.tensorboard_dir)):
    os.makedirs(os.path.dirname(opt.tensorboard_dir))
board = SummaryWriter(opt.tensorboard_dir)

generator_stage1_1 = Stage_2_generator(input_dim_1=20).cuda()
optimerG_1 = torch.optim.Adam(params=generator_stage1_1.parameters(),lr=0.00005,betas=(0.5,0.999))

for epoch in range(opt.epochs):
    for iteration,item in enumerate(data_loader):
        if iteration >max_iter:
            max_iter = iteration

        source_cloth_parsing = item['source_cloth_parsing'].cuda()
        target_cloth_parsing = item['target_cloth_parsing'].cuda()
        source_cloth_im = item['source_cloth_im'].cuda()
        target_cloth_im = item['target_cloth_im'].cuda()
        input_1_up = target_cloth_parsing
        input_1_down = source_cloth_parsing
        flow_list,res_flowlistst = generator_stage1_1(input_1_up,input_1_down)

        warp_parsing = warp(source_cloth_parsing, flow_list[-1])
        warp_img = warp(source_cloth_im, flow_list[-1])
        loss_Vgg = criterion_Vgg(warp_img, target_cloth_im)
        loss_photo = compute_photometric_loss(target_cloth_im, source_cloth_im, flow_list)
        loss_tv = TVLoss(res_flowlistst)
        lossG_all1 = loss_tv + 5 * loss_photo + 2 * loss_Vgg
        optimerG_1.zero_grad()
        lossG_all1.backward()
        optimerG_1.step()
        board.add_scalars('loss_1', {'loss_tv': loss_tv, 'loss_photo': loss_photo, 'loss_vgg': loss_Vgg},
                          global_step=max_iter * epoch + iteration)

        if (iteration + 1) % 10 == 0:
            with torch.no_grad():
                flow_png = flowtoimg(F.interpolate(flow_list[-2].detach(), size=(256, 192), mode='bilinear'))
                flow_png3 = flowtoimg(F.interpolate(flow_list[-3].detach(), size=(256, 192), mode='bilinear') )
                flow_png4 = flowtoimg(F.interpolate(flow_list[-4].detach(), size=(256, 192), mode='bilinear') )
                flow_png5 = flowtoimg(F.interpolate(flow_list[-5].detach(), size=(256, 192), mode='bilinear') )
            visuals = [[source_cloth_parsing, target_cloth_parsing, warp_parsing],
                       [flow_png,flow_png3,flow_png4,flow_png5],
                       [source_cloth_im,target_cloth_im,warp_img]]
            board_add_images(board, 'combine', visuals, max_iter * epoch + iteration)
    if (epoch + 1) % opt.save_count == 0:
        save_checkpoint('1', generator_stage1_1, opt.model_dir, epoch)
        print('save')


