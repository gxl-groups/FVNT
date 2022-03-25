import argparse
from mine.dataset_viton_stage3_nomask_addpants import DataSet
import torch
from torch.utils.tensorboard import SummaryWriter
from mine.visualization import save_images
import torch.nn as nn
from mine.visualization import tensor_list_for_board
import torch.nn.functional as F
from mine.networks_mine_gan3_batch_2_multyscale import Generator
from mine.network_stage_1_mine_final_viton import define_G
# 没有 target_posea'a
def get_opt():
    parser =argparse.ArgumentParser()
    parser.add_argument("--mode",default='test')
    parser.add_argument("--point_num",default=18)
    parser.add_argument("--data_root",default='./dataset/viton_resize')
    parser.add_argument("--file_path", default='test_pairs.txt')
    parser.add_argument("--stage2_model", default='./model/stage2_model')
    parser.add_argument("--stage3_model", default='./model/stage3_model')
    parser.add_argument("--genetate_parsing",default='generate_parsing_cross')
    parser.add_argument("--result", default='./result')
    parser.add_argument("--batch_size",default=1)
    parser.add_argument('--image_size', type=int,default=256, help='input image size')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=192)

    opt = parser.parse_args()
    return opt

opt = get_opt()
dataSet = DataSet(opt)
data_loader = torch.utils.data.DataLoader(dataSet, batch_size=opt.batch_size, pin_memory=True,shuffle=False,num_workers=8)
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
    # mask = torch.ones(x.size()).cuda()
    # mask = nn.functional.grid_sample(mask, vgrid, mode='bilinear')
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1
    return output
cloth_list = [-1,-1,-1,-1,-1,5,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
parsing_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
def test_model_gan( model,dir=''):
    for iteration,item in enumerate(data_loader):
        target_pose_maps = item['target_pose_map'].cuda()
        target_pose = item['target_pose'].cuda()
        target_img = item['target_img'].cuda()
        target_parsing = item['target_parsing_20'].cuda()
        target_cloth_parsing = item['target_cloth_parsing'].cuda()
        source_cloth_parsing = item['source_cloth_parsing'].cuda()
        target_body = item['target_body'].cuda()
        target_face = item['target_face'].cuda()
        source_cloth_im = item['source_cloth_im'].cuda()
        input = torch.cat((target_parsing, target_pose_maps, target_body), dim=1)
        with torch.no_grad():
            fake, flow_list = model(source_cloth_im, input, target_face, target_cloth_parsing, source_cloth_parsing)
            warp_cloth = warp(source_cloth_im, flow_list[-1].detach())
            visuals = [[source_cloth_im, target_pose, target_body, target_face, warp_cloth, fake, target_img]]
            imgs = tensor_list_for_board(visuals)
        save_images(fake.cpu(), '{}'.format(iteration) + '.png', dir)

# model_stage_1 = define_G(42, 20, 64, 'unet_128', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02).cuda()
# # checkpoint = torch.load("/data2/wt/model/Access/viton/1_model_18")
# checkpoint = torch.load(opt.stage1_model)
# model_stage_1.load_state_dict(checkpoint['G'])
# model_stage_1.eval()
# model_stage_1.cuda()

model = Generator(3,41).cuda()
checkpoint2 = torch.load(opt.stage2_model)
checkpoint3 = torch.load(opt.stage3_model)
model.load_state_dict(checkpoint3['G'])
model.load_flownet(checkpoint2)
model.eval()
model.cuda()
test_model_gan(model,opt.result)


