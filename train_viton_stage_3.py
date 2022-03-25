import argparse
from mine.dataset_viton_stage3_addpants import DataSet
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from mine.networks_mine_gan3_batch_2_multyscale import Generator
import torch.nn.functional as F
from mine.loss import VGGLoss
from mine.loss import GANLoss
import torch.nn as nn
from mine.loss import StyleLoss
from torch.optim import lr_scheduler
def get_opt():
    parser =argparse.ArgumentParser()
    parser.add_argument("--mode",default='train')
    parser.add_argument("--point_num",default=18)
    parser.add_argument("--data_root", default='./dataset/viton_resize')
    parser.add_argument("--file_path", default='train_pairs.txt')
    parser.add_argument("--stage2_model", default='./model/stage2_model')
    parser.add_argument("--batch_size",default=8)
    parser.add_argument('--epochs',default=100)
    parser.add_argument('--ndf',default=64)
    parser.add_argument('--n_layers',default=2)
    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--num_D',default=3)
    parser.add_argument('--getIntermFeat',default=False)
    parser.add_argument('--tensorboard_dir',default='./tensorboard/stage3')
    parser.add_argument('--model_dir',default='./model/stage3')
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

opt = get_opt()
dataSet = DataSet(opt)
data_loader = torch.utils.data.DataLoader(dataSet, batch_size=opt.batch_size, pin_memory=True,shuffle=True,num_workers=8)
max_iter = 0

if not os.path.exists(os.path.dirname(opt.tensorboard_dir)):
    os.makedirs(os.path.dirname(opt.tensorboard_dir))
board = SummaryWriter(opt.tensorboard_dir)


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

model = Generator(3,41).cuda()
checkpoint = torch.load(opt.stage2_model)
model.load_flownet(checkpoint)
model.cuda()
# discriminator_stage1_1 = define_D(44,64,3,norm='batch', use_sigmoid=False,num_D=3).cuda()
optimerG = torch.optim.Adam(params=model.parameters(),lr=opt.lr,betas=(0.5,0.999))
# optimerD = torch.optim.Adam(params=discriminator_stage1_1.parameters(),lr=opt.lr,betas=(0.5,0.999))
scheduler_G = lr_scheduler.StepLR(optimerG,step_size=1,gamma = 0.9)
# scheduler_D = lr_scheduler.StepLR(optimerD,step_size=1,gamma = 0.9)
criterion_Vgg = VGGLoss().cuda()
criterion_L1 = nn.L1Loss()
criterion_Style = StyleLoss().cuda()
criteriion_G = GANLoss().cuda()
for epoch in range(opt.epochs):
    for iteration,item in enumerate(data_loader):
        if iteration >max_iter:
            max_iter = iteration
        target_pose_maps = item['target_pose_map'].cuda()
        target_pose = item['target_pose'].cuda()
        target_img = item['target_img'].cuda()
        target_parsing = item['target_parsing_20'].cuda()
        target_cloth_parsing = item['target_cloth_parsing'].cuda()
        source_cloth_parsing = item['source_cloth_parsing'].cuda()
        target_body = item['target_body'].cuda()
        target_face = item['target_face'].cuda()
        source_cloth_im = item['source_cloth_im'].cuda()
        mask = item['mask']
        input = torch.cat((target_parsing, target_pose_maps,target_body), dim=1)
        fake, flow_list = model(source_cloth_im, input,target_face,target_cloth_parsing , source_cloth_parsing)

        loss_L1 = criterion_L1(fake, target_img)
        loss_VGG = criterion_Vgg(fake, target_img)
        loss_style =criterion_Style(fake,target_img)
        lossG_all = loss_VGG + loss_L1 + 400*loss_style
        optimerG.zero_grad()
        lossG_all.backward()
        optimerG.step()

        board.add_scalars('loss_1',
                          { 'loss_style': loss_style, 'loss_l1': loss_L1, 'loss_Vgg': loss_VGG,
                            # 'loss_G':lossG,'loss_D':loss_D
                           }, global_step=max_iter * epoch + iteration)

    if (epoch + 1) % opt.save_count == 0:
        save_checkpoint('1', model, opt.model_dir, epoch)
        print('save')


