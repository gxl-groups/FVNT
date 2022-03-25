import argparse
from mine.dataset1_viton_fixhfs import DataSet
import torch
import os
from mine.network_stage_1_mine_final_viton import define_G
from torch.utils.tensorboard import SummaryWriter
from mine.loss import GANLoss
from mine.visualization import board_add_images
from mine.network_stage_1_mine_final_viton import define_D
def get_opt():
    parser =argparse.ArgumentParser()
    parser.add_argument("--mode",default='train')
    parser.add_argument("--point_num",default=18)
    parser.add_argument("--data_root",default='./dataset/viton_resize')
    parser.add_argument("--file_path", default='train_pairs.txt')
    parser.add_argument("--batch_size",default=18)
    parser.add_argument('--epochs',default=100)
    parser.add_argument('--ndf',default=64)
    parser.add_argument('--n_layers',default=2)
    parser.add_argument('--num_D',default=3)
    parser.add_argument('--getIntermFeat',default=False)
    parser.add_argument('--lr',default=0.0001)
    parser.add_argument('--tensorboard_dir',default='./tensorboard/stage1')
    parser.add_argument('--model_dir',default='./model/stage3_model')
    parser.add_argument('--display_count',default=50)
    parser.add_argument('--save_count',default=1)
    parser.add_argument('--tex_size', type=int, default=3, help='input tex size')
    parser.add_argument('--image_size', type=int, default=256, help='input image size')

    opt = parser.parse_args()
    return opt

def save_checkpoint(mode,model_g, save_path,epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path,mode+'_model_'+str(epoch))
    torch.save({'G':model_g.state_dict()}, save_file)

def cross_entropy(fake_out,real_parsing):
    fake_out = fake_out + 0.00001
    log_output = torch.log(fake_out)
    loss = -1* torch.sum((real_parsing*log_output))
    loss = loss/(256*256)
    return loss

opt = get_opt()
dataSet = DataSet(opt)
data_loader = torch.utils.data.DataLoader(dataSet, batch_size=opt.batch_size, pin_memory=True,shuffle=True)
max_iter = 0

if not os.path.exists(os.path.dirname(opt.tensorboard_dir)):
    os.makedirs(os.path.dirname(opt.tensorboard_dir))
board = SummaryWriter(opt.tensorboard_dir)

model = define_G(42, 20, 64, 'unet_128', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02).cuda()
discriminator_stage1_1 = define_D(62,64,3,norm='batch', use_sigmoid=False,num_D=3).cuda()
optimerG = torch.optim.Adam(params=model.parameters(),lr=opt.lr,betas=(0.5,0.999))
optimerD = torch.optim.Adam(params=discriminator_stage1_1.parameters(),lr=opt.lr,betas=(0.5,0.999))
criteriion_G = GANLoss()
for epoch in range(opt.epochs):
    for iteration,item in enumerate(data_loader):
        if iteration >max_iter:
            max_iter = iteration
        target_pose_maps = item['target_pose_map'].cuda()
        target_pose = item['target_pose'].cuda()
        target_parsing_20 = item['target_parsing_20'].cuda()
        hair_face_shoes = item['hair_face_shoes'].cuda()
        source_cloth_img = item['source_cloth_img'].cuda()
        source_cloth_parsing = item['source_cloth_parsing'].cuda()

        input = torch.cat((source_cloth_img,source_cloth_parsing,hair_face_shoes,target_pose_maps), dim=1)
        fake = model(input)

        input_fake = torch.cat((input, fake), dim=1)
        input_real = torch.cat((input, target_parsing_20), dim=1)
        pre_fake = discriminator_stage1_1(input_fake.detach())
        pre_real = discriminator_stage1_1(input_real)
        loss_real = criteriion_G(pre_real, True)
        loss_fake = criteriion_G(pre_fake, False)
        loss_D = (loss_real + loss_fake) * 0.5
        optimerD.zero_grad()
        loss_D.backward()
        optimerD.step()

        pre_fake2 = discriminator_stage1_1(input_fake)
        lossG = criteriion_G(pre_fake2, True)
        loss_corss = cross_entropy(fake,target_parsing_20)
        lossG_all = loss_corss + 0.2*lossG
        optimerG.zero_grad()
        lossG_all.backward()
        optimerG.step()

        board.add_scalars('loss_1',
                          { 'loss_corss': loss_corss,'loss_G':lossG,'loss_D':loss_D
                           }, global_step=max_iter * epoch + iteration)
        if (iteration + 1) % 20 == 0:
            visuals = [[source_cloth_parsing,hair_face_shoes,target_pose],
                    [fake,target_parsing_20]]
            board_add_images(board, 'combine', visuals, max_iter * epoch + iteration)

    if (epoch + 1) % opt.save_count == 0:
        save_checkpoint('1', model, opt.model_dir, epoch)
        model.eval()
        # test_model(model, '/public/home/guxl/wt/result/stage_1/mine_viton_unet_facehairshoes2/epoch{}'.format(epoch))
        model.train()
        torch.cuda.empty_cache()
        print('save')


