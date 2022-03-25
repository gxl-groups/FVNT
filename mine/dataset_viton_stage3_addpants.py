import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import ImageDraw
import torch.nn.functional as F
import json
from PIL import Image
import pickle
import numpy as np
import os
class DataSet(data.Dataset):
    def __init__(self,opt):
        super(DataSet, self).__init__()
        self.data_root = opt.data_root
        self.file_path = opt.file_path
        self.mode = opt.mode # train or test

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform_2 = transforms.Compose([transforms.ToTensor()
                            ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.point_num = opt.point_num
        self.height = 256
        self.width = 192
        self.radius = 4
        self.data_list = []
        self.parsing_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        self.body_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 15, 16, 17, -1, -1]
        self.face_list = [-1, -1, 2, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1]
        self.shoes_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 19]
        self.cloth_list = [-1,-1,-1,-1,-1,5,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        # get data_list
        with open(os.path.join(self.data_root,self.file_path),'r') as f:
            for line in f.readlines():
                line = line.strip()
                self.data_list.append(line)
            f.close()
        print(len(self.data_list))

    def __getitem__(self, item):
        item_list = self.data_list[item]
        target,source = item_list.split()
        target_img = Image.open(os.path.join(self.data_root,self.mode,'image',target))
        target_img = self.transform_2(target_img)
        source_cloth_img = Image.open(os.path.join(self.data_root,self.mode,'cloth', source))
        source_cloth_img = self.transform_2(source_cloth_img)
        source_cloth_mask = Image.open(os.path.join(self.data_root,self.mode,'cloth-mask',source))
        source_cloth_mask = np.array(source_cloth_mask)
        source_cloth_mask = (source_cloth_mask >= 128).astype(np.float32)
        source_cloth_mask = torch.from_numpy(source_cloth_mask)
        source_cloth_img = source_cloth_img*source_cloth_mask.float()
        target_parsing = Image.open(os.path.join(self.data_root,self.mode,'image-parse',target.replace('.jpg','.png')))
        target_parsing = np.array(target_parsing).astype(np.long)
        target_im_parsing = torch.from_numpy(target_parsing)
        target_parsing_cloth_20 = torch.zeros((20, self.height, self.width))
        target_im_parsing_20 = torch.zeros((20, self.height, self.width))
        target_cloth_parsing_1 = torch.zeros((1, self.height, self.width))
        target_parsing_body_1 = torch.zeros((1, self.height, self.width))
        target_parsing_face_1 = torch.zeros((1, self.height, self.width))
        target_parsing_shoes_1 = torch.zeros((1, self.height, self.width))

        for i in range(20):
            target_parsing_cloth_20[i] += (target_im_parsing == self.cloth_list[i]).float()
            target_parsing_body_1 += (target_im_parsing == self.body_list[i]).float()
            target_parsing_face_1 += (target_im_parsing == self.face_list[i]).float()
            target_parsing_shoes_1 += (target_im_parsing == self.shoes_list[i]).float()
            target_cloth_parsing_1 += (target_im_parsing == self.cloth_list[i]).float()
            target_im_parsing_20[i] += (target_im_parsing == self.parsing_list[i]).float()

        target_face_im = target_img * target_parsing_face_1
        target_body_im = target_img * target_parsing_body_1
        target_shoes_im = target_img * target_parsing_shoes_1
        target_cloth_im = target_img * target_cloth_parsing_1

        mask_id = self.get_maskid()
        mask = Image.open(os.path.join(self.data_root,self.mode,'mask', mask_id + ".png"))
        mask = self.transform(mask)
        mask = mask[0].reshape(1, 1, 512, 512)
        mask = F.interpolate(mask, size=[self.height, self.width], mode='nearest').reshape(1, self.height, self.width)
        target_body_im_hole = target_body_im * (1 - mask) + target_shoes_im

        source_parsing_cloth_20 = torch.zeros((20, self.height, self.width))
        for i in range(20):
            if target_parsing_cloth_20[i].sum()>0:
                source_parsing_cloth_20[i] += source_cloth_mask.float()

        target_pose_name = target.replace('/', '_').replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.data_root, self.mode, 'pose', target_pose_name)) as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        target_pose_maps = torch.zeros((self.point_num, self.height, self.width))
        target_im_pose = Image.new('RGB', (self.width, self.height))
        target_pose_draw = ImageDraw.Draw(target_im_pose)
        for i in range(self.point_num):
            one_map = Image.new('RGB', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            pointX = pose_data[i][0]
            pointY = pose_data[i][1]
            if pointX > 1 or pointY > 1:
                draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                              pointY + self.radius), 'white', 'white')
                target_pose_draw.ellipse((pointX - self.radius, pointY - self.radius, pointX + self.radius,
                                          pointY + self.radius), 'white', 'white')
            one_map = self.transform(one_map)[0]
            target_pose_maps[i] = one_map
        target_im_pose_array = self.transform_2(target_im_pose)

        result = {
                'source_cloth_parsing':source_parsing_cloth_20,
                'target_cloth_parsing':target_parsing_cloth_20,
                'target':target,
                'source':source,
                'source_cloth_im':source_cloth_img,
                'target_cloth_im':target_cloth_im,
                'target_body': target_body_im_hole,
                'target_face': target_face_im,
                'mask':mask,
                'target_img': target_img,
                'target_pose_map': target_pose_maps,
                'target_pose': target_im_pose_array,
                'target_parsing_20': target_im_parsing_20,
            }
        return result

    def get_maskid(self):
        id = np.random.randint(12000)
        id = str(id)
        l = len(id)
        for i in range(l, 5):
            id = '0' + id
        return id

    def load_pickle_file(self,pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            f.close()
        return data

    def __len__(self):
        return len(self.data_list)