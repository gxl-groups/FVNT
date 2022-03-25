import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import json
from PIL import Image
import pickle
import cv2 as cv
from utils import cv_utils
class DataSet(data.Dataset):
    def __init__(self,opt):
        super(DataSet, self).__init__()
        self.data_root = opt.data_root
        self.file_path = opt.file_path
        self.mode = opt.mode # train or test

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform_2 = transforms.Compose([transforms.ToTensor()
                            ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.transform_3 = transforms.Compose([transforms.Normalize((0.0011,-0.0106,0.5621),(0.0820,0.2219,1.0620))])
        self.point_num = opt.point_num
        self.height = 256
        self.width = 192
        self.radius = 4
        self.data_list = []
        self.parsing_list = [-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        # self.parsing_list = [0,1,2,3,5,6,7,8,9,-1,10,12,-1,13,14,15,16,17,18,19]
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
        for i in range(20):
            target_parsing_cloth_20[i] += (target_im_parsing == self.cloth_list[i]).float()
        target_cloth_parsing_1 = torch.zeros((1, self.height, self.width))
        for i in range(20):
            target_cloth_parsing_1 += (target_im_parsing == self.cloth_list[i]).float()
        target_cloth_im = target_img * target_cloth_parsing_1
        source_parsing_cloth_20 = torch.zeros((20, self.height, self.width))
        for i in range(20):
            if target_parsing_cloth_20[i].sum()>0:
                source_parsing_cloth_20[i] += source_cloth_mask.float()
        result = {
                'source_cloth_parsing':source_parsing_cloth_20,
                'target_cloth_parsing':target_parsing_cloth_20,
                'name':target,
                'source_cloth_im':source_cloth_img,
                'target_cloth_im':target_cloth_im,
            }

        return result

    def load_pickle_file(self,pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            f.close()
        return data

    def __len__(self):
        return len(self.data_list)