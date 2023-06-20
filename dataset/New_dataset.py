import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.utils.data as Data
from common_li import read_npy,read_npy_1

class my_Dataset(Dataset):
    def __init__(self,root_dir,transform=None,sample_num=240):

        self.root_dir=root_dir
        self.sample_num=sample_num
        self.transform=transform
        self.img_path_list=glob.glob(self.root_dir+'/*')     #['path1','path2',...]
        self.label_pat_list=glob.glob(self.root_dir.replace('images','labels')+'/*')   #[labe_path1,...]

    def __getitem__(self, index):
        item=self.img_path_list[index]
        img=Image.open(item)

        item_label=self.label_pat_list[index]
        label=Image.open(item_label)

        if self.transform is not None:
            img=self.transform(img)
            label=self.transform(label)
        img=np.array(img).astype('float32')
        label=np.array(label).astype('float32')
        img=np.expand_dims(img,axis=0)
        label=np.expand_dims(label,axis=0)
        return img,label

    def __len__(self):
        l=len(self.img_path_list)
        return  l

class my_Dataset_npy(Dataset):
    def __init__(self,root_dir,transform=None):

        self.root_dir=root_dir
        self.transform=transform
        self.img_path_list=glob.glob(self.root_dir+'/*')     #['path1','path2',...]
        self.label_pat_list=glob.glob(self.root_dir.replace('images','labels')+'/*')   #[labe_path1,...]

    def __getitem__(self, index):
        item=self.img_path_list[index]
        item_label = self.label_pat_list[index]

        img_1=read_npy_1(item)
        label_1=read_npy_1(item_label)

        if self.transform is not None:
            img_1=self.transform(img_1)
            label_1=self.transform(label_1)

        img=np.expand_dims(img_1,axis=0)
        label=np.expand_dims(label_1,axis=0)
        return img,label

    def __len__(self):
        l=len(self.img_path_list)
        return  l



