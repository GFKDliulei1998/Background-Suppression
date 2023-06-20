import os
import torch
import argparse
import cv2
import PIL.Image as Image
from utils.common import read_img
from common_li import read_npy_1
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from model.li_Unet2 import UNet2,fixed_loss3,UNet3
from model.fina_net import my_final
from model.fina_net_v1 import my_final_v1
from model.final_net_v3 import my_final_v3
from model.final_v2 import my_final_v2
from model.fina_net_v4 import my_final_v4
from model.final_net_v5 import my_final_v5
from model.final_net_v6_1 import my_final_v6_1
from model.final_net_6_2 import my_final_v6_2
from model.final_net_v6 import my_final_v6
from model.final_net_v7 import my_final_v7
from model.final_net_v8 import my_final_v8
def pre(root_dir,out_save_file):
    inp_image=os.listdir(root_dir)
    model =my_final_v1()
    save_dir = './save_model/'
    model.cuda()
    model.eval()
    b='checkpoint.pth.tar'
    if os.path.exists(os.path.join(save_dir, b)):
        # load existing model
        model_info = torch.load(os.path.join(save_dir, b))
        model.load_state_dict(model_info['state_dict'])
        print('1')
    else:
        print('Error: no trained model detected!')
        exit(1)

    for i in range(len(inp_image)):
        input_image=read_npy_1(os.path.join(root_dir,inp_image[i]))
        input_image = np.expand_dims(input_image,axis=0)
    #input_var =  torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
        input_var =  torch.from_numpy(input_image).unsqueeze(0).cuda()
        with torch.no_grad():
            q = model(input_var)
            output=q
    #output_image = chw_to_hwc(output[0,...].cpu().numpy())
        output_image = output[0,0,...].cpu().numpy()
        out_save_files=os.path.join(out_save_file,inp_image[i])
        np.save(out_save_files,output_image)
#root_dir=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\test\images'
root_dir=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\test\images'
out_save_file=r'C:\Users\zl\Desktop\model_v2\mynet_v3\result\new'
# pre(root_dir,out_save_file)