import os
import argparse

import torch
from dataset.New_dataset import *
from utils import AverageMeter
from dataset.New_dataset import my_Dataset
from model.my_net_ca_v2 import myNetwork_ca_v2
from model.li_Unet2 import UNet2,fixed_loss3,UNet3
from tqdm import tqdm
import torch.utils.data as Data
from SpareDeris import calculate_psnr
from tensorboardX import SummaryWriter
from model.fina_net import my_final
from model.fina_net_v1 import my_final_v1
from model.final_net_v3 import my_final_v3
from model.final_v2 import my_final_v2
from model.fina_net_v4 import my_final_v4
from model.final_net_v5 import my_final_v5
from model.final_net_v6 import my_final_v6
from model.final_net_v6_1 import my_final_v6_1
from model.final_net_6_2 import my_final_v6_2
from model.final_net_v7 import my_final_v7
from model.final_net_v8 import my_final_v8
from predict import pre
from SpareDeris import ssim,show_test
writer = SummaryWriter('runs/')
'''----------------参数------------'''
parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--bs', default=4, type=int, help='batch size')
parser.add_argument('--ps', default=16, type=int, help='patch size')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=2000, type=int, help='sum of epochs')
args = parser.parse_args()

'''------------------预处理设置-----------------'''
learning_rate = 3e-4
def train(train_loader ,model ,criterion ,optimizer):
    losses = AverageMeter()
    model.train()
    tbar =tqdm(train_loader)
    #	for (noise_img,clean_img) in train_loader:
    for (noise_img, clean_img) in tbar:
        input_var = noise_img.cuda()
        target_var = clean_img.cuda()
        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg

if __name__ == '__main__':
    save_dir = './save_model/'
    val_save_dir='./save_model/val/'
    loss_file = r'C:\Users\zl\Desktop\model_v2\mynet_v4\save_model\loss.txt'
    model = my_final_v8()
    model.cuda()
    #确定好验证集的数据
    min_val=0
    root_dir=r'C:\Users\zl\Desktop\model_v2\mynet_v4\data\02_sample\02_test\images_02'
    inp_image = os.listdir(root_dir)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(model_info['state_dict'])
        cur_epoch = model_info['epoch']
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler.load_state_dict(model_info['scheduler'])

    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # weight_decay=0.01
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs ,eta_min=1e-6)
        cur_epoch = 0

    criterion=fixed_loss3()
    criterion.cuda()

    train_dataset = my_Dataset_npy(root_dir='./data/02_sample/images_02')
    #train_dataset = my_Dataset_npy(root_dir='./data/DJ/train/images_train')
    train_loader = Data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    for epoch in range(cur_epoch, args.epochs + 1):
        loss = train(train_loader, model, criterion, optimizer)
        # writer.add_scalar('loss' ,loss ,global_step=epoch)
        scheduler.step()
        sum_res = 0.0
        sum_psnr=0.0
        all_sum=0.0

        model.eval()
        with torch.no_grad():
            for i in range(len(inp_image)):
                input_image = read_npy_1(os.path.join(root_dir, inp_image[i]))   #images  labels
                s_lab = read_npy_1(os.path.join(r'C:\Users\zl\Desktop\model_v2\mynet_v4\data\02_sample\02_test\labels_02', inp_image[i]))
                img_lab = torch.from_numpy((s_lab.reshape(1, 1, 512, 512).astype(np.float32)))

                input_image = np.expand_dims(input_image, axis=0)
                input_var = torch.from_numpy(input_image).unsqueeze(0).cuda()
                output=model(input_var)
                output_image = output[0, 0, ...].cpu().detach().numpy()
                psnr_res = calculate_psnr(output_image, s_lab)
                img_my = torch.from_numpy((output_image.reshape(1, 1, 512, 512).astype(np.float32)))
                my_res=ssim(img_my,img_lab).numpy()

                sum_psnr+=psnr_res
                sum_res+=my_res
                all_sum=sum_psnr+sum_res

            if all_sum>min_val:
                max_val=sum_res
                max_psnr=sum_psnr
                min_val=all_sum

                print('save model')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),},
                    os.path.join(val_save_dir, 'checkpoint.pth.tar'))

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join(save_dir, 'checkpoint.pth.tar'))

        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}\t'
            .format(
            epoch,
            lr=optimizer.param_groups[-1]['lr'],
            loss=loss
            ))
        print('sumres:',sum_res)
        print('psnr:',sum_psnr)
        loss = round(loss, 5)  # 保留5位小数
        with open(loss_file, 'a') as f:
            f.write(str(loss))
            f.write(' ' * 4)
            f.write(str(sum_res))
            f.write(' ' * 4)
            f.write(str(sum_psnr))
            f.write("\n")
    print('max_val:', max_val)
    print('max_psnr:',max_psnr)