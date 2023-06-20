import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import *
from thop import profile

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()  # 第一句话这里都是调用父类的构造函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 如果是序列网络结构可以直接用sequential来构造
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)  # 按照这个卷积方法,会变成原来的两倍,通道变成原来1/2

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 先对x1做一次反卷积,做完反卷积后,x1的size是1,128,64,64
        #   print(x1.size())
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))  # 这里是查看两边的尺寸差,如果有差值,就会对x1四周做一次填充,这样就可以很好地相加了
        x = x2 + x1  # 并不是维度的拓展,而是数值的相加
        #   print(x.size())
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 尺寸不变#输入尺度改为1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),  # 输出尺度改为1
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)  # 直接调用序列


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(2, 64),  # 输入变成2
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 1)  # 输出变成1

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

        self.inc = nn.Sequential(
            single_conv(1, 64),  # 输入变成2
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 1)  # 输出变成1

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out

#############

class UNet3(nn.Module):
    def __init__(self):
        super(UNet3, self).__init__()

        self.inc = nn.Sequential(
            single_conv(1, 64),  # 输入变成2
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),

        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),

        )
        self.down3=nn.AvgPool2d(2)
        self.conv3=nn.Sequential(
            single_conv(256,512),
            single_conv(512,512),
            single_conv(512, 512),
        )

        self.up1 = up(512)
        self.conv4 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
        )

        self.up2 = up(256)
        self.conv5 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128)
        )
        self.up3 = up(128)
        self.conv6 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 1)  # 输出变成1

    def forward(self, x):
        inx = self.inc(x)           #64 512 512

        down1 = self.down1(inx)         #128 256 256
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)       #256 128 128

        down3=self.down3(conv2)
        conv3=self.conv3(down3)     #512 64 64

        up1 = self.up1(conv3, conv2)   #256 128 128
        conv4 = self.conv4(up1)


        up2 = self.up2(conv4, conv1)
        conv5 = self.conv5(up2)

        up3=self.up3(conv5,inx)
        conv6=self.conv6(up3)

        out = self.outc(conv6)
        return out

# a=torch.rand(1,1,512,512)
# model=UNet3()
# flops,params=profile(model,(a,))
# print(model(a).size())
# print('flops:',flops,'params:',params)
##
class Network_li(nn.Module):
    def __init__(self):
        super(Network_li, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()

    def forward(self, x):
        noise_level = self.fcn(x)
        # 在这里可以通过一个参数a对平滑程度进行调整
        # 例如:noise_level=a*noise_level
        concat_img = torch.cat([x, noise_level], dim=1)  # 相加之后变成2通道
        out = self.unet(concat_img) + x
        return noise_level, out


class fixed_loss_li(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):  # 分别是输出图像,真实图像,估计噪声,真实噪声,flag
        A = 0.35
        # l2_loss = F.mse_loss(out_image, gt_image)#计算输出图像和真实图像的差距
        #   L1_loss=F.l1_loss(out_image,gt_image)
        smooth_l1 = F.smooth_l1_loss(out_image, gt_image)
        #  print('aaa',  out_image.shape())
        asym_loss = torch.mean(
            if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise,
                                                                                         2))  # 在真实图像中这个是0,可以不考虑
        # tvloss
        h_x = est_noise.size()[2]  # 由3通道变成1通道,所以从2->1,从3->2
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w
        # ssim_loss
        ssim_loss = ssim(out_image, gt_image)
        # 总loss

        #   loss =(1-A)*l2_loss +  0.01 * asym_loss + 0.05 * tvloss+A*(-ssim_loss)#在这一步加上SSIM
        loss = (1 - A) * smooth_l1 + A * (1 - ssim_loss) + 0.05 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class fixed_loss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, if_asym):  # 分别是输出图像,真实图像,估计噪声,真实噪声,flag
        l2_loss = F.mse_loss(out_image, gt_image)
        # asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss + 0.05 * tvloss
        # loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class fixed_loss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image):  # 分别是输出图像,真实图像,估计噪声,真实噪声,flag
        A = 0.35
        # l2_loss = F.mse_loss(out_image, gt_image)#计算输出图像和真实图像的差距
        #   L1_loss=F.l1_loss(out_image,gt_image)
        smooth_l1 = F.smooth_l1_loss(out_image, gt_image)
        # ssim_loss
        ssim_loss = ssim(out_image, gt_image)
        # 总loss
        #   loss =(1-A)*l2_loss +  0.01 * asym_loss + 0.05 * tvloss+A*(-ssim_loss)#在这一步加上SSIM
        # loss = (1 - A) * smooth_l1 + A * (1 - ssim_loss)
        loss = 0.65*smooth_l1 + 0.35*(1 - ssim_loss)
        return loss

