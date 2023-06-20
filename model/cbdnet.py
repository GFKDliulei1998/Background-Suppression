
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import *
from common_li import read_npy
'''初级模块'''
class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

'''目前理解:
        逆卷积与填充，上采样到与编码器部分一样的'''

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x

#####################与1*1conv##########################
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

'''舍弃'''
###########################
# class FCN(nn.Module):
#     def __init__(self):
#         super(FCN, self).__init__()
#         self.fcn = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.fcn(x)

'''###############新增结构##################################### '''
class pool(nn.Module):
    #########面向输入图片的池化，应用与f(input)*x  self.channels 为x的通道数 b,c,h,w--->b,
    def __init__(self,channels):
        super(pool, self).__init__()
        self.channels = channels
        self.l=nn.MaxPool2d(self.channels//64)
        self.f=nn.AvgPool2d(self.channels//64)
        self.conv2d=nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid(),
        )

    def forward(self,input):
        re=torch.cat([self.l(input),self.f(input)],1)       #2,1,256,256-->2,2,32,32
        re=self.conv2d(re)                                  #2,2,32,32 ---->2,516,32,32
        return re


class CA(nn.Module):
    def __init__(self,channels):   #channels:以顶层x的C为基准  1，64，128，256，512// 0,1,2,3,4//256 ,128, 64, 32
        super(CA, self).__init__()
        self.channels=channels
        self.bottleneck_channels=channels//4
        self.MAX_pool=nn.AdaptiveMaxPool2d(1)
        self.Avg_pool=nn.AdaptiveAvgPool2d(1)


##########f(x)   *y       x=顶  y=低  b,c,h,w  -->b,c/2,h,w
        self.p=nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels//2, momentum=0.9),
            nn.Sigmoid()
        )

###########f(input)   *x
        self.p2=pool(self.channels)        #2,512,32,32-->mul(512,32,32)
        self.Up = nn.UpsamplingBilinear2d(scale_factor=2)   #512 64 64
        self.p2_1=nn.Conv2d(in_channels=self.channels, out_channels=self.channels//2, kernel_size=1,stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        # 256 64 64  加一个relu

##########up(x)
        self.p3=nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) ,        #2,516,64,64
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels//2, kernel_size=1, stride=1, padding=0),
 #           nn.BatchNorm2d(self.channels//2, momentum=0.9),
 #           nn.Sigmoid()
            nn.ReLU()
        )
        '''
        以底层4为例
        '''
        self.fina_conv= nn.Conv2d(in_channels=self.channels//2, out_channels=self.channels//2, kernel_size=1, stride=1, padding=0)

    def forward(self,x,y,input):
        #f(x)*y  顶调制高
        pool_1=self.MAX_pool(x)
        pool_2=self.Avg_pool(x)
        pool=pool_1+pool_2                      #2,512,1,1
        rs_pool=self.p(pool)                    #(channels//2,1,1)  256,1,1
        rs_1=torch.mul(rs_pool,y)               #(2,256,1,1)*(2,256,64,64)  通道注意力 2,256,64,64

        #f(input)*x   输入调制顶
        rs_2=torch.mul(self.p2(input),x)     #将输入先从1*256*256--》2，2，32，32---》2，512，32,32
        rs_2 = self.relu(self.p2_1(self.Up(rs_2)))             #2,512,32,32->2 512 64 64-> 2 256 64 64
        #顶层上采样

        rs_3=self.p3(x)      #2,256,64,64

        rs_final=self.fina_conv(rs_1+rs_2+rs_3)
        return rs_final


'''#########################################################################################################'''
##########################################
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(1, 64),
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
        )

        self.down3 = nn.AvgPool2d(2)
        self.conv3 = nn.Sequential(
            single_conv(256, 512),
            single_conv(512, 512),
            single_conv(512, 512),
            single_conv(512, 512),
            single_conv(512, 512),
        )

        self.ca1=CA(512)              #256 64 64
        self.conv4 = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        # self.up1 = up(256)
        # self.conv3 = nn.Sequential(
        #     single_conv(128, 128),
        #     single_conv(128, 128),
        #     single_conv(128, 128)
        # )

        self.ca2=CA(256)             #128 128 128
        self.conv5 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),

        )

        self.ca3 = CA(128)           #64 256 256
        self.conv6 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64),

        )

        # self.up2 = up(128)
        # self.conv4 = nn.Sequential(
        #     single_conv(64, 64),
        #     single_conv(64, 64)
        # )

        self.outc = outconv(64, 1)

        for m in self.modules():

            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')



    def forward(self, x):
        inx = self.inc(x)                   #0

        down1 = self.down1(inx)             #1
        conv1 = self.conv1(down1)           #   128 128 128

        down2 = self.down2(conv1)           #2  256 64 64
        conv2 = self.conv2(down2)

        down3=self.down3(conv2)             #3 512 32 32
        conv3=self.conv3(down3)

 #       up1 = self.up1(conv2, conv1)
        ca1 = self.ca1(conv3,conv2,x)          #decode 1      x,y,input
        conv4 = self.conv4(ca1)                 #256 64 64

 #       up2 = self.up2(conv3, inx)
        ca2 = self.ca2(conv4,conv1,x)
        conv5 = self.conv5(ca2)                 #128 128 128

        ca3 = self.ca3(conv5,inx,x)
        conv6 = self.conv6(ca3)                 #64, 256 256

        out = self.outc(conv6)                  #1 256 256
        return out

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.unet=UNet()
    def forward(self,x):
        out=self.unet(x)
        return out

'''----------------------------------------------------'''

class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
#    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
    def forward(self, out_image, gt_image):
#        l1_loss=F.l1_loss(out_image,gt_image)
#       l2_loss = F.mse_loss(out_image, gt_image)

        loss=F.smooth_l1_loss(out_image,gt_image)
        return loss



class my_fixed_loss(nn.Module):
    def __init__(self,c,z):
        super(my_fixed_loss, self).__init__()
        self.c=c
        self.z=z

    def forward(self,out_image ,labe_image):
        # print(out_image.dtype)
        loss_ssim=ssim(out_image,labe_image)
        res_1=1-loss_ssim

        q = (labe_image - labe_image.min()) / (labe_image.max() - labe_image.min())

        q_res = 1+((1-q).pow(self.c))*self.z

        res_mul=torch.mul(q_res,abs(out_image-labe_image))/(512*512)
        res_2=res_mul.sum()/2

        res= 0.625*res_2+res_1*0.375
        return res


class my_smooth_l1loss(nn.Module):
    def __init__(self,c,z):
        super(my_smooth_l1loss, self).__init__()
        self.c=c
        self.z=z
    def forward(self,out_image,label_image):
        loss_ssim = ssim(out_image, label_image)
        res_1 = 1 - loss_ssim

        q = (label_image - label_image.min()) / (label_image.max() - label_image.min())
        q_res = 1 + ((1 - q).pow(self.c)) * self.z
        t=abs(out_image-label_image)
        t_res=torch.where(t<1,0.5*t**2,t-0.5)
        res_mul=torch.mul(q_res,t_res)
        res_mul=res_mul.sum()/(4*128*128)

        return res_mul

# if __name__=='__main__':
#     out_image=read_npy(r'C:\Users\zl\Desktop\model_v2\mynet_v3\result\out_1.npy')
#     labe_image=read_npy(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\real_train\labels_512\000005_1.npy')
#     out_image=torch.from_numpy(out_image).unsqueeze(0).cuda()
#     labe_image=torch.from_numpy(labe_image).unsqueeze(0).cuda()
#
#     loss_ssim = ssim(out_image, labe_image)
#     print(loss_ssim)
