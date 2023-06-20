
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import cv2 as cv
import glob
from common_li import *
import shutil
import PIL.Image as Image
import torch
import torch.nn.functional as F
from math import exp
import numpy
def slecdata(rootpath,newpath):
    file=os.listdir(rootpath)
    os.chdir(rootpath)
    for i in range(0,900,3):
        shutil.copy(file[i],os.path.join(newpath,file[i]))
rootpath=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\DJ\L1_crop_all_txt'
newpath=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\DJ\L1_sample_crop'
# slecdata(rootpath,newpath)
# print(len(os.listdir(newpath)))



 ########################函数##############################
'''-------------------------------------------------------'''

'''转换图片格式（png,jpg)'''
def imgTypeCge(path,begtyp,tartyp):
    file_list=os.listdir(path)
    for i in file_list:
        if begtyp in i:
            os.rename(path+'/'+i,path+'/'+i[:-4]+tartyp)

'''---------------------------------------------------'''

'''阈值分割，将分给后的图片转移到指定文件夹下'''
def thres_img(file_path,tgt_path):
    path_list=glob.glob(file_path+'/*')
    print(path_list)
    file_names=os.listdir(file_path)
    k=0
    for i in path_list:
        f=cv.imread(i,cv.IMREAD_GRAYSCALE)
        _,rst=cv.threshold(f,65,255,cv.THRESH_BINARY)
        img=Image.fromarray(rst,mode='L')
        tgt_path_label=os.path.join(tgt_path,file_names[k])
        img.save(tgt_path_label)
        k=k+1
    return True


def get_file(old_path,root_path=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\real_train\labels_1'):
    '''复制文件到指定文件夹下，并重新命名'''

    i=1
    file_names=os.listdir(old_path)
    for file in file_names:
        print(file)
        if '.raw' in file:
            #shutil.copy(old_path+'/'+file,root_path+'/'+'0'*(6-len(str(i)))+'%d'%(i)+'.raw')
            shutil.copy(old_path + '/' + file, root_path + '/' + file[-10:-4] + '.raw')
            #i+=1
# old=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\real_train\labels'
# get_file(old)


'''-------------------------------------------------------'''
def read_IPD(file_IPD):
    '''读取ipd文件信息，返回列表以及恒星、碎片个数'''
    star_all={}
    tgt_all={}

    file_names=os.listdir(file_IPD)
    for file_IPD in file_names:
        D='D:/pythonProject/exper-1/example-im/ipd_all/'+file_IPD
        blcyle=True
        i = 0
        n = 0
        with open(D,'r') as file_ima:
            while blcyle:
                a=file_ima.readline().rstrip()
                if 'STAROBJ_NUM' not in a:
                    continue
                else:
                    n_starNum=int(a[12:])
                    a=file_ima.readline().rstrip()
                    n_tgtNum=int(a[12:])
                break
            '''列号  |  行号  | 长度 |  宽度  |   像素数    | 灰度和  |  背景平均灰度  |  背景标准差##'''
            while blcyle:
                a=file_ima.readline().rstrip()#继续往下毒，前n_starNum行是恒星，后n_tgtNum是目标
                if not a:
                    break
                if 'SATEOBJ'not in a:
                    print(a)
                    if i<n_starNum:
                        star_all[i]=[float(a[0:8]),float(a[9:17]),float(a[23:27]),float(a[28:32]),float(a[18:22]),float(a[33:43]),float(a[44:53]),float(a[54:63])]
                        i+=1
                    else:
                        tgt_all[n]=[float(a[0:8]),float(a[9:17]),float(a[23:27]),float(a[28:32]),float(a[18:22]),float(a[33:43]),float(a[44:53]),float(a[54:63])]
                        n+=1
        return star_all,tgt_all,n_starNum,n_tgtNum

'''--------------------------------------------------------------'''



def tre_image(img):
    image = processdat(img, 4096, 4096)
    #f=(image-image.min())/(image.max()-image.min())*255
    #da_trans=transforms.CenterCrop(512)
    #f=Image.fromarray(image)
    #f=da_trans(f)
    return image


'''---------------------------------------------------------------'''


def randCrop(file_name_path, label_name_path , h, w, save_path):
    '''步骤
        1.定位到指定的文件夹，搜索到ori,sup的dat文件名
        2.每个dat文件随机裁剪4次，以及中心裁剪1次，共5张
        3。将原始图与对应的标签分别放置到文件夹下，以9：1的比例随机抽取训练集以及验证集
    '''
    image_name_list=os.listdir(file_name_path)
    label_name_list=os.listdir(label_name_path)

    for i in range(len(image_name_list)):
        filename=os.path.join(file_name_path,image_name_list[i])

        labelname=os.path.join(label_name_path,label_name_list[i])
        file_image, h_f, w_f = read_dat2(filename)
        file_label, h_l, w_l = read_dat2(labelname)

        for y in range(5):
            if y!=4:

                aix_x=random.randint(0,4096-w)
                aix_y=random.randint(0,4096-h)
                print(aix_x)

                file_image_crop=file_image[aix_x:aix_x+w-1,aix_y:aix_y+h-1]
                file_label_crop=file_label[aix_x:aix_x+w-1,aix_y:aix_y+h-1]
            else:
                file_image_crop = file_image[1920:2176, 1920:2176]
                file_label_crop = file_label[1920:2176, 1920:2176]


                                                #000002_1.csv
            filename_1=image_name_list[i][:-4]+f'_{y}'+'.npy'
            print('filename_1:',filename_1)
            save_path_1=os.path.join(save_path,filename_1)
            print('save_path:',save_path_1)
            save_labe_path=save_path_1.replace('image','label')

            np.save(save_path_1,file_image_crop)
            np.save(save_labe_path,file_label_crop)



            # cv.imwrite(f'{save_path}',file_image_crop)
            # cv.imwrite(f'{save_labe_path}',file_label_crop)

# filename='04_ori_000001.dat'
# labelname='04_sup_000001.dat'
# h=128
# w=512
# save_path=r'D:\pythonProject\exper-1\really_dataset\image'
# randCrop(filename,labelname,h,w,save_path)

'''------------------------------------------------------'''

# file_name_path =r'C:\Users\zl\Desktop\model_v2\CBDNet-v2\data\real_train\ori'
# label_name_path=r'C:\Users\zl\Desktop\model_v2\CBDNet-v2\data\real_train\sur'
# h, w, save_path=257,257,r'C:\Users\zl\Desktop\model_v2\CBDNet-v2\data\real_train\image'
# randCrop(file_name_path, label_name_path , h, w, save_path)

'''--------------------------------------------------------'''

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    #print(gauss)
    return gauss / gauss.sum()




# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
   # print(_1D_window)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    ff=np.array((_2D_window)).shape
   # print(ff)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    return window



# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

# img_1=read_img('rs.png')
# #img_2=cv.imread(r'D:\pythonProject\exper-1\example-im\label_all\000000.png',cv.IMREAD_GRAYSCALE)
# img_2=read_img(r'D:\pythonProject\CBDNet-pytorch-master\data\train\labels\000000.png')
#
# plt.subplot(1,2,1)
# plt.imshow(img_1)
# plt.subplot(1,2,2)
# plt.imshow(img_2)
# plt.show()
# img_tor_1=torch.from_numpy((img_1.reshape(1,1,256,256).astype(np.float32)))
# img_tor_2=torch.from_numpy((img_2.reshape(1,1,256,256).astype(np.float32)))
#
#
# res=ssim(img_tor_1,img_tor_2)
# print(res)
def show_test(my_predic, li_predic, save_psnr=None, save_snr=None, save_ssim=None, save_targe=None, one=1):
    my = os.listdir(my_predic)
    li = os.listdir(li_predic)
    sum_my=0.0
    sum_li=0.0
    for i in range(len(my)):
        o_my = read_npy(os.path.join(my_predic, my[i]))  # 训练时分别把两个网络的输出放在两个文件夹下
        o_li = read_npy(os.path.join(li_predic, li[i]))
        # s_ori = read_npy(os.path.join(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\test\images', my[i]))
        # s_lab = read_npy(os.path.join(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\test\labels', my[i]))
        s_ori = read_npy(os.path.join(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\02_test\images_02', my[i]))
        s_lab = read_npy(os.path.join(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\02_test\labels_02', my[i]))

        img_my = torch.from_numpy((o_my.reshape(1, 1, 512, 512).astype(np.float32)))
        img_li = torch.from_numpy((o_li.reshape(1, 1, 512, 512).astype(np.float32)))
        img_lab = torch.from_numpy((s_lab.reshape(1, 1, 512, 512).astype(np.float32)))
        my_res = ssim(img_my, img_lab)
        li_res = ssim(img_li, img_lab)
        sum_my+=my_res.numpy()
        sum_li+=li_res.numpy()
            # my_res = calculate_psnr(o_my, s_lab)
            # li_res = calculate_psnr(o_li, s_lab)
        print(my_res, ' ', li_res)
        # if one == 1:
        #     plt.subplot(1, 4, 1)
        #     plt.imshow(s_ori, 'gray')
        #     plt.subplot(1, 4, 2)
        #     plt.imshow(s_lab, 'gray')
        #     plt.subplot(1, 4, 3)
        #     plt.imshow(o_my, 'gray')
        #     plt.subplot(1, 4, 4)
        #     plt.imshow(o_li, 'gray')
        #     plt.show()
        # else:
        #     s_ori = processdat(s_ori)
        #     s_lab = processdat(s_lab)
        #     o_my = processdat(o_my)
        #     o_li = processdat(o_li)
        #     plt.subplot(1, 4, 1)
        #     plt.imshow(s_ori, 'gray')
        #     plt.title(f'{my[i]}')
        #     plt.subplot(1, 4, 2)
        #     plt.imshow(s_lab, 'gray')
        #     plt.subplot(1, 4, 3)
        #     plt.imshow(o_my, 'gray')
        #     plt.subplot(1, 4, 4)
        #     # plt.imshow(cv2.medianBlur(o_my,3), 'gray')
        #     plt.imshow(o_li, 'gray')
        #     plt.show()
    # print(sum_my)
    # print(sum_li)
    return sum_my

#my_predic = r'C:\Users\zl\Desktop\model_v2\mynet_v3\result\n_bse'
my_predic = r'C:\Users\zl\Desktop\model_v2\mynet_v3\result\1\out_my_final_v1'
#li_predic = r'C:\Users\zl\Desktop\model_v2\mynet_v3\result\n_'
li_predic = r'C:\Users\zl\Desktop\model_v2\mynet_v3\result\1\out_li'
# show_test(my_predic, li_predic, one=0)

# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def get_motion_dsf(image_size, motion_angle, motion_dis):
    PSF = np.zeros(image_size)  # 点扩散函数
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    # 将对应角度上motion_dis个点置成1
    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return PSF / PSF.sum()  # 归一化

