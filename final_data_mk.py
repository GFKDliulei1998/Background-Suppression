import os
import shutil
import scipy.io as io
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from common_li import processdat,read_npy
import imageio
import cv2

def dat_to_npy(filename,save_path):
    p=os.listdir(filename)
    os.chdir(filename)
    for i in p:
        img = np.fromfile(i,dtype='uint16',sep="")
        g=i[:-4]+'.npy'
        img = img.reshape([6144,6144])
        img = np.array(img).astype('float32')
        np.save(os.path.join(save_path,g),img)

# filename=r'C:\Users\zl\Desktop\all_data\DJ_final\labels_DJ\labels_01_dat'
# save_path=r'C:\Users\zl\Desktop\all_data\DJ_final\labels_DJ\labels_01_npy'
# dat_to_npy(filename,save_path)
def mat_to_npy(rootdir,savepath):
    f=os.listdir(rootdir)
    for i in f:
        matr=io.loadmat(os.path.join(rootdir,i))
        matr_data=matr['mn_img']
        g=i[:-4]+'.npy'
        save_dir=os.path.join(savepath,g)
        print(save_dir)
        np.save(save_dir,matr_data)
def npy_to_mat(rootdir,savepath):
    os.chdir(rootdir)
    f=os.listdir(rootdir)
    for i in f:
        if '.npy' in i:
            npy_data=np.load(os.path.join(rootdir,i),allow_pickle=True)
            g = i[:-4] + '.mat'
            save_dir = os.path.join(savepath, g)
            io.savemat(save_dir, {'data': npy_data})
            print(save_dir)
from PIL import Image

def tiff_to_jpg(imagesDirectory,distDirectory):
    for imageName in os.listdir(imagesDirectory):
        imagePath = os.path.join(imagesDirectory, imageName)
        image = Image.open(imagePath)  # 打开tiff图像
        print(imagePath)
        image.save(os.path.join(distDirectory, imageName[:-4] + '.jpg'),dpi=(300.0,300.0))  # 保存jpg图像

imagesDirectory = r"D:\dataset\02\02data_all_mat\loss-psnr\loss-psnr-tif"
distDirectory = r"D:\dataset\02\02data_all_mat\loss-psnr\loss-psnr-jpg"
tiff_to_jpg(imagesDirectory,distDirectory)
# rootdir=r'D:\dataset\mynet_v4\03_86_d_1'
# savepath=r'D:\dataset\02\sanwei_mat\03_86_d_1'
# npy_to_mat(rootdir,savepath)

def npy_to_png(rootdir,savepath):
    os.chdir(rootdir)
    f = os.listdir(rootdir)
    for i in f:
        npy_data = read_npy(os.path.join(rootdir, i))
        img=np.array(npy_data).astype(np.uint8)
        imageio.imsave(os.path.join(savepath,f'{i[:-4]}.png'),img)
        print(i)
rootdir=r'C:\Users\adminn\Desktop\report\实验\codeimage\Infrared-Small-Target-Detection-based-on-PSTNN-master\images'
savepath=r'C:\Users\adminn\Desktop\report\实验\codeimage\Infrared-Small-Target-Detection-based-on-PSTNN-master\images_test'
# npy_to_png(rootdir,savepath)

def read_IPD(file_IPD):
    '''
    1、读取IPD文件，并把每张图片的目标信息保存到对应的txt文件中
    2、把txt文件整合到一个文件夹下
    3、把bmp文件保存为mat，然后python打开，并保存相应的npy（images与标签一一对应)
    4、把dat文件保存为npy文件
    5、将目标周围分割为512*512
    '''
    file_names=os.listdir(file_IPD)

    for file in file_names:

        D=os.path.join(file_IPD,file)

        d1=file[:-4]+'.txt'
        D2=os.path.join(file_IPD,d1)

        with open(D2,'x') as f1:
            with open(D,'r') as file_ima:

                all_lines_list=file_ima.readlines()
                length=len(all_lines_list)

                '''列号  |  行号 | 像素数  | 长度 |  宽度   | 灰度和  |  背景平均灰度  |  背景标准差##'''

                for gk in all_lines_list:
                    if 'SATEOBJ_NUM' in gk:
                        n_tgtNum = int(gk[12:])
                        for i in range(length-n_tgtNum,length):
                            f1.write(all_lines_list[i])
                        break

                            #tgt_all[d1][n]=[float(all_lines_list[i][0:8]),float(all_lines_list[i][9:17]),float(all_lines_list[i][23:27]),float(all_lines_list[i][28:32]),float(all_lines_list[i][18:22]),float(all_lines_list[i][33:43]),float(all_lines_list[i][44:53]),float(all_lines_list[i][54:63])]

    return None
'''--------------------------------------------------------------'''
# file_ipd=r'C:\Users\zl\Desktop\all_data\DJ_final\images_ipd\L1'
# a=read_IPD(file_ipd)
# print(a)
def gh(rootdir,newpath):
    os.chdir(rootdir)
    f=os.listdir(rootdir)
    for i in f:
        if '.txt' in i:
            shutil.copy(i,os.path.join(newpath,i))
    return None
# newpath=r'C:\Users\zl\Desktop\all_data\DJ_final\images_ipd\L1_txt'
# rootdir=r'C:\Users\zl\Desktop\all_data\DJ_final\images_ipd\L1'
# gh(rootdir,newpath)
'''-----------------------------------------------------------'''
def mat_to_npy(rootdir,savepath):
    f=os.listdir(rootdir)
    for i in f:
        matr=io.loadmat(os.path.join(rootdir,i))
        matr_data=matr['mn_img']
        g=i[:-4]+'.npy'
        save_dir=os.path.join(savepath,g)
        print(save_dir)
        np.save(save_dir,matr_data)
# rootdir=r'C:\Users\zl\Desktop\all_data\DJ_final\images_DJ\images_01_mat'
# savepath=r'C:\Users\zl\Desktop\all_data\DJ_final\images_DJ\images_01_npy'
# mat_to_npy(rootdir,savepath)


def rancrop_txt(txt_file,txt_crop_file):
    '''
    先读取txt位置，长宽
    '''
    p=os.listdir(txt_file)
    os.chdir(txt_file)
    for file in p:
        re=open(file,'r',encoding='UTF-8')
        n_re=open(os.path.join(txt_crop_file,file),'a',encoding='UTF-8')
        maseg=re.readlines()
        for s in maseg:
            s.rstrip()
            tgt=s.split(' ')
            w=math.floor(float(tgt[0]))
            h=math.floor(float(tgt[1]))
            leth=int(tgt[3])
            wide=int(tgt[4])
            bg_pix_ave=float(tgt[6])   #平均背景灰度
            bg_pix_std=float(tgt[7])   #背景标准差
            #######
            if w<20:
                continue
            if 20<=w<=511:
                x_min=random.randint(0, w-10)
            if 511<w<=6143-512:
                x_min=random.randint(w-480, w-10)
            if w>6143-512 and w<=6123:
                x_min=random.randint(w-500, 5630)
            if w>6123:
                continue
            if h<511:
                y_min=random.randint(0, h-10)
            if 511<h<6143-512:
                y_min=random.randint(h-500, h-10)
            if h>6143-512 and h<=6123:
                y_min=random.randint(h-500, 5630)
            if h>6123:
                continue
            #######
            w_new=w-x_min           #切割后的坐标
            h_new=h-y_min

            n_re.write(str(x_min))
            n_re.write(' ')
            n_re.write(str(y_min))
            n_re.write(' ')

            n_re.write(str(w_new))
            n_re.write(' ')
            n_re.write(str(h_new))
            n_re.write(' ')

            n_re.write(str(w))
            n_re.write(' ')
            n_re.write(str(h))
            n_re.write(' ')

            n_re.write(tgt[3])
            n_re.write(' ')
            n_re.write(tgt[4])
            n_re.write(' ')
            n_re.write(tgt[6])
            n_re.write(' ')
            n_re.write(tgt[7])
# txt_file=r'C:\Users\zl\Desktop\all_data\DJ_final\images_ipd\L1_txt'
# txt_crop_file=r'C:\Users\zl\Desktop\all_data\DJ_final\images_ipd\L1_crop_txt'
# rancrop_txt(txt_file,txt_crop_file)
def hw_to_wh(img):
	return np.transpose(img, axes=[1, 0])

def final_read_npy(filename):
    img = np.load(filename)
    img = np.array(img)
    kernel= np.ones((3, 3),np.float32)

    # img2=np.load(filename.replace('labels','images'))
    fig,ax=plt.subplots(1)
    ax.imshow(processdat(img),'gray')       #第二个是x坐标，第一个是y坐标
    rect = patches.Rectangle((222, 199), 18, 19, linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    plt.show()

def read_npy_1(filename):
	img=np.load(filename)
	# img=np.array(img).astype('float32')
	# img=img
	return img


def product_npy(txt_file,npy_file,label_file):
    '''
    1 先读取txt文件信息，获取x_min,y_min(w,h)坐标
    -------
    2 用对应的坐标切割npy文件，得到新的npy文件并存储
    '''
    save_images_path=r'C:\Users\zl\Desktop\all_data\DJ_final\images_DJ\images_01_512_npy'
    save_labels_path=r'C:\Users\zl\Desktop\all_data\DJ_final\labels_DJ\labels_01_512_npy'
    p=os.listdir(txt_file)
    os.chdir(txt_file)
    for i in p:
        npy_file_name=os.path.join(npy_file,i.replace('.txt','.npy'))
        label_file_name=os.path.join(label_file,i.replace('.txt','.npy'))
        origi_images_npy=read_npy_1(npy_file_name)
        origi_labels_npy=hw_to_wh(read_npy_1(label_file_name))

        re=open(i,'r',encoding='UTF-8')
        all_mes=re.readlines()
        po=0
        for g in all_mes:
            g.rstrip()
            g_r=g.split(' ')
            aix_x=int(g_r[0])
            aix_y=int(g_r[1])
            file_image_crop=origi_images_npy[aix_y:aix_y+512,aix_x:aix_x+512]
            file_label_crop=origi_labels_npy[aix_y:aix_y+512,aix_x:aix_x+512]
            np.save(os.path.join(save_images_path,i.replace('.txt',f'_{po}.npy')),file_image_crop)
            np.save(os.path.join(save_labels_path,i.replace('.txt',f'_{po}.npy')),file_label_crop)
            po+=1
txt_file=r'C:\Users\zl\Desktop\all_data\DJ_final\images_ipd\L1_crop_txt'
npy_file=r'C:\Users\zl\Desktop\all_data\DJ_final\images_DJ\images_01_npy'
label_file=r'C:\Users\zl\Desktop\all_data\DJ_final\labels_DJ\labels_01_npy'
# product_npy(txt_file,npy_file,label_file)

def fj(root,newp):
    os.chdir(root)
    a=os.listdir(root)
    for i in a:
        f=open(i,'r')
        b=f.readlines()
        po=0
        for g in b:
            g=g.strip()
            gg=os.path.join(newp,i.replace('.txt',f'_{po}.txt'))
            tf=open(gg,'w')
            tf.write(g)
            po+=1
root=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\DJ\L1_crop_txt'
newp=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\DJ\L1_crop_all_txt'

'''
1.label要中值滤波
2.要把验证搞出来，选择最好的一组
   --测试集
   --验证集
   --评价标准--验证集的平均snr
        --先读取坐标，大小等文件信息
        --计算
'''
