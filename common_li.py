import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import PIL.Image as Image
from torchvision import transforms
import bm3d
'''--------------------新增函数----------------------'''


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

#读取dat文件
# (img_type分为已知大小和未知大小,如果图像大小放在文件的前两个字节,直接读即可)
# (如果是已知大小且没有放在前两个字节,在img_type处输入图像大小,如1024)

def my_read_dat1(filename):
	size = os.path.getsize(filename)

	if size == 2204032:  # 01星
		mn_image = np.fromfile(filename, dtype='>u2', offset=2144)
		rows = 1072
		cols = 1027
		mn_image = mn_image[0:1100944]
		mn_image1 = mn_image.reshape(cols, rows)
		mn_image1 = np.transpose(mn_image1)
		mn_image2 = mn_image1[24:1048, 1:1025]
		img = np.array(mn_image2).astype('float32')

	elif size == 8819688:  # 02、03星
		mn_image = np.fromfile(filename, dtype='>u2', offset=4296)
		rows = 2148
		cols = 2052
		mn_image = mn_image[0:4407696]
		mn_image1 = mn_image.reshape(cols, rows)
		mn_image1 = np.transpose(mn_image1)

		mn_image2 = mn_image1[50:2098, 3:2051]
		img = np.array(mn_image2).astype('float32')


	else:
		print('read_dat1 is error!')

	return img

def read_dat1(filename,img_type='Unkown'):
	img = np.fromfile(filename,dtype='uint16',sep="")
	print(img.shape)
	if img_type=='Unkown':
		w=1024
		h=1024
		img=img[1072:-3]
	else:
		h=w=int(img_type)
		img=img
	img=img.reshape([h,w])
	img=np.array(img).astype('float32')
	return img,h,w



def read_dat2(filename):
	img=np.fromfile(filename,dtype='uint16',sep="")
	print(np.size(img))
	h=w=0
	if np.size(img)==1048576:
		h=w=1024
	elif np.size(img)==4194304:
		h=w=2048
	else:
		h=w=4096
	#img=img.reshape([h,w,1])
	img=img.reshape([h,w])
	img=np.array(img).astype('float32')
	return img


###############################################

#读取npy文件
def read_npy(filename):
	img=np.load(filename)
	img=np.array(img).astype(np.float32)
	return img



def read_npy_1(filename):
	img=np.load(filename)
	img=np.array(img).astype('float32')
	return img


#处理dat文件,up-value和down-value的取值将影响到图像的显示效果
def processdat(img,up_value=3,down_value=3):
	mean=np.mean(img)
	std=np.std(img)
	down=mean-down_value*std
	top=mean+up_value*std
	down=down if down>0 else 0
	img[img>top]=top
	img[img<down]=down
	return img
img,h,w=read_dat1(r'D:\dataset\mynet_v4\data\GEOGC_000001.raw')
print(img)
plt.imshow(processdat(img),'gray')
plt.show()

##############################################
#save_flag=1或0,表示存储或不存储图像为png格式,show_flag=1或0,表示显示或不显示图像.
def show_dat(img,h,w,output_filename,save_flag=1,show_flag=1,dpi=800):
	plt.ion()
	plt.imshow(np.reshape(img,[h,w]),cmap='gray',vmin=0,vmax=255)
	if save_flag==1:
		plt.savefig(output_filename,dpi=dpi,bbox_inches='tight')
	if show_flag==1:
		plt.show(block=False)
		plt.pause(5)#显示图像的时间间隔,通过这个数字调整
		plt.close()

def read_data_dat(filename,labelsname):
	f=os.listdir(filename)
	l=os.listdir(labelsname)
	for i in range(len(f)):
		image=read_dat2(os.path.join(filename,f[i]))
		label=read_dat2(os.path.join(labelsname,l[i]))
		plt.subplot(1,2,1)
		plt.imshow(processdat(image),'gray')
		plt.subplot(1,2,2)
		plt.imshow(processdat(label),'gray')
		plt.show()
# read_data_dat(r'D:\dataset\TianJ\images',r'D:\dataset\TianJ\labels')

def normalize(img,h,w):
	ymax=255
	ymin=0
	img=np.where(img>0,img,0)
	xmax=np.max(img)
	xmin=np.min(img)
	img=(ymax-ymin)*(img-np.min(img))/(np.max(img)-np.min(img))
	return img


def hw_to_wh(img):
	return np.transpose(img, axes=[1, 0])


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).astype('float32')

#分割图片,最终输出的形式为[图片数目,H,W,C]
def get_patch(img,patch_size):
	H=img.shape[0]
	W=img.shape[1]
	IMG=[]
	patch_H=math.ceil(H/patch_size)
	patch_W=math.ceil(W/patch_size)
	print('patch_H&patch_W:',patch_H,patch_W)
	for i in range(patch_H):
		H_up=(i+1)*patch_size if H>(i+1)*patch_size else H
		for j in range(patch_W):
			W_up=(j+1)*patch_size if W>(j+1)*patch_size else W
			img_seg=img[i*patch_size:H_up,j*patch_size:W_up,:]
			IMG.append(img_seg)
	return IMG,patch_H,patch_W

#拼接图像
def concate_patch(IMG,patch_H,patch_W):
	img_num=len(IMG)
	A=[]
	AW=[]
	if img_num==1:
		A=IMG[0]
	else:
		for k in range(img_num):
			if k%patch_W==0:
				if k==patch_W:
					A=AW
					AW=[]
				else:
					A=np.concatenate((A,AW),axis=0)
					AW=[]
				AW=IMG[k]
			else:
				AW=np.concatenate((AW,IMG[k]),axis=1)
		A=np.concatenate((A,AW),axis=0)
	return A

def mesh_3d(img,h,w):
	x=np.arange(0,w,1)
	y=np.arange(0,h,1)
	X,Y=np.meshgrid(x,y)
	Z=np.reshape(img,[h,w])
	return X,Y,Z
	
def read_img_1(filename):
	img = Image.open(filename)
	pre_transform=transforms.CenterCrop(256)
	img=pre_transform(img)
	img = np.array(img).astype('float32')
	#img = np.expand_dims(img,axis=0)

	return img


'''-----------------------------------'''
import random
'''
制作预测集
1.查找到对应的.dat文件，转化为.npy,读取npy文件为数组,并将npy文件保存到对应的文件夹中
2.输入到predict
'''
def save_product_test_i(file_name_path, label_name_path ,save_path,h=512,w=512):
	image_name_list = os.listdir(file_name_path)
	label_name_list = os.listdir(label_name_path)

	for i in range(len(image_name_list)):
		while i%10 ==0:
			print('i:',i)
			filename = os.path.join(file_name_path, image_name_list[i])
			labelname = os.path.join(label_name_path, label_name_list[i])

			file_image = my_read_dat1(filename)
			file_label = my_read_dat1(labelname)

			for y in range(2):
				if y == 0:

					aix_x = random.randint(0, 1024 - w)
					aix_y = random.randint(0, 1024 - h)
					print(aix_x)

					file_image_crop = file_image[aix_x:aix_x + w , aix_y:aix_y + h ]
					file_label_crop = file_label[aix_x:aix_x + w , aix_y:aix_y + h ]

					plt.subplot(1,2,1)
					plt.imshow(file_image,'gray')
					plt.subplot(1,2,2)
					plt.imshow(file_image_crop,'gray')
					plt.show()
				else:
					file_image_crop = file_image[512:1024, 512:1024]
					file_label_crop = file_label[512:1024, 512:1024]

				filename_1 = image_name_list[i][:-4] + f'_{y}' + '.npy'

				save_path_1 = os.path.join(save_path, filename_1)
				print(save_path_1)
				save_labe_path = save_path_1.replace('images', 'labels')
				print(save_labe_path)
				np.save(save_path_1, file_image_crop)
				np.save(save_labe_path, file_label_crop)
			break
# file_name_path=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\real_train\images_1'
# label_name_path=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\real_train\labels_1'
# save_path=r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\real_test\images_512'
# save_product_test_i(file_name_path, label_name_path ,save_path)

'''
读取输入与输出的npy文件，并展示对比

'''
def calculate_psnr(img1,img2,border=0):
	if not img1.shape == img2.shape:
		raise ValueError('Input imagesmust have the same dimensions')
	mse=np.mean((img1-img2)**2)
	return 20*math.log10(65535/math.sqrt(mse))

def calculate_snr(img1,img2):
	noise_signal=img1-img2
	clean_signal=img2
	noise_signal_2=noise_signal**2
	clean_signal_2=clean_signal**2
	sum1=np.sum(clean_signal_2)
	sum2=np.sum(noise_signal_2)
	snrr=10*math.log10(math.sqrt(sum1)/math.sqrt(sum2))
	return snrr
# a=read_npy(r'D:\dataset\mynet_v4\testfile\BM3D-Denoise-master\1_1.npy')
# b=read_npy_1(r'D:\dataset\mynet_v4\data\02_sample\02_test\images_02\000084_0.npy')
# print(calculate_psnr(a,b))


##挑数据集 512
def read_show(ori_path,down_value=0,up_value=0,process=None):
	p=os.listdir(ori_path)
	for i in range(len(p)):
		pa=os.path.join(ori_path,p[i])
		ori_image=read_npy_1(pa)
		label_image=read_npy_1(pa.replace('images','labels'))
		# label_image=hw_to_wh(label_image)
		##对比拉伸
		if process is not None:
			mean_ori = np.mean(ori_image)
			mean_label=np.mean(label_image)
			std_label= np.std(label_image)
			std_ori = np.std(ori_image)

			down_ori = mean_ori - down_value * std_ori
			top_ori = mean_ori + up_value * std_ori

			down_label=mean_label-down_value*std_label
			top_label=mean_label+up_value*std_label

			down_ori = down_ori if down_ori > 0 else 0
			down_out = down_out if down_out > 0 else 0
			down_label=down_label if down_label>0 else 0

			ori_image[ori_image > top_ori] = top_ori
			ori_image[ori_image < down_ori] = down_ori

			label_image[label_image > top_label] = top_label
			label_image[label_image < down_label] = down_label

		plt.subplot(1,2,1)
		plt.imshow(processdat(ori_image),'gray')
		#plt.imshow(ori_image,'gray')
		plt.title(f'original_image_{pa[-10:-4]}')
		plt.subplot(1,2,2)
		plt.imshow(processdat(label_image),'gray')
		# plt.imshow(label_image, 'gray')
		plt.title(f'suppress_label_{pa[-10:-4]}')

		plt.show()

#read_show(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\test\images')

# '''--------------------------------------'''
def show_test_cp(my_predic,li_predic,save_psnr=None,save_snr=None,save_ssim=None,save_targe=None,one=1):

	my=os.listdir(my_predic)
	li=os.listdir(li_predic)
	sum_res=0
	for i in range(len(my)):
		o_my=read_npy_1(os.path.join(my_predic,my[i]))					#训练时分别把两个网络的输出放在两个文件夹下
		o_li=read_npy_1(os.path.join(li_predic,li[i]))
		# s_ori=read_npy(os.path.join(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\test\images',my[i]))
		# s_lab=read_npy(os.path.join(r'C:\Users\zl\Desktop\model_v2\mynet_v3\data\02_sample\test\labels',my[i]))
		s_ori = read_npy_1(os.path.join(r'D:\dataset\mynet_v4\data\02_sample\02_test\images_02', my[i]))
		s_lab = read_npy_1(os.path.join(r'D:\dataset\mynet_v4\data\02_sample\02_test\images_02', my[i]))
		# print(o_li)

		# my_res=ssim(o_my,s_lab)
		# li_res=ssim(o_li,s_lab)
		my_res = calculate_psnr(o_my, s_lab)
		li_res = calculate_psnr(o_li, s_lab)
		sum_res += li_res
		print(my_res,' ',li_res)
		if one==1:
			plt.subplot(1,4,1)
			plt.imshow(s_ori,'gray')
			plt.subplot(1,4,2)
			plt.imshow(s_lab,'gray')
			plt.subplot(1,4,3)
			plt.imshow(o_my,'gray')
			plt.subplot(1,4,4)
			plt.imshow(o_li,'gray')
			plt.show()
		else:
			s_ori = processdat(s_ori)
			s_lab = processdat(s_lab)
			o_my  = processdat(o_my)
			o_li  = processdat(o_li)

			plt.subplot(1, 4, 1)
			plt.imshow(s_ori,'gray')
			plt.title(f'{my[i]}')
			plt.subplot(1, 4, 2)
			plt.imshow(s_lab, 'gray')
			plt.subplot(1, 4, 3)
			plt.imshow(o_my,'gray')
			plt.subplot(1, 4, 4)
			# plt.imshow(cv2.medianBlur(o_my,3), 'gray')
			plt.imshow(o_li,'gray')
			plt.show()
		# with open(save_path,'a') as f:
		# 	f.write(str(my_res))
		# 	f.write(' '*5)
		# 	f.write(str(li_res))
		# 	f.write('\n')
	print(sum_res/16)


# #
my_predic=r'D:\dataset\mynet_v4\result\1\v6-1-2000-3x3'
li_predic=r'D:\dataset\mynet_v4\result\source_ex\02'
#show_test_cp(my_predic, li_predic, one=0)
