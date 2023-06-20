# import cv2 as cv
# import sys
# from common_li import read_npy
# if __name__ == '__main__':
#     # 读取图像并判断是否读取成功
#     # img = read_npy(r'D:\dataset\mynet_v4\show_resulte\image\11.npy')
#     img = cv.imread(r'D:\dataset\mynet_v4\imgs\CBDNet_v13.png')
#     # 需要放大的部分
#     part = img[300:400, 250:350]
#     # 双线性插值法
#     mask = cv.resize(part, (300, 300), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
#     if img is None is None:
#         print('Failed to read picture')
#         sys.exit()
#
#     # 放大后局部图的位置img[210:410,670:870]
#     img[110:410, 570:870] = mask
#
#     # 画框并连线
#     cv.rectangle(img, (250, 300), (350, 400), (0, 255, 0), 1)
#     cv.rectangle(img, (570, 110), (870, 410), (0, 255, 0), 1)
#     img = cv.line(img, (350, 300), (570, 110), (0, 255, 0))
#     img = cv.line(img, (350, 400), (570, 410), (0, 255, 0))
#     # 展示结果
#     cv.imshow('img', img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from common_li import normalize,show_dat,processdat
import os
import imageio
input_file='./02-000092_0/'
save_file='./02-000092_0_show/'
# take_name='v6_3-000092_0.npy'
# take_name='out_li-000092_0.npy'
# take_name='md-f-000092_0.npy'
# take_name='tophat_000092_0.npy'
# take_name='s_ex000092_0.npy'
# take_name='bm3d_000092_0.npy'
take_name='dncnn-000092_0.npy'
filename=input_file+take_name
(name,ext)=os.path.splitext(take_name)
output_filename=save_file+name+'.jpg'

#图像的信息
# SNR='8.24'
# up=86
# down=0
#星点的范围,h对应y,w对应x
# tx0=293
# tx1=316
# ty0=467
# ty1=484

tx0=175
tx1=205
ty0=15
ty1=45
#---------
# t2x0=338
# t2x1=368
# t2y0=210
# t2y1=240
#----------
#绘主图
img=np.load(filename)
img2=np.load(r'D:\dataset\mynet_v4\02_000090_0\v6_3_000090_0.npy')
print(img2.max())
print(img.max())
# img=imageio.imread(filename)

h=np.shape(img)[0]
w=np.shape(img)[1]
# img=show_dat(img,h,w,output_filename,1,0,dpi=800)

img_ins=img[ty0:ty1,tx0:tx1]
h1=np.shape(img_ins)[0]
w1=np.shape(img_ins)[1]

fig,ax=plt.subplots(1,1)#用来控制子图的个数
ax.axis('off')
# ax.margins(0,0)

ax.imshow(np.reshape(processdat(img,3,3),[h,w]),cmap='gray')
#嵌入局部放大图的坐标系!!一般是20%，loc一般是upper right
#/////////////////
axins=inset_axes(ax,width='35%',height='35%',loc='upper right',bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes)
axins.axis('off')
#axins.imshow(np.reshape(processdat(img_ins,3.5,3.5),[h1,w1]),cmap='gray')
axins.imshow(np.reshape(processdat(img_ins,6,3),[h1,w1]),cmap='gray')

#########
# img_ins_2=img[t2y0:t2y1,t2x0:t2x1]
# h2=np.shape(img_ins_2)[0]
# w2=np.shape(img_ins_2)[1]
# axins_2=inset_axes(ax,width='35%',height='35%',loc='upper left',bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes)
# axins_2.axis('off')
# axins_2.imshow(np.reshape(img_ins_2,[h2,w2]),cmap='gray',vmin=0,vmax=255)
#########



#画方框
###/////////////////
sx=[tx0,tx1,tx1,tx0,tx0]
sy=[ty0,ty0,ty1,ty1,ty0]
x1=-1
y1=-1
ax.plot(sx,sy,'red',linewidth=2)
sx=[x1,w1,w1,x1,x1]
sy=[y1,y1,h1,h1,y1]
axins.plot(sx,sy,'red',linewidth=2)
###//////////////////////////
###
#画第二个方框
######################
# sx=[t2x0,t2x1,t2x1,t2x0,t2x0]
# sy=[t2y0,t2y0,t2y1,t2y1,t2y0]
# x2=-1
# y2=-1
# ax.plot(sx,sy,'red',linewidth=2)
# s2x=[x2,w2,w2,x2,x2]
# s2y=[y2,y2,h2,h2,y2]
# axins_2.plot(s2x,s2y,'red',linewidth=2)
######################
#画两条线
# xy=(tx0,ty0)#在大图中的坐标
# #xy2=(tx0,ty1)
# xy2=(-1,h1)#在小图中的坐标
#
# con=ConnectionPatch(xyA=xy2,xyB=xy,coordsA='data',coordsB='data',axesA=axins,axesB=ax,color='r',linewidth=2)
# axins.add_artist(con)
#
# xy=(tx1,ty0)
# xy2=(w1,h1)
# #xy2=(-1,w1)
#
# con=ConnectionPatch(xyA=xy2,xyB=xy,coordsA='data',coordsB='data',axesA=axins,axesB=ax,color='r',linewidth=2)
# axins.add_artist(con)

#################################
#画另外两条线
# xy=(t2x0,t2y0)#在大图中的坐标
# #xy2=(tx0,ty1)
# xy2=(-1,h2)#在小图中的坐标
#
# con=ConnectionPatch(xyA=xy2,xyB=xy,coordsA='data',coordsB='data',axesA=axins_2,axesB=ax,color='r',linewidth=0.5)
# axins_2.add_artist(con)
#
# xy=(t2x1,t2y0)
# xy2=(w2,h2)
# #xy2=(-1,w1)
#
# con=ConnectionPatch(xyA=xy2,xyB=xy,coordsA='data',coordsB='data',axesA=axins_2,axesB=ax,color='r',linewidth=0.5)
# axins_2.add_artist(con)


plt.show()

fig.savefig(output_filename,dpi=600,bbox_inches='tight',pad_inches = -0.1,transparent=True)
plt.close()
