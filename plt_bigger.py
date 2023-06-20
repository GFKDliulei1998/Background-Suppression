import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from common_li import normalize,show_dat
import os
input_file='./show_resulte/image/'
take_name='11.npy'
filename=input_file+take_name
(name,ext)=os.path.splitext(take_name)
output_filename='./show_resulte_png/image/'+name+'.png'

#图像的信息
# SNR='8.24'
# up=86
# down=0
#星点的范围,h对应y,w对应x
# tx0=10
# tx1=100
# ty0=410
# ty1=500
tx0=293
tx1=316
ty0=467
ty1=484
#绘主图
img=np.load(filename)
#img2=np.load('EVAL_INPUT/foruse/04-013/04-013FG.npy')
#img=img-img2
h=np.shape(img)[0]
w=np.shape(img)[1]

img=normalize(img,h,w)
#img=show_dat(img,h,w,'EVAL_INPUT/foruse/04-013/04-013back.png',1,0,dpi=300)

img_ins=img[ty0:ty1,tx0:tx1]
h1=np.shape(img_ins)[0]
w1=np.shape(img_ins)[1]
fig,ax=plt.subplots(1,1)#用来控制子图的个数
ax.axis('off')
ax.imshow(np.reshape(img,[h,w]),cmap='gray',vmin=0,vmax=255)
#嵌入局部放大图的坐标系!!一般是20%，loc一般是upper right
axins=inset_axes(ax,width='40%',height='40%',loc='center left',bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes)
#,bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes
#axins=ax.inset_axes((0.4,0.1,0.4,0.3))
axins.axis('off')
axins.imshow(np.reshape(img_ins,[h1,w1]),cmap='gray',vmin=0,vmax=255)

#画方框
sx=[tx0,tx1,tx1,tx0,tx0]
sy=[ty0,ty0,ty1,ty1,ty0]
x1=-1
y1=-1
ax.plot(sx,sy,'red')
sx=[x1,w1,w1,x1,x1]
sy=[y1,y1,h1,h1,y1]
axins.plot(sx,sy,'red')

#画两条线
xy=(tx0,ty0)#在大图中的坐标
#xy2=(tx0,ty1)
xy2=(0,w1)#在小图中的坐标
con=ConnectionPatch(xyA=xy2,xyB=xy,coordsA='data',coordsB='data',axesA=axins,axesB=ax,color='r')
axins.add_artist(con)
xy=(tx1,ty0)
xy2=(h1,w1)
#xy2=(-1,w1)
con=ConnectionPatch(xyA=xy2,xyB=xy,coordsA='data',coordsB='data',axesA=axins,axesB=ax,color='r')
axins.add_artist(con)
#写SNR
#axins.text(-1,h1+5,'SNR='+SNR,color='r')
plt.show()
fig.savefig(output_filename,dpi=300,bbox_inches='tight',transparent=True)
plt.close()
