# coding=utf-8
import numpy as np
import scipy.ndimage as ndi
from skimage import measure, color
import matplotlib.pyplot as plt

a=np.random.random((3,3))
b=np.array(a>0.5)
print(b.astype('float'))
# 编写一个函数来生成原始二值图像
# def microstructure(l=256):
#     n = 5
#     x, y = np.ogrid[0:l, 0:l]  # 生成网络
#     mask = np.zeros((l, l))
#     generator = np.random.RandomState(1)  # 随机数种子
#     points = l * generator.rand(2, n ** 2)
#     mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
#     mask = ndi.gaussian_filter(mask, sigma=l / (4. * n))  # 高斯滤波
#     return mask > mask.mean()
#
#
# data = microstructure(l=128) * 1  # 生成测试图片
#
# labels = measure.label(data, connectivity=2)  # 8连通区域标记
# dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
# print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# ax1.imshow(data, plt.cm.gray, interpolation='nearest')
# ax1.axis('off')
# ax2.imshow(labels, plt.cm.gray,interpolation='nearest')
# ax2.axis('off')
#
# fig.tight_layout()
# plt.show()