import cv2
import sys

if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv2.imread('C:/Users/zzz/Desktop/conf/images/sharp_0.tif')
    # 需要放大的部分
    part = img[222:252, 150:190]
    # 双线性插值法
    mask = cv2.resize(part, (80, 60), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    if img is None is None:
        print('Failed to read picture')
        sys.exit()

    # 放大后局部图的位置img[210:410,670:870]
    img[150:210, 310:390] = mask

    # 画框并连线
    cv2.rectangle(img, (150, 222), (190, 252), (0, 255, 0), 1)
    cv2.rectangle(img, (310, 150), (390, 210), (0, 255, 0), 1)
    img = cv2.line(img, (190, 222), (310, 150), (0, 255, 0))
    img = cv2.line(img, (190, 252), (310, 210), (0, 255, 0))
    # 展示结果
    cv2.imwrite('C:/Users/zzz/Desktop/conf/images2/sharp_amp.tif', img, [cv2.IMWRITE_TIFF_XDPI,300, cv2.IMWRITE_TIFF_YDPI,300])
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
