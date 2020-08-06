# *- coding: utf-8 -*
from PIL import Image
import os
import os.path
import numpy as np
import cv2

# 指明被遍历的文件夹
rootdir = './camvid352'
for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        _, postfix = filename.split('.')
        if postfix != 'png':
            continue
        # print('parent is :' + parent)
        # print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        # print('the fulll name of the file is :' + currentPath)

        img = Image.open(currentPath)
        # print(img.format, img.size, img.mode)
        # img.show()
        box1 = (0, 0, 480, 352)  # 设置左、上、右、下的像素
        image1 = img.crop(box1)  # 图像裁剪
        # print(image1.format, image1.size, image1.mode)
        print(currentPath)
        image1.save(currentPath)  # 存储裁剪得到的图像