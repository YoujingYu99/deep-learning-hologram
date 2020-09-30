
from PIL import Image
import os
import numpy as np

#https://cocodataset.org/#download

path = "C:/Users/Youjing Yu/PycharmProjects/deep-learning-cgh/coco/val2017/1/original/"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        im = Image.open(item)
        f, e = os.path.splitext(item)
        imResize = im.resize((256,256), Image.ANTIALIAS)
        imgray = imResize.convert('L')
        imgray.save(f + ' resized.jpg')



resize()
