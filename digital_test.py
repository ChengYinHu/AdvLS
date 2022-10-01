import cv2
import random
# from resnet import classify
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.autograd import Variable
import torch
# import natsort
import os
from laser_simulation import classify, img_neon_effetc
# 图片霓虹等效果
#视频霓虹灯效果

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.resnet50(pretrained=True).eval().to(device)

if __name__ == "__main__":

    pic_path = 'green_lizard.jpg'

    for step in range(0, 500):
        print("step = ", step)

        img = cv2.imread(pic_path)
        height, width, n = img.shape

        img_neon_effetc(img, width, height, "1")

        label_adv = classify('result/result.jpg', net)

        img = plt.imread('result/result.jpg')
        # plt.imshow(img)
        # plt.show()

        print("label_adv = ", label_adv)

        if int(label_adv) != 46:

            print("Successfully attack")

            break

