import sys
import math
import cv2
import numpy as np
import random
from PIL import  Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torchvision.models as models
import matplotlib.pylab as pyl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.resnet50(pretrained=True).eval().to(device)

def classify(dir, net):
    img = Image.open(dir)
    img = img.convert("RGB")

    mean = [0.485, 0.456, 0.406]
    std = [0.29, 0.224, 0.225]

    img = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,std = std)
    ])(img).to(device)

    f_image = net.forward(Variable(img[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]  # 从小到大排序, 再从后往前复制一遍，So相当于从大到小排序
    I = I[0:10]  # 挑最大的num_classes个(从0开始，到num_classes结束)
    # print(I)
    label = I[0]  # 最大的判断的分类

    return label

def img_neon_effetc(img, width, height, filenameSize):

    r, g, b, = 0, 0, 0

    for i in range(0, 28):

        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        cv2.ellipse(img, (x, y), (1, 1), 0, 0, 360, (b, 255, r), 3, 4)  # 椭圆

        cv2.imwrite( "54_" + filenameSize + ".jpg", img)

        img = cv2.imread( "54_" + filenameSize + ".jpg")

        cv2.circle(img, (x, y), 0, (255, 255, 255), -1)

        cv2.imwrite("result.jpg", img)

def img_laser_effetc(img, r, g, b, width, height, path):

    # r, g, b, = 0, 0, 0

    for i in range(0, 35):

        x = random.randint(0, width)
        y = random.randint(0, height)
        # r = random.randint(0, 255)
        # g = random.randint(0, 255)
        # b = random.randint(0, 255)

        cv2.ellipse(img, (x, y), (1, 1), 0, 0, 360, (r, g, b), 3, 4)  # 椭圆

        cv2.imwrite(path, img)

        img = cv2.imread(path)

        cv2.circle(img, (x, y), 0, (255, 255, 255), -1)

        cv2.imwrite(path, img)



