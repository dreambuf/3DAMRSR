###main code for 3DAMRSR training
import argparse, os
import time

import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage import img_as_ubyte
from AMRSR_x4_3d import AMRSR
from load_data import TrainDatasetFromFolder, TestDatasetFromFolder
import math
import matplotlib as mpl
from matplotlib import ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import io
import cv2
from math import log10
import easydict
from ssim3D import *
from test3D import *
from PIL import Image
from LS import lsloss

opt = easydict.EasyDict({
    "dir_train_data": '.\\data2\\sandstone3D\\TRAIN\\',
    "dir_test_data": '.\\data2\\sandstone3D\\TEST\\',
    "batchSize": 1,
    "nEpochs": 1,
    "lr": 1e-4,
    "step": 10,
    "start_epoch": 1,
    "momentum": 0.9,
    "cuda": 0,
    "threads": 1,
})
device = torch.device("cuda:{}".format(opt.cuda))

def main():
    global opt, model
    cuda = opt.cuda
    starttime = time.time()
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    train_set = TrainDatasetFromFolder(opt.dir_train_data)
    test_set = TestDatasetFromFolder(opt.dir_test_data)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    model = AMRSR()
    criterion = lsloss()
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(opt.start_epoch, opt.nEpochs + 40):
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
        lr = adjust_learning_rate(optimizer, epoch - 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        train(training_data_loader, optimizer, model, criterion, epoch)
        if epoch % 1 == 0:
            time1 = time.time() - starttime
            print("Epoch={}, lr={} , time = {}".format(epoch, optimizer.param_groups[0]["lr"], time1))
            test(test_data_loader, optimizer, model, criterion, epoch)
    torch.save(model.state_dict(), '.\\3D_AMRSR2.pt')
    endtime = time.time()
    print("消的时间：{}".format(endtime - starttime))

def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    model.train()

    for iteration, batch in enumerate(training_data_loader, 40):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        input = input.to(device)
        target = target.to(device)
        sr = model(input)
        loss = criterion(sr, target)
        loss = loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse = torch.mean((sr - target) ** 2)
        psnr = 10 * math.log10(1.0 / torch.mean(mse))



if __name__ == "__main__":
    main()
