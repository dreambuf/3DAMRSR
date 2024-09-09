import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import easydict
from ssim3D import *

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
def test(test_data_loader, optimizer, model, criterion, epoch):
    model.eval()
    psnrs = 0
    losss = 0
    t_SSIM_end = 0

    for iteration, batch in enumerate(test_data_loader, 1):
        t_SSIM = 0
        with torch.no_grad():
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            input = input.to(device)
            target = target.to(device)
            sr = model(input)
            loss = criterion(sr, target)
            loss = loss.to(device)
            mse = torch.mean((sr - target) ** 2)
            psnr = 10 * math.log10(1.0 / torch.mean(mse))
            psnrs += psnr
            losss += loss
        if epoch == 40 and (iteration % 2 == 0):
            lr1 = input.squeeze(0).permute(1, 0, 2, 3)
        sr = sr.squeeze(0).permute(1, 0, 2, 3)
        L = 100
        L = int(L)
        target = target.squeeze(0).permute(1, 0, 2, 3)

        for i in range(L):
            SSIM = ssim3D(sr[i], target[i])
            if epoch == 40 and (iteration % 2 == 0) and i == 99:
                j = 24
                imagel = lr1[j].cpu().numpy()
                images = sr[i].cpu().numpy()
                imageh = target[i].cpu().numpy()
                if images.shape[0] == 3:
                    imagel = (np.transpose(imagel, (1, 2, 0)) + 1) / 2.0 * 255.0
                    images = (np.transpose(images, (1, 2, 0)) + 1) / 2.0 * 255.0
                    imageh = (np.transpose(imageh, (1, 2, 0)) + 1) / 2.0 * 255.0
                elif images.shape[0] == 1:
                    imagel = (imagel[0] + 1) / 2.0 * 255.0
                    images = (images[0] + 1) / 2.0 * 255.0
                    imageh = (imageh[0] + 1) / 2.0 * 255.0

                imagel = imagel.astype(np.uint8)
                images = images.astype(np.uint8)
                imageh = imageh.astype(np.uint8)
                imagel = Image.fromarray(imagel)
                images = Image.fromarray(images)
                imageh = Image.fromarray(imageh)

                imagel.save("img" + "/" + str(iteration) + str(i) + 'lr' + ".png")
                images.save("img" + "/" + str(iteration) + str(i) + 'sr' + ".png")
                imageh.save("img" + "/" + str(iteration) + str(i) + 'hr' + ".png")

            t_SSIM += SSIM
        t_SSIM_end += t_SSIM / L

    long = len(test_data_loader)
    loss_avg = losss / long
    psnr_avg = psnrs / long
    ssim_avg = t_SSIM_end / long
    if epoch % 1 == 0:
        print("{}:{}:{:.10f}:{:.10f}:{}".format(epoch, iteration, loss_avg, psnr_avg, ssim_avg))