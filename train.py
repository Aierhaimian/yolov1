# -*-coding: utf-8-*-

"""
YOLO v1 train.
by Earl Du
duzexu92@gmail.com
"""

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from vgg import vgg16, vgg16_bn
from resnet import resnet50, resnet18
from loss import yolov1Loss
from utils.dataset import yolov1Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_gpu = torch.cuda.is_available()

file_root = '/home/dzx/workSpace/dataset/VOCdevkit/VOC2012/JPEGImages/'
learning_rate = 0.001
num_epochs = 50
batch_size = 24
use_resnet = True

if use_resnet:
    net = resnet50()
else:
    net = vgg16_bn()

print(net)
print('load pre-trained model')
if use_resnet:
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and not k.startswith('fc'):
            print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
else:
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and k.startswith('features'):
            print('yes')
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
if False:
    net.load_state_dict(torch.load('best.pth'))
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yolov1Loss(7, 2, 5, 0.5)
if use_gpu:
    net.cuda()

params = []
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]

optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

train_dateset = yolov1Dataset(root=file_root, list_file='voc2012.txt', train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dateset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = yolov1Dataset(root=file_root, list_file='voc2007_test.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print('The dataset has %d images' % (len(train_dateset)))
print('The batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001
    for param_group in optimizer.param_group:
        params['ir'] = learning_rate

    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0

    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0], total_loss / (i + 1)))
            num_iter += 1

        # validation
        validation_loss = 0.0
        net.eval()
        for i, (images, target) in enumerate(test_loader):
            images = Variable(images, volatile=True)
            target = Variable(target, volatile=True)
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            loss = criterion(pred, target)
            validation_loss += loss.data[0]
        validation_loss /= len(test_loader)

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), 'best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()
        torch.save(net.state_dict(), 'yolo.pth')
