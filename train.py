import argparse

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from deeplab.model import Deeplab, Deeplabv3
from deeplab.datasets import ImageDataSet
import time

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--data_dir", type=str, default="./datasets")
parser.add_argument("--data_list", type=str, default="./datasets/train.txt")
parser.add_argument("--ignore_label", type=int, default=255)
parser.add_argument("--input_size", type=str, default="300,300")
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--power", type=float, default=0.9)
parser.add_argument("--random_mirror", action="store_true")
parser.add_argument("--random_scale", action="store_true")
parser.add_argument("--snapshot_dir", type=str, default="./snapshots")
parser.add_argument("--save_steps", type=int, default=50)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--v3", action="store_true")
parser.add_argument("--distributed", action="store_true")
args = parser.parse_args()


def loss_calc(pred, label, gpu):
    label = Variable(label.long())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model, distributed=False):
    b = []

    if distributed:
        b.append(model.module.conv1)
        b.append(model.module.bn1)
        b.append(model.module.layer1)
        b.append(model.module.layer2)
        b.append(model.module.layer3)
        b.append(model.module.layer4)
    else:
        b.append(model.conv1)
        b.append(model.bn1)
        b.append(model.layer1)
        b.append(model.layer2)
        b.append(model.layer3)
        b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model, distributed=False):
    b = []

    if distributed:
        b.append(model.module.layer5.parameters())
    else:
        b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def adjust_lr(optimizer, i_iter):
    lr = lr_poly(args.lr, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    if args.gpu >= 0:
        cudnn.enabled = True

    if args.v3:
        model = Deeplabv3(num_classes=args.num_classes)
    else:
        model = Deeplab(num_classes=args.num_classes)

    model.train()

    if args.gpu >= 0:
        if args.distributed:
            model = nn.DataParallel(model).cuda()
        else:
            model.cuda(args.gpu)
        cudnn.benchmark = True

    if args.model is not None:
        saved_state_dict = torch.load(args.model)
        model.load_state_dict(saved_state_dict)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        ImageDataSet(args.data_dir, args.data_list,
                     max_iters=args.num_steps*args.batch_size,
                     crop_size=input_size,
                     scale=args.random_scale,
                     mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True,
        num_workers=5, pin_memory=False)

    optimizer = optim.SGD(
        [{'params': get_1x_lr_params_NOscale(model, True), 'lr': args.lr},
         {'params': get_10x_lr_params(model, True), 'lr': 10*args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear')

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.cuda(async=True)
        labels = labels.cuda(async=True)
        images = Variable(images)

        optimizer.zero_grad()
        adjust_lr(optimizer, i_iter)
        pred = interp(model(images))
        loss = loss_calc(pred, labels, args.gpu)

        print("Step: {}, Loss: {}".format(i_iter,
                                          float(loss.data)))
        loss.backward()
        optimizer.step()

        if i_iter >= args.num_steps-1:
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir,
                                'model_' + str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_steps == 0 and i_iter != 0:
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir,
                                'model_' + str(i_iter)+'.pth'))


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("Elapsed time:{0}".format(elapsed_time) + "[sec]")
