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
from deeplab.model import Res_Deeplab
from deeplab.datasets import ImageDataSet
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--data-list", type=str, default="./datasets/train.txt")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--input-size", type=str, default="321,321")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--snapshot-dir", type=str, default="./snapshots")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    label = Variable(label.long())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    b = []

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
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    if args.gpu >= 0:
        cudnn.enabled = True

    model = Res_Deeplab(num_classes=args.num_classes)

    if args.model is not None:
        saved_state_dict = torch.load(args.model)
        model.load_state_dict(saved_state_dict)

    model.train()

    if args.gpu >= 0:
        model.cuda(args.gpu)
        cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    trainloader = data.DataLoader(ImageDataSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=False)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.lr},
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}],
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear')


    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = Variable(images)

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        pred = interp(model(images))
        loss = loss_calc(pred, labels, args.gpu)
        print("Step: {}, Loss: {}".format(i_iter, float(loss.data)))
        loss.backward()
        optimizer.step()


        if i_iter >= args.num_steps-1:
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'model_'+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'model_'+str(i_iter)+'.pth'))

    end = timeit.default_timer()

if __name__ == '__main__':
    main()
