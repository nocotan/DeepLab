import os
import argparse
import time
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')

import torch
from torch.autograd import Variable
from torch.utils import data
from deeplab.model import Deeplab, Deeplabv3
from deeplab.datasets import ImageDataSet

import torch.nn as nn
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')


def show_all(gt, pred, index):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    # plt.show()
    plt.savefig("outputs/result/result" + str(index) + ".png")


def get_rt_and_lb(rt, lb, output):
    diff_rt = 1e+9
    diff_lb = 1e+9
    res_rt = None
    res_lb = None
    for i in range(len(output[0])):
        xi = output[1][i]
        yi = output[0][i]
        diff_i_rt = abs(rt[0] - xi) + abs(rt[1] - yi)
        diff_i_lb = abs(lb[0] - xi) + abs(lb[1] - yi)
        if diff_i_rt < diff_rt:
            diff_rt = diff_i_rt
            res_rt = (xi, yi)
        if diff_i_lb < diff_lb:
            diff_lb = diff_i_lb
            res_lb = (xi, yi)

    return res_rt, res_lb


def get_corner(output):
    top = output[0].min()
    left = output[1].min()
    bottom = output[0].max()
    right = output[1].max()

    return (left, top), (right, top), (left, bottom), (right, bottom)


def show_circle(corner, fname, name, index):
    print(fname)
    img = cv2.imread(fname)
    # print(img.shape)
    for i in range(4):
        cv2.circle(img, corner[i], 15, (0, 0, 255, 10))
    cv2.imwrite("outputs/"+name+".jpg", img)
    # cv2.imwrite("outputs/"+"%04d" % index + ".jpg", img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--data-list", type=str, default="./datasets/val.txt")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--v3", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    args = parser.parse_args()

    if args.v3:
        model = Deeplabv3(num_classes=args.num_classes)
    else:
        model = Deeplab(num_classes=args.num_classes)

    if args.gpu >= 0:
        if args.distributed:
            model = nn.DataParallel(model).cuda()
        else:
            model.cuda(args.gpu)

    saved_state_dict = torch.load(args.model)
    model.load_state_dict(saved_state_dict)

    model.eval()

    testloader = data.DataLoader(ImageDataSet(args.data_dir, args.data_list, crop_size=(720, 1280),
                                              mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=False)

    interp = nn.Upsample(size=(720, 1280), mode='bilinear')
    data_list = []

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % (index))
        image, label, size, name = batch
        image = image.cuda()
        label = label
        size = size[0].numpy()

        start = time.time()
        output = model(Variable(image, volatile=True))
        output = interp(output).cpu().data[0].numpy()

        #output = output[:, :size[0], :size[1]]

        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        target = np.where(output > 0)
        corner = get_corner(target)
        print(corner)
        rt, lb = get_rt_and_lb(corner[1], corner[2], target)
        corner = (corner[0], rt, lb, corner[3])

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        show_circle(corner, os.path.join(args.data_dir, "images/"+name[0]+".jpg"), name[0], index)

        show_all(gt, output, index)
        data_list.append([gt.flatten(), output.flatten()])

    get_iou(data_list, args.num_classes)


if __name__ == '__main__':
    main()
