import argparse
import time
import numpy as np
import cv2

import torch
from torch.autograd import Variable
from torch.utils import data
from deeplab.model import Res_Deeplab
from deeplab.datasets import ImageDataSet

import matplotlib as mpl
mpl.use('tkagg')
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


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

def show_all(gt, pred):
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

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--data-list", type=str, default="./datasets/val.txt")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    args =  parser.parse_args()

    model = Res_Deeplab(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.model)
    model.load_state_dict(saved_state_dict)

    model.eval()
    if args.gpu >= 0:
        model.cuda(args.gpu)

    testloader = data.DataLoader(ImageDataSet(args.data_dir, args.data_list, crop_size=(720, 1280),
                                              mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=False)

    interp = nn.Upsample(size=(720, 1280), mode='bilinear')
    data_list = []

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, label, size, name = batch
        size = size[0].numpy()

        start = time.time()
        output = model(Variable(image, volatile=True))
        output = interp(output).cpu().data[0].numpy()

        output = output[:,:size[0],:size[1]]

        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)

        output = output.transpose(1,2,0)
        print(np.where(output>0))
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        show_all(gt, output)
        data_list.append([gt.flatten(), output.flatten()])

    get_iou(data_list, args.num_classes)


if __name__ == '__main__':
    main()
