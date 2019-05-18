import os
import sys
import time
from PIL import Image
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from track_eval.test_net import *
from track_eval.arg_test import data_root


def num2(s):
    r = [int(x) for x in s]
    return r


if __name__ == '__main__':
    gt_path = os.path.join(data_root, 'Boy/groundtruth_rect.txt')
    ipath = os.path.join(data_root, 'Boy/img/%04d.jpg')
    # showfig_path = 'sfc_temp/show/Boy/%04d.pdf'
    with open(gt_path, 'r') as f:
        a = f.readlines()
    b = [s.split(',') for s in a]
    c = [num2(s) for s in b]
    n = len(a)
    # print(n)
    # exit()
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.add_patch(plt.Rectangle(
    #     (0.1, 0.1),  # (x,y)
    #     0.5,  # width
    #     0.5,  # height
    # ))
    tracker = SFC('SHOW01')
    for k in range(n):
        i = k + 1
        im = ipath % i
        im = Image.open(im)
        if i == 1:
            tracker.init(im, c[k])
            box = c[k]
        else:
            box = tracker.update(im)
            box = [x for x in box]
            box[0] -= 1
            box[1] -= 1
        # print(k)
        # if k < 200:
        #     continue
        fig = plt.figure(figsize=(im.size[0] / 72, im.size[1] / 72), dpi=72)
        ax = fig.add_subplot(111, aspect='equal')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.imshow(im)
        ax.add_patch(plt.Rectangle(c[k][:2], c[k][2], c[i][3], fill=False, edgecolor='r', linewidth=3))
        ax.add_patch(plt.Rectangle(box[:2], box[2], box[3], fill=False, edgecolor='g', linewidth=3))
        ax.axis('off')
        fig.show()
        # fig.savefig(showfig_path % k, dpi=300)
        time.sleep(0.1)
        ax.cla()
        fig.clf()
        # if k > 210:
        #     break
    # print(c)
