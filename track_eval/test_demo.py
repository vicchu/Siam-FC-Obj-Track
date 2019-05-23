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


def plot_show(path_exp: str, start_index: int):
    gt_path = os.path.join(data_root, 'OTB', path_exp, 'groundtruth_rect.txt')
    ipath = os.path.join(data_root, 'OTB', path_exp, 'img/%04d.jpg')
    # showfig_path = os.path.join('/sfc_temp/show', path_exp, '%04d.png')
    with open(gt_path, 'r') as f:
        a = f.readlines()
    b = [s.split(',') for s in a]
    c = [num2(s) for s in b]
    n = len(a)
    print(n)
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
        print(k)
        if k < start_index:
            continue
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
        if k > (start_index + 10):
            break


if __name__ == '__main__':
    plot_show('Boy', 500)
    plot_show('Car1', 500)
    plot_show('Human6', 400)
    plot_show('Panda', 200)
    plot_show('Tiger2', 200)
