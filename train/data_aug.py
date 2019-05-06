import os
import cv2
import torch
import random
import logging
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop
from train import arg_train
from preprocess import arg_set
from preprocess.ILSVRC2015.crop_video import total_dataset_info
from preprocess.ILSVRC2015.video_list import get_frame_number, get_video_list


def make_label():
    """
    Give the fixed ground truth label in corr-feature map like this:
    b  b  b  b  b  b  b
    b  b  0  0  0  b  b
    b  0  a  a  a  0  b
    b  0  a  a  a  0  b
    b  0  a  a  a  0  b
    b  b  0  0  0  b  b
    b  b  b  b  b  b  b
    :return:
    """
    corr_size = arg_set.corr_size
    if arg_train.stretch_trans:  # todo
        corr_size -= 2
    label = np.zeros([corr_size, corr_size], dtype=np.float32)
    lb_weight = label.copy()
    index = np.array(range(corr_size)) - np.floor(corr_size / 2)
    index = np.square(index)
    x_index = np.tile(index, [corr_size, 1])
    distance = x_index + np.transpose(x_index)
    pos_region = distance <= (arg_train.rPos / arg_set.total_stride)
    neg_region = distance > (arg_train.rNeg / arg_set.total_stride)
    num_pos = np.count_nonzero(pos_region)
    num_neg = np.count_nonzero(neg_region)
    if num_neg > 0:
        label[neg_region] = -1
        lb_weight[neg_region] = 0.5 / num_neg
    if num_pos > 0:
        label[pos_region] = 1 / num_pos
        lb_weight[pos_region] = 0.5 / num_pos
    label = np.tile(label, [arg_train.batch_size, 1, 1, 1])
    lb_weight *= (num_pos + num_neg)  # todo
    lb_weight = np.tile(lb_weight, [arg_train.batch_size, 1, 1, 1])
    label = torch.from_numpy(label)
    lb_weight = torch.from_numpy(lb_weight)
    return label, lb_weight


truth_label, label_weight = make_label()


def random_stretch(img, crop_before_resize: int, org_size: int):
    new_size = (2 * random.random() - 1) * arg_train.stretch_delta + 1
    new_size *= crop_before_resize
    new_size = min(int(new_size), crop_before_resize)
    gap = min((org_size - new_size) // 2, arg_train.stretch_max)
    if gap > 0:
        top = (org_size - new_size) // 2
        left = top
        top += random.randint(-gap, gap)
        left += random.randint(-gap, gap)
        new_img = img[top:(top + new_size), left:(left + new_size)]
    else:
        new_img = img[:new_size, :new_size]
    if new_size != crop_before_resize:
        new_img = cv2.resize(new_img, (crop_before_resize, crop_before_resize))
    return new_img


class TrainPair(Dataset):
    def __init__(self, *data_phase):
        # get videos list and xml files list
        v_l, x_l = get_video_list(arg_set.data_main_path)
        # get frame numbers list
        f_l = get_frame_number(v_l, arg_set.data_main_path)

        # get info
        data_info = total_dataset_info(*data_phase, v_l=v_l, x_l=x_l, f_l=f_l)
        assert data_info, 'Empty data set! Maybe caused by invalid phase or original data.'

        # store info
        self.info, self.video_list, self.xml_list, self.frame_num_list = [], [], [], []
        for phase in data_phase:
            self.info.extend(data_info[phase])
            self.video_list.extend(v_l[phase])
            self.xml_list.extend(x_l[phase])
            self.frame_num_list.extend(f_l[phase])

        # statics computation: image average, channel average, channel variance
        if arg_train.rgb_noise:
            self.z_im_avg, self.z_chl_avg, self.z_var_rgb, \
            self.x_im_avg, self.x_chl_avg, self.x_var_rgb = self.stats_data()

    def __len__(self):
        return len(self.info)

    def __getitem__(self, item, dumb=False, statics_model=''):
        """
        return a pair of frames sampled randomly in the selected video
        :param item: index for the video
        :param dumb: True for only return the path
        :param statics_model: only return z_frame index
        :return:
        """
        if statics_model:
            assert statics_model in ['z', 'x'], "Invalid statics_model:%s" % statics_model

        video_path = os.path.join(arg_set.data_temp_path, self.info[item]['path'])
        trackid = random.sample(self.info[item]['trackid'], 1)[0]
        frame_set = self.info[item]['frame_idx'][trackid]
        z_frame = random.sample(frame_set, 1)[0]  # randomly sample a frame as z_frame
        frame_set = np.array(frame_set, dtype=np.int)
        frame_set = frame_set[np.abs(frame_set - z_frame) <= arg_train.pair_frame_range]
        frame_set = frame_set[frame_set != z_frame]  # x_frame should be near to z_frame
        if len(frame_set) < 1:
            x_frame = z_frame  # todo
        else:
            x_frame = np.random.choice(frame_set, 1)[0]

        # get the path to frame pair
        z_path = os.path.join(video_path, arg_set.frame_format % z_frame + arg_set.crop_z_format % trackid)
        x_path = os.path.join(video_path, arg_set.frame_format % x_frame + arg_set.crop_x_format % trackid)

        # dumb is True in statics computation model, return path for simplification
        if dumb:
            if statics_model == 'z':
                return z_path
            elif statics_model == 'x':
                # replace the z_path with x_format, not x_path !!!
                return os.path.join(video_path, arg_set.frame_format % z_frame + arg_set.crop_x_format % trackid)
            else:
                pair_frame = {'z_path': z_path, 'x_path': x_path}
        else:
            # read images
            z_img = cv2.imread(z_path, cv2.COLOR_GRAY2BGR)
            x_img = cv2.imread(x_path, cv2.COLOR_GRAY2BGR)

            # transfer the color images to gray images randomly
            if arg_train.gray_trans and random.random() < arg_train.gray_proportion:
                z_img = cv2.cvtColor(z_img, cv2.COLOR_BGR2GRAY)
                z_img = cv2.cvtColor(z_img, cv2.COLOR_GRAY2BGR)
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
            if arg_train.stretch_trans:
                z_img = random_stretch(z_img, arg_set.exemplar_size, arg_set.exemplar_size)
                x_img = random_stretch(x_img, arg_set.search_size - 2 * arg_set.total_stride, arg_set.search_size)
            z_img = z_img.astype(np.float32)
            x_img = x_img.astype(np.float32)

            # random flip
            if arg_train.flip_trans and random.random() < arg_train.flip_proportion:
                z_img = z_img[:, ::-1, :]
                x_img = x_img[:, ::-1, :]

            # add random noise
            if arg_train.rgb_noise:
                bgr_offset = self.z_var_rgb.dot(np.random.rand(3, 2))
                z_img -= bgr_offset[:, 0]
                x_img -= bgr_offset[:, 1]

            # transfer images to tensors
            z_img = z_img.transpose([2, 0, 1])
            x_img = x_img.transpose([2, 0, 1])
            z_img = torch.from_numpy(z_img.copy())
            x_img = torch.from_numpy(x_img.copy())
            pair_frame = {'z_img': z_img, 'x_img': x_img, 'z_path': z_path, 'x_path': x_path}
        return pair_frame

    def _mean_var_(self, video_index_sample, statics_model):
        # image average, channel average, channel variance
        im_avg, chl_avg, chl_corr = [], [], []
        n = len(video_index_sample)
        logging.info('WAITING: statics data for statics_model: %s' % statics_model)
        with Pool(arg_set.worker_num) as p:
            # compute for each video
            for i, v in enumerate(video_index_sample):
                sample_path = [self.__getitem__(v, True, statics_model) for i in range(arg_train.statics_frame_num)]
                im = p.map(partial(cv2.imread, flags=cv2.COLOR_GRAY2BGR), sample_path)
                # im = [cv2.imread(x, flags=cv2.COLOR_GRAY2BGR) for x in sample_path]  # todo equal to the line above
                im = np.asarray(im, dtype=np.float32)
                im_avg.append(im.mean(0))
                chl_avg.append(im.mean((0, 1, 2)))
                corr = im.transpose([3, 0, 1, 2]).reshape(3, -1)
                chl_corr.append(np.cov(corr))
                print('%d/%d videos in statics computation' % (i, n), end='\r')

        # compute statics data on all samples
        im_avg = np.asarray(im_avg, dtype=np.float32).mean(0)
        chl_avg = np.asarray(chl_avg, dtype=np.float32).mean(0)
        chl_corr = np.asarray(chl_corr, dtype=np.float32).mean(0)
        eig, vec = np.linalg.eigh(chl_corr)
        eig = np.sign(eig) * np.sqrt(np.sign(eig) * eig)
        var_rgb = 0.1 * np.dot(np.diag(eig), vec.transpose())
        logging.info('FINISHED: statics data for statics_model: %s' % statics_model)
        return im_avg, chl_avg, var_rgb

    def stats_data(self):
        """
        compute statics data on z_frame and x_frame samples
        :return: image average, channel average, channel variance
        """
        num_sample = int(round(len(self.info) * arg_train.statics_proportion))
        assert num_sample > 1, 'Too small statics_proportion = %s' % arg_train.statics_proportion
        logging.info('Number of sampled videos for statics computation: %d' % num_sample)
        z_video_index_sample = random.sample(list(range(len(self.info))), num_sample)
        x_video_index_sample = random.sample(list(range(len(self.info))), num_sample)
        z_im_avg, z_chl_avg, z_var_rgb = self._mean_var_(z_video_index_sample, 'z')
        x_im_avg, x_chl_avg, x_var_rgb = self._mean_var_(x_video_index_sample, 'x')
        return z_im_avg, z_chl_avg, z_var_rgb, x_im_avg, x_chl_avg, x_var_rgb
