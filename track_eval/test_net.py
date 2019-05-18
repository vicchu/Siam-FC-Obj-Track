import os
import cv2
import sys
import torch
# import logging
import numpy as np
from torch import nn
from got10k.trackers import Tracker
from got10k.experiments import ExperimentOTB

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from track_eval import arg_test
from preprocess import arg_set
from preprocess.arg_set import logging
from preprocess.ILSVRC2015.crop_video import crop_obj_simple, cut_size
from architecture.net_structure import Siamese

cpu_device = torch.device('cpu')


class SFC(Tracker):
    def __init__(self, name, config=None):
        """
        Initialize the tracker
        :param name: name for a certain tracker, give DIFFERENT name each time to avoid rewriting the report record
        :param config: the parameter set
        """
        assert isinstance(name, str)
        super(SFC, self).__init__(name=name, is_deterministic=True)
        self.model_path = arg_test.model_path
        self.scale_step = arg_test.scale_step
        self.scale_penalty = arg_test.scale_penalty
        self.scale_lr = arg_test.scale_lr
        self.hann_weight = arg_test.hann_weight
        self.scale_num = arg_test.scale_num
        self.interpolation_beta = arg_test.interpolation_beta
        self.s_x_limit_beta = arg_test.s_x_limit_beta
        self.new_arg(config)
        net = Siamese()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        model_weight = torch.load(self.model_path, map_location=device)
        # the model maybe CUDA or CPU type, use "try" to confirm it
        try:
            net.load_state_dict(model_weight['model_state_dict'])
        except RuntimeError:
            net = nn.DataParallel(net)
            net.to(device)
            net.load_state_dict(model_weight['model_state_dict'])
            net = net.module
        net.eval()
        self.net = net
        self.device = device
        self.center = None  # [x_c, y_c], coordinate of the object center
        self.target_size = None  # [w, h], width and height of the object
        self.target_size_min = None  # lowest bound of the size
        self.target_size_max = None  # highest bound of the size
        # scale factor for each scale channel
        self.scale_alpha = [self.scale_step ** x for x in
                            range(-(self.scale_num - 1) // 2, 1 + (self.scale_num - 1) // 2)]
        scale_penalty = [self.scale_penalty] * self.scale_num
        scale_penalty[(self.scale_num - 1) // 2] = 1
        # scale penalty factor for each scale channel
        self.scale_penalty = np.asarray(scale_penalty).reshape([1, 1, self.scale_num])
        # size of interpolated feature map
        score_size = arg_set.corr_size * self.interpolation_beta
        hann = np.outer(np.hanning(score_size), np.hanning(score_size))
        self.hann = hann / hann.sum()  # hanning window whose sum is 1
        self.score_size = (score_size, score_size)
        self.box = None  # [xmin, ymin, width ,height] and 1-indexed! The box coordinate for the object
        self.width = None
        self.height = None
        self.s_z = 0  # used in cropping the exemplar image
        self.s_x = 0  # used to record the current scale for the object
        self.s_x_min = 0
        self.s_x_max = 0

    def init(self, image, box):
        """
        # called by GOT10k
        :param image: the first frame of the object read by PIL, not by OpenCV!
        :param box: [xmin, ymin, width ,height], Warning: xmin and ymin are 1-indexed
        :return: None
        """
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # we use BGR in training
        self.box = box  # [xmin, ymin, width ,height] and 1-indexed!!!
        self.target_size = [box[2], box[3]]
        # the bound of size
        self.target_size_min = (box[2] / self.s_x_limit_beta, box[3] / self.s_x_limit_beta)
        self.target_size_max = (box[2] * self.s_x_limit_beta, box[3] * self.s_x_limit_beta)
        # the center coordinate of object, 0-indexed!!!
        self.center = [box[0] - 1 + (box[2] - 1) / 2, box[1] - 1 + (box[3] - 1) / 2]

        # the crop param
        self.s_z, self.s_x = cut_size(box[2], box[3])
        self.s_x_min, self.s_x_max = self.s_x / self.s_x_limit_beta, self.s_x * self.s_x_limit_beta
        self.height, self.width = image.shape[:2]
        img_mean_z = np.mean(image, (0, 1))

        # crop an exemplar image and extend it by scale_num times
        img_z = crop_obj_simple(image, self.s_z, self.center, image.shape[1::-1], arg_set.exemplar_size, img_mean_z)
        with torch.no_grad():
            img_z = torch.from_numpy(img_z.astype(np.float32)).to(self.device)
            img_z = img_z.permute(2, 0, 1).reshape(1, 3, arg_set.exemplar_size, arg_set.exemplar_size).expand(
                self.scale_num, -1, -1, -1)
            self.net.set_target(img_z)

    def update(self, image):
        """
        output the tracking result for each frame
        :param image: the frame read by PIL!
        :return: the object box [xmin, ymin, width ,height] in 1-indexed
        """
        # read frame and crop it
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img_mean_x = np.mean(image, (0, 1))
        img_x = [
            crop_obj_simple(image, self.s_x * x, self.center, image.shape[1::-1], arg_set.search_size, img_mean_x)
            for x in self.scale_alpha]
        img_x = np.asarray(img_x, dtype=np.float32)

        # get the output from the net
        with torch.no_grad():
            img_x = torch.from_numpy(img_x).to(self.device)
            img_x = img_x.permute(0, 3, 1, 2)
            score = self.net.forward(img_x).to(cpu_device).numpy()

        score = score[:, 0, :, :].transpose(1, 2, 0)  # convert to H*W*C
        score = cv2.resize(score, self.score_size, interpolation=cv2.INTER_CUBIC)
        score *= self.scale_penalty

        # the scale index of the max value
        best_scale_index = np.unravel_index(score.argmax(), score.shape)[2]
        score = score[:, :, best_scale_index]
        score -= score.min()
        score /= score.sum()

        # add Hanning window and get the coordinate of max value
        score = score * (1 - self.hann_weight) + self.hann * self.hann_weight
        y_c, x_c = np.unravel_index(score.argmax(), score.shape)
        # coordinate in the original size feature map
        x_c /= self.interpolation_beta
        y_c /= self.interpolation_beta
        # shift the origin point to the center of the feature map
        x_c -= (arg_set.corr_size - 1) / 2
        y_c -= (arg_set.corr_size - 1) / 2
        # fix the scale
        x_c *= arg_set.total_stride * self.scale_alpha[best_scale_index]
        y_c *= arg_set.total_stride * self.scale_alpha[best_scale_index]
        # shift the origin point to left-top
        x_c += self.center[0]
        y_c += self.center[1]
        # center should not beyond the image
        self.center[0] = min(self.width - 1, max(0, x_c))
        self.center[1] = min(self.height - 1, max(0, y_c))
        # update the learning rate
        lr_beta = 1 - self.scale_lr + self.scale_lr * self.scale_alpha[best_scale_index]
        s_x = lr_beta * self.s_x  # update the current scale
        self.s_x = min(self.s_x_max, max(s_x, self.s_x_min))  # bound it
        # update the object size and bound it
        self.target_size[0] *= lr_beta
        self.target_size[1] *= lr_beta
        self.target_size[0] = min(self.target_size_max[0], max(self.target_size_min[0], self.target_size[0]))
        self.target_size[1] = min(self.target_size_max[1], max(self.target_size_min[1], self.target_size[1]))
        # the coordinate of the left-top point in the box
        xmin = self.center[0] - (self.target_size[0] - 1) / 2
        ymin = self.center[1] - (self.target_size[1] - 1) / 2
        # convert to 1-indexed!
        self.target_size[0] = min(self.target_size[0], image.shape[1])
        self.target_size[1] = min(self.target_size[1], image.shape[0])
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        self.box = [xmin + 1, ymin + 1, self.target_size[0], self.target_size[1]]  # 1-indexed!!!

        return self.box

    def new_arg(self, config=None):
        if config is not None:
            self.scale_step = config[0]
            self.scale_penalty = config[1]
            self.scale_lr = config[2]
            self.hann_weight = config[3]
        self.check_arg()

    def check_arg(self):
        assert os.path.isfile(self.model_path), "Invalid path to model:%s" % self.model_path
        assert isinstance(self.scale_num, int)
        assert self.scale_num > 0, "Invalid scale level:%s" % self.scale_num
        assert self.scale_num % 2 == 1, "Scale level must be odd:%d" % self.scale_num
        if self.scale_num > 1:
            assert self.scale_step > 1, "Invalid scale step:%s" % self.scale_step
        assert self.scale_penalty >= 0, "Invalid scale penalty:%s" % self.scale_penalty
        assert self.scale_penalty <= 1, "Invalid scale penalty:%s" % self.scale_penalty
        assert self.scale_lr >= 0, "Invalid scale learning rate:%s" % self.scale_lr
        assert self.scale_lr <= 1, "Invalid scale learning rate:%s" % self.scale_lr
        assert self.hann_weight >= 0, "Invalid hanning window weight:%s" % self.hann_weight
        assert self.hann_weight < 1, "Invalid hanning window weight:%s" % self.hann_weight
        assert self.s_x_limit_beta > 1, "Invalid s_x limit factor:%s" % self.s_x_limit_beta
        if __name__ == '__main__':
            logging.info('%21s :%s' % ('Model path', self.model_path))
            logging.info('%21s :%s' % ('Scale step', self.scale_step))
            logging.info('%21s :%s' % ('Scale penalty', self.scale_penalty))
            logging.info('%21s :%s' % ('Scale learning rate', self.scale_lr))
            logging.info('%21s :%s' % ('Hanning window weight', self.hann_weight))
            logging.info('%21s :%d' % ('Scale level', self.scale_num))
            logging.info('%21s :%d' % ('Interpolation factor', self.interpolation_beta))
            logging.info('%21s :%s' % ('s_x limit factor', self.s_x_limit_beta))


if __name__ == '__main__':
    tracker = SFC(arg_test.name_tracker)
    exp = ExperimentOTB(arg_test.data_root,
                        version=2013,
                        result_dir=arg_test.result_dir,
                        report_dir=arg_test.report_dir)
    exp.run(tracker)
    exp.report([tracker.name])
