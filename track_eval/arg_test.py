import os
from preprocess.arg_set import logging

model_path = '/sfc_temp/check_point/2019_05_01_16-53-19/50_epoch_chk.pth'
data_root = '/sfc_temp/OTB'
result_dir = '/sfc_temp/results'
report_dir = '/sfc_temp/reports'
scale_step = 1.0375
scale_penalty = 0.9745
scale_lr = 0.59
hann_weight = 0.176
scale_num = 3
interpolation_beta = 16
s_x_limit_beta = 5

assert os.path.isfile(model_path), "Invalid path to model:%s" % model_path
assert isinstance(scale_num, int)
assert scale_num > 0, "Invalid scale level:%s" % scale_num
assert scale_num % 2 == 1, "Scale level must be odd:%d" % scale_num
if scale_num > 1:
    assert scale_step > 1, "Invalid scale step:%s" % scale_step
assert scale_penalty >= 0, "Invalid scale penalty:%s" % scale_penalty
assert scale_penalty <= 1, "Invalid scale penalty:%s" % scale_penalty
assert scale_lr >= 0, "Invalid scale learning rate:%s" % scale_lr
assert scale_lr <= 1, "Invalid scale learning rate:%s" % scale_lr
assert hann_weight >= 0, "Invalid hanning window weight:%s" % hann_weight
assert hann_weight < 1, "Invalid hanning window weight:%s" % hann_weight
assert s_x_limit_beta > 1, "Invalid s_x limit factor:%s" % s_x_limit_beta

logging.info('%21s :%s' % ('Model path', model_path))
logging.info('%21s :%s' % ('Scale step', scale_step))
logging.info('%21s :%s' % ('Scale penalty', scale_penalty))
logging.info('%21s :%s' % ('Scale learning rate', scale_lr))
logging.info('%21s :%s' % ('Hanning window weight', hann_weight))
logging.info('%21s :%d' % ('Scale level', scale_num))
logging.info('%21s :%d' % ('Interpolation factor', interpolation_beta))
logging.info('%21s :%s' % ('s_x limit factor', s_x_limit_beta))
