import os
import yaml
# from preprocess.arg_set import logging

yaml_path = os.path.join(os.path.dirname(__file__), 'arg_test.yaml')
with open(yaml_path, 'r') as f:
    cfg_file = f.read()
cfg = yaml.load(cfg_file, yaml.FullLoader)
model_path = cfg['model_path']
data_root = cfg['data_root']
result_dir = cfg['result_dir']
report_dir = cfg['report_dir']
name_tracker = cfg['name_tracker']
scale_step = cfg['scale_step']
scale_penalty = cfg['scale_penalty']
scale_lr = cfg['scale_lr']
hann_weight = cfg['hann_weight']
scale_num = cfg['scale_num']
interpolation_beta = cfg['interpolation_beta']
s_x_limit_beta = cfg['s_x_limit_beta']

# assert os.path.isfile(model_path), "Invalid path to model:%s" % model_path
assert isinstance(name_tracker, str)
# assert isinstance(scale_num, int)
# assert scale_num > 0, "Invalid scale level:%s" % scale_num
# assert scale_num % 2 == 1, "Scale level must be odd:%d" % scale_num
# if scale_num > 1:
#     assert scale_step > 1, "Invalid scale step:%s" % scale_step
# assert scale_penalty >= 0, "Invalid scale penalty:%s" % scale_penalty
# assert scale_penalty <= 1, "Invalid scale penalty:%s" % scale_penalty
# assert scale_lr >= 0, "Invalid scale learning rate:%s" % scale_lr
# assert scale_lr <= 1, "Invalid scale learning rate:%s" % scale_lr
# assert hann_weight >= 0, "Invalid hanning window weight:%s" % hann_weight
# assert hann_weight < 1, "Invalid hanning window weight:%s" % hann_weight
# assert s_x_limit_beta > 1, "Invalid s_x limit factor:%s" % s_x_limit_beta
#
# logging.info('%21s :%s' % ('Model path', model_path))
# logging.info('%21s :%s' % ('Scale step', scale_step))
# logging.info('%21s :%s' % ('Scale penalty', scale_penalty))
# logging.info('%21s :%s' % ('Scale learning rate', scale_lr))
# logging.info('%21s :%s' % ('Hanning window weight', hann_weight))
# logging.info('%21s :%d' % ('Scale level', scale_num))
# logging.info('%21s :%d' % ('Interpolation factor', interpolation_beta))
# logging.info('%21s :%s' % ('s_x limit factor', s_x_limit_beta))
