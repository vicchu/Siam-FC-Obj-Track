import os
import yaml

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

assert isinstance(name_tracker, str)
