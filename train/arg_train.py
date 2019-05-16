import logging
from os import path
from preprocess.arg_set import time_str

checkpoint_path = '/sfc_temp/check_point/'

# about learning rate
init_lr = 1e-2  # todo
momentum = 0.9  # todo
weight_decay = 5e-4
gamma = 10 ** -0.06  # final_lr = init_lr * (gamma ^ epoch)

# about sample numbers
batch_size = 8  # todo
iter_num = 53200 // 4  # // batch_size  # todo
epoch = 50
check_interval = 1  # how often to save checkpoint
num_workers = 16
statics_proportion = 0.1
statics_frame_num = 16

# about augmentation
rgb_noise = True
gray_trans = True
gray_proportion = 0.25
flip_trans = False  # todo
flip_proportion = 0.5
stretch_trans = True
stretch_max = 4
stretch_delta = 0.05

# about loss
rPos = 16
rNeg = 0
pair_frame_range = 100

# others
loss_report_iter = 100  # how often to report loss in cmd

checkpoint_path = path.normpath(checkpoint_path)
assert isinstance(batch_size, int)
assert isinstance(iter_num, int)
assert isinstance(epoch, int)
assert isinstance(num_workers, int)
assert isinstance(statics_frame_num, int)
assert isinstance(rPos, int)
assert isinstance(rNeg, int)
assert isinstance(pair_frame_range, int)
assert isinstance(check_interval, int)
assert isinstance(loss_report_iter, int)
assert isinstance(rgb_noise, bool)
assert isinstance(gray_trans, bool)
assert isinstance(flip_trans, bool)
assert isinstance(stretch_trans, bool)
assert isinstance(stretch_max, int)
assert init_lr > 0, 'Invalid learning rate %s' % init_lr
assert momentum >= 0, 'Invalid momentum = %s' % momentum
assert weight_decay >= 0, 'Invalid weight_decay=%s' % weight_decay
assert gamma > 0, 'Invalid gamma = %s' % gamma
assert gamma <= 1, 'Invalid gamma = %s' % gamma
assert batch_size > 0, 'Invalid batch_size = %d' % batch_size
assert iter_num > 1, 'iter_num = %d is not greater than 1' % iter_num
assert epoch > 0, 'Invalid epoch = %d' % epoch
assert num_workers > 0, 'Invalid num_workers = %d' % num_workers
assert num_workers <= 64, 'Too big num_workers = %d' % num_workers
assert gray_proportion >= 0, 'Invalid gray_proportion = %s' % gray_proportion
assert gray_proportion <= 1, 'Invalid gray_proportion = %s' % gray_proportion
assert flip_proportion >= 0, 'Invalid flip_proportion = %s' % flip_proportion
assert flip_proportion <= 1, 'Invalid flip_proportion = %s' % flip_proportion
assert stretch_max >= 0, 'Invalid stretch_max = %d' % stretch_max
assert stretch_delta >= 0, 'Invalid stretch_delta = %s' % stretch_delta
assert stretch_delta < 0.5, 'Invalid stretch_delta = %s' % stretch_delta
assert statics_frame_num > 0, 'Invalid statics_frame_num = %d' % statics_frame_num
assert statics_proportion > 0, 'Invalid statics_proportion = %s' % statics_proportion
assert statics_proportion <= 1, 'Invalid statics_proportion = %s' % statics_proportion
assert pair_frame_range > 0, 'Invalid pair_frame_range = %d' % pair_frame_range
assert check_interval > 0, 'Invalid check_interval = %d' % check_interval
assert check_interval <= epoch, 'Invalid check_interval = %d' % check_interval
assert loss_report_iter > 0, 'Invalid loss_report_iter = %d' % loss_report_iter
assert path.isdir(checkpoint_path), 'Invalid path to save checkpoint: %s' % checkpoint_path
checkpoint_path = path.join(checkpoint_path, time_str)
chk_file = path.join(checkpoint_path, r'%02d_epoch_chk.pth')
fig_file = path.join(checkpoint_path, r'loss_fig.pdf')

logging.info('%21s :%s' % ('Initial learning rate', init_lr))
logging.info('%21s :%s' % ('Final learning rate', init_lr * (gamma ** epoch)))
logging.info('%21s :%s' % ('Gamma', gamma))
logging.info('%21s :%s' % ('Weight Decay', weight_decay))
logging.info('%21s :%s' % ('Momentum', momentum))
logging.info('%21s :%d' % ('Epoch', epoch))
logging.info('%21s :%d' % ('Batch Size', batch_size))
logging.info('%21s :%d' % ('Iteration', iter_num))
logging.info('==================================================================')
logging.info('%21s :%s' % ('RGB color noise', rgb_noise))
logging.info('%21s :%s' % ('Gray transform', gray_trans))
if gray_trans:
    logging.info('%21s :%s' % ('Gray proportion', gray_proportion))
logging.info('%21s :%s' % ('Flip transform', flip_trans))
if flip_trans:
    logging.info('%21s :%s' % ('Flip proportion', flip_proportion))
logging.info('%21s :%s' % ('Stretch_transform', stretch_trans))
if stretch_trans:
    logging.info('%21s :%s' % ('Stretch max', stretch_max))
logging.info('==================================================================')
logging.info('%21s :%s' % ('Checkpoint path', checkpoint_path))
logging.info('%21s :%d epoches' % ('Save checkpoint every', check_interval))
logging.info('%21s :%s batches' % ('Report loss every', loss_report_iter))
logging.info('==================================================================')
