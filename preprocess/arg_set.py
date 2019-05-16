import os
import time
import logging

data_main_path = '/ILSVRC2015'
data_temp_path = '/sfc_temp/data_temp'
log_path = '/sfc_temp/log'
exemplar_size = 127
search_size = 255
total_stride = 8
corr_size = 17
context_amount = 0.5
worker_num = 32
chunksize = 100
frame_format = '%06d'
crop_z_format = '.%02d.crop.z.jpg'
crop_x_format = '.%02d.crop.x.jpg'
_CLASS_IDS = (
    'n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116', 'n02958343', 'n02402425', 'n02084071',
    'n02121808', 'n02503517', 'n02118333', 'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
    'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227', 'n02129604', 'n04468005', 'n01662784',
    'n04530566', 'n02062744', 'n02391049')
_CLASS_NAMES = (
    'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
    'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep',
    'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra')
CLASS_IDS = dict()
CLASS_NAMES = dict()
for i, n in enumerate(_CLASS_IDS):
    CLASS_IDS[n] = i
    CLASS_NAMES[i] = _CLASS_NAMES[i]

assert isinstance(total_stride, int)
assert isinstance(corr_size, int)
assert isinstance(worker_num, int)
assert isinstance(chunksize, int)
assert total_stride > 0, 'Invalid total_stride = %d' % total_stride
assert corr_size > 0, 'Invalid corr_size = %d' % corr_size
assert worker_num > 0, 'Invalid worker_num = %d' % worker_num
assert worker_num <= 64, 'Too big worker_num = %d' % worker_num
assert chunksize > 0, 'Invalid chunksize = %d' % chunksize
assert context_amount >= 0, 'Invalid context_amount = %s' % context_amount
assert context_amount <= 1, 'Invalid context_amount = %s' % context_amount
time_str = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())


def make_new_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.isfile(path):
        raise OSError('Failed to make dir:\n%s.\nThere exists a file with SAME name.' % path)
    logging.info('Make dir: %s' % path)


def log_set():
    """
    Set the log file and screen output
    :return:Command parameters unchanged
    """
    # Log file format
    format_set = logging.Formatter(fmt='%(asctime)s: %(filename)s-[line:%(lineno)d]-%(levelname)s: %(message)s',
                                   datefmt='%Y/%m/%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Log file name
    logfile = os.path.join(log_path, time_str + '.log')
    logfile = logging.FileHandler(filename=logfile, mode='w')
    logfile.setFormatter(format_set)
    logfile.setLevel(logging.DEBUG)
    logger.addHandler(logfile)
    # Screen output format
    format_set = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    cons = logging.StreamHandler()
    cons.setLevel(logging.INFO)
    cons.setFormatter(format_set)
    logger.addHandler(cons)
    logging.info('==================================================================')
    logging.info('%17s: %s' % ('Data Main Path', os.path.abspath(data_main_path)))
    logging.info('%17s: %s' % ('Data Temp Path', os.path.abspath(data_temp_path)))
    logging.info('%17s: %s' % ('Log path', os.path.abspath(log_path)))
    logging.info('%17s: %s' % ('Time Spot', time_str))
    logging.info('==================================================================')
    logging.info('%17s: %d' % ('Exemplar size', exemplar_size))
    logging.info('%17s: %d' % ('Search size', search_size))
    logging.info('%17s: %d' % ('Total stride', total_stride))
    logging.info('%17s: %d' % ('Correlation size', corr_size))
    logging.info('==================================================================')
    return logfile, logger


logfile_handler, logger_handler = log_set()
