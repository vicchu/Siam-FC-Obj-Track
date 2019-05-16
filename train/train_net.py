import sys
import torch
import os.path
import logging
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from train import arg_train
from preprocess.arg_set import make_new_dir
from architecture.net_structure import Siamese
from train.data_aug import truth_label, label_weight, TrainPair

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_pixel = nn.SoftMarginLoss(reduction='none')
truth_label = truth_label.to(device)
label_weight = label_weight.to(device)


def loss_batch(predict_score):
    """
    compute batch loss
    :param predict_score: output of the net
    :return: batch loss
    """
    weighted_loss = label_weight * loss_pixel(predict_score, truth_label)
    loss_mean = weighted_loss.sum([1, 2, 3]).mean()
    return loss_mean


def save_checkpoint(epoch_now: int, net_model, loss_value, optimizer, lr_stepper=None):
    """
    save checkpoint once every some ireations
    :param epoch_now: epoch number now
    :param net_model: the net
    :param loss_value: the batch loss
    :param optimizer: the optimizer (SGD)
    :param lr_stepper: the learning rate scheduler
    :return: None
    """
    if epoch_now % arg_train.check_interval == 0:
        s_path = arg_train.chk_file % epoch_now
        if lr_stepper is None:
            torch.save(
                {'epoch': epoch_now,
                 'model_state_dict': net_model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss_value},
                s_path)
        else:
            torch.save(
                {'epoch': epoch_now,
                 'model_state_dict': net_model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'lr_state_dict': lr_stepper.state_dict(),
                 'loss': loss_value},
                s_path)
        logging.info('Save checkpoint: %s' % s_path)


def weight_bias_param_init(network):
    param_set = []
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            # the init method
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # todo
            nn.init.zeros_(m.bias)
            param_set.append({'params': m.weight})
            # for bias, lr is doubled, but there's no weight decay
            param_set.append({'params': m.bias, 'lr': 2 * arg_train.init_lr, 'weight_decay': 0})
        elif isinstance(m, nn.BatchNorm2d):
            # for BN, weight = 1, and bias = 0 in initialization
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            # there's no weight decay for BN
            param_set.append({'params': m.weight, 'weight_decay': 0})
            param_set.append({'params': m.bias, 'lr': 2 * arg_train.init_lr, 'weight_decay': 0})
    logging.info('FINISHED: initialize the parameters')
    return param_set


def loss_plot(loss_record):
    """
    Plot and save loss curve figure file
    :param loss_record: the loss reported along the train phase
    :return: None
    """
    x_axis = [(1 + x) * arg_train.epoch / len(loss_record) for x in range(len(loss_record))]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_axis, loss_record)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig.savefig(arg_train.fig_file)
    logging.info('Savefig loss fig file: %s' % arg_train.fig_file)


if __name__ == '__main__':
    net = Siamese()
    logging.info('Network Structure')
    logging.info(str(net))
    nGPU = torch.cuda.device_count()
    if torch.cuda.is_available() and nGPU > 1:
        net = nn.DataParallel(net)
    net.to(device)
    logging.info('Device: ' + str(device))
    net.train(True)
    logging.info('Set TRAIN model')
    logging.info('==================================================================')
    logging.info('WAITING: Loading dataset...')
    train_data = TrainPair('val', 'train')  # todo use val and train dataset
    train_loader = DataLoader(train_data,
                              num_workers=arg_train.num_workers,  # comment this line when debug
                              batch_sampler=BatchSampler(
                                  sampler=RandomSampler(train_data,
                                                        replacement=True,
                                                        num_samples=arg_train.batch_size * arg_train.iter_num * arg_train.epoch),
                                  batch_size=arg_train.batch_size,
                                  drop_last=True))
    dict_para = weight_bias_param_init(net)
    sgd_opt = optim.SGD(dict_para,
                        lr=arg_train.init_lr,
                        # dampening=arg_train.momentum,  # todo damp = momentun, otherwise the SGD will output NaN Loss!
                        momentum=arg_train.momentum,
                        weight_decay=arg_train.weight_decay)
    lr_adjust = optim.lr_scheduler.StepLR(sgd_opt, step_size=1, gamma=arg_train.gamma)

    make_new_dir(arg_train.checkpoint_path)
    loss_report = 0
    loss_fig = []
    n = 0

    for k, sp in enumerate(train_loader):
        epoch = k // arg_train.iter_num + 1
        if (k + 1) % arg_train.iter_num == 0:
            save_checkpoint(epoch, net, loss, sgd_opt, lr_adjust)
            lr_adjust.step()
        net.zero_grad()
        z_img = sp['z_img'].to(device)
        x_img = sp['x_img'].to(device)
        score = net(x_img, z_img)
        loss = loss_batch(score)
        loss_report += float(loss.item())
        n += 1
        loss.backward()
        sgd_opt.step()
        if (k + 1) % arg_train.loss_report_iter == 0:
            logging.info('%d/%d:loss = %.5f' % (1 + k % arg_train.iter_num, epoch, loss_report / n))
            loss_fig.append(loss_report)
            loss_report = 0
            n = 0
    loss_plot(loss_fig)
