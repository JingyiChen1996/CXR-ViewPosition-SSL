#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import torch
import torchvision

# in order to avoid complaining warning from tensorflow logger
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False


class SlackClientHandler(logging.Handler):

    def __init__(self, credential_file, ch_name):
        super().__init__()
        with open(credential_file, 'r') as f:
            tmp = yaml.safe_load(f)
        self.slack_token = tmp['token']
        self.slack_recipients = tmp['recipients']
        #self.slack_token = os.getenv("SLACK_API_TOKEN")
        #self.slack_user = os.getenv("SLACK_API_USER")
        if self.slack_token is None or self.slack_recipients is None:
            raise KeyError

        from slack import WebClient
        self.client = WebClient(self.slack_token)

        # getting user id
        ans = self.client.users_list()
        users = [u['id'] for u in ans['members'] if u['name'] in self.slack_recipients]
        # open DM channel to the users
        ans = self.client.conversations_open(users=','.join(users))
        self.channel = ans['channel']['id']
        ans = self.client.chat_postMessage(channel=self.channel, text=f"*{ch_name}*")
        self.thread = ans['ts']

    def emit(self, record):
        try:
            msg = self.format(record)
            self.client.chat_postMessage(channel=self.channel, thread_ts=self.thread, text=f"```{msg}```")
        except:
            self.handleError(record)


class MyFilter(logging.Filter):

    def __init__(self, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


class MyLogger(logging.Logger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addFilter(MyFilter(0))
        self.formatter = logging.Formatter('%(asctime)s %(rank)s [%(levelname)-5s] %(message)s')

    def set_rank(self, rank):
        self.removeFilter(self.filter)
        self.addFilter(MyFilter(rank))

    def set_log_to_stream(self, level=logging.DEBUG):
        chdr = logging.StreamHandler(sys.stdout)
        chdr.setLevel(level)
        chdr.setFormatter(self.formatter)
        self.addHandler(chdr)

    def set_log_to_file(self, log_file, level=logging.DEBUG):
        log_path = Path(log_file).resolve()
        Path.mkdir(log_path.parent, parents=True, exist_ok=True)
        fhdr = logging.FileHandler(log_path)
        fhdr.setLevel(level)
        fhdr.setFormatter(self.formatter)
        self.addHandler(fhdr)

    def set_log_to_slack(self, credential_file, ch_name, level=logging.INFO):
        try:
            credential_path = Path(credential_file).resolve()
            shdr = SlackClientHandler(credential_path, ch_name)
            shdr.setLevel(level)
            shdr.setFormatter(self.formatter)
            self.addHandler(shdr)
        except:
            raise RuntimeError


logging.setLoggerClass(MyLogger)
logger = logging.getLogger("pytorch-cxr")
logger.setLevel(logging.DEBUG)


def print_versions():
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info(f"torchvision version: {torchvision.__version__}")


def get_devices(cuda=None):
    if cuda is None:
        logger.info(f"use CPUs")
        return [torch.device("cpu")]
    else:
        assert torch.cuda.is_available()
        avail_devices = list(range(torch.cuda.device_count()))
        use_devices = [int(i) for i in cuda.split(",")]
        assert max(use_devices) in avail_devices
        logger.info(f"use cuda on GPU {use_devices}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in use_devices])
        return [torch.device(f"cuda:{k}") for k in use_devices]

def get_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def get_commit():
    import git
    repo = git.Repo(search_parent_directories=True)
    assert not repo.is_dirty(), "current repository has some changes. please make a commit to run"

    try:
        branch = repo.head.ref.name
    except TypeError:
        branch = "(detached)"
    sha = repo.head.commit.hexsha
    dttm = repo.head.commit.committed_datetime
    return f"{branch} / {sha} ({dttm})"


#from __future__ import print_function, absolute_import

#__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mixup(x, l, beta=0.75):
    assert x.shape[0] == l.shape[0]
    mix = torch.distributions.Beta(beta, beta).sample(
        (x.shape[0], )).to(x.device).view(-1, 1, 1, 1)

    mix = torch.max(mix, 1 - mix)
    perm = torch.randperm(x.shape[0])

    xmix = x * mix + x[perm] * (1 - mix)
    lmix = l * mix[..., 0, 0] + l[perm] * (1 - mix[..., 0, 0])

    return xmix, lmix

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

