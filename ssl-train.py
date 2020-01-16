import argparse
import os
import shutil
import time
import random
import math

import numpy as np 
from tqdm import tqdm
import pdb
import bisect

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

# import model
from dnn121 import DenseNet121, MobileNet
# import dataset
from dataset import train_val_split, STANFORD_CXR_BASE, CxrDataset, CXR_unlabeled
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, mixup, interleave, interleave_offsets, linear_rampup
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='SSL Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenet',
                    help='model architecture: '+ ' (default: mobilnet)')
parser.add_argument('--model', '-m', metavar='MODEL', default='baseline',
                    help='model: '+' (default: baseline)', choices=['baseline', 'pi', 'mt', 'mixmatch'])
parser.add_argument('--split', default=False, type=bool, metavar='split',
                    help='whether to split the dataset (default: True)')
# Optimization options
parser.add_argument('--optim', '-o', metavar='OPTIM', default='adam',
                    help='optimizer: '+' (default: adam)', choices=['adam', 'sgd'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--weight_l1', '--l1', default=1e-3, type=float,
                    metavar='W1', help='l1 regularization (default: 1e-3)')
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--beta', default=0.75, type=float)

# Checkpoints 
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Method option
parser.add_argument('--n-labeled', type=int, default=1000,
                        help='Number of labeled data')
parser.add_argument('--l-num', type=int, default=999,
                        help='Number of labeled data After Split')
parser.add_argument('--u-num', type=int, default=9001,
                        help='Number of labeled data After Split')                       
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Number of labeled data')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')                       

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    agrs.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_prec1 = 0
best_val_prec1 = 0
best_val_prec1_t = 0
acc1_tr, losses_tr = [], []
losses_cl_tr, losses_x_tr, losses_u_tr = [], [], []
acc1_val, losses_val, losses_et_val = [], [], []
acc1_t_tr, acc1_t_val = [], []
learning_rate, weights_cl, weights = [], [], []

def main():

    global args, best_prec1, best_val_prec1, best_val_prec1_t
    global acc1_tr, losses_tr 
    global losses_cl_tr, losses_x_tr, losses_u_tr
    global acc1_val, losses_val, losses_et_val
    global weights_cl, weights

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # data
    # train-val split
    if args.model == 'baseline':
        num_classes = 3
        train_labeled_set = CxrDataset(STANFORD_CXR_BASE, "data/Stanford_train_small.csv")
    else: 
        num_classes = train_val_split('data/Stanford_train_small.csv', n_labeled = args.n_labeled, split=args.split)
        train_labeled_set = CXR_unlabeled(STANFORD_CXR_BASE, "data/train_labeled_{}.csv".format(args.l_num))
        train_unlabeled_set = CXR_unlabeled(STANFORD_CXR_BASE, "data/train_unlabeled_{}.csv".format(args.u_num))
        
    val_set = CxrDataset(STANFORD_CXR_BASE, "data/Stanford_valid.csv")

    batch_size_label = args.batch_size//2
    batch_size_unlabel = args.batch_size//2
    if (args.model == 'baseline'): batch_size_label=args.batch_size

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=batch_size_label, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size_unlabel, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model
    print("==> creating model")

    # create model
    def create_model(num_classes, ema=False):
        model = DenseNet121(num_classes)
        model = torch.nn.DataParallel(model).cuda()
        #model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(num_classes=num_classes, ema=False)
    print("num classes:", num_classes)
    if args.model == 'mixmatch':
        ema_model = create_model(num_classes=num_classes, ema=True)
    if args.model == 'mt':
        import copy  
        model_teacher = copy.deepcopy(model)
        model_teacher = torch.nn.DataParallel(model_teacher).cuda()
        #model_teacher = model.cuda()
    
    ckpt_dir = args.out
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir)

    cudnn.benchmark = True
    print('Ttoal params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # deifine loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    criterion_mse = nn.MSELoss(size_average=False).cuda()
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()    
    criterion_l1 = nn.L1Loss(size_average=False).cuda()
   
    criterions = (criterion, criterion_mse, criterion_kl, criterion_l1)

    if args.optim == 'adam':
        print('Using Adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    betas=(0.9,0.999),
                                    weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        print('Using SGD optimizer')
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    if args.model == 'mixmatch':
        ema_optimizer = WeightEMA(model, ema_model, args, alpha=args.ema_decay)

    # resume
    title = 'ssl-NIH'
    if args.resume:
        # load checkpoints
        print('==> Resuming from checkpoints..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_prec1 = checkpoint['best_prec1']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.model=='mt': model_teacher.load_state_dict(checkpoint['state_dict'])
        if args.model=='mixmatch': ema_model.load_state_dict(checkpoint['ema_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss','Valid Loss','Train Acc.1', 'Valid Acc.1'])

    writer = SummaryWriter(args.out)
    step = 0

    # train and val
    for epoch in range(args.start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch+1, args.epochs, state['lr']))

        if args.optim == 'adam':
            print('Learning rate schedule for Adam')
            lr = adjust_learning_rate_adam(optimizer, epoch)
        elif args.optim == 'sgd':
            print('Learning rate schedule for SGD')
            lr = adjust_learning_rate(optimizer, epoch)

        if args.model == 'baseline':
            print('Supervised Training')
            #for i in range(5): #baseline repeat 5 times since small number of training set 
            prec1_tr, loss_tr = train_sup(labeled_trainloader, model, criterions, optimizer, epoch, args)
            weight_cl = 0.0
        elif args.model == 'pi':
            print('Pi model')
            prec1_tr, loss_tr, loss_cl_tr, weight_cl = train_pi(labeled_trainloader, unlabeled_trainloader, num_classes, model, criterions, optimizer, epoch, args)
        elif args.model == 'mt':
            print('Mean Teacher model')
            prec1_tr, loss_tr, loss_cl_tr, prec1_t_tr, weight_cl = train_mt(labeled_trainloader, unlabeled_trainloader, num_classes, model, model_teacher, criterions, optimizer, epoch, args)
        elif args.model == 'mixmatch':
            print('MixMatch model')
            prec1_tr, loss_tr, loss_x_tr, loss_u_tr, weight = train_mixmatch(labeled_trainloader, unlabeled_trainloader, num_classes, model, optimizer, ema_optimizer, epoch, args)

        # evaluate on validation set 
        if args.model == 'mixmatch':
            prec1_val, loss_val = validate(val_loader, model, criterions, args, 'valid')
            prec1_ema_val, loss_ema_val = validate(val_loader, ema_model, criterions, args, 'valid')
        else: 
            prec1_val, loss_val = validate(val_loader, model, criterions, args, 'valid')        
        if args.model=='mt':
            prec1_t_val, loss_t_val = validate(val_loader, model_teacher, criterions, args, 'valid')

        # append values
        acc1_tr.append(prec1_tr)
        losses_tr.append(loss_tr)
        acc1_val.append(prec1_val)
        losses_val.append(loss_val)
        if (args.model != 'baseline') and (args.model != 'mixmatch'): 
            losses_cl_tr.append(loss_cl_tr)
        if args.model == 'mixmatch':
            losses_x_tr.append(loss_x_tr)
            losses_u_tr.append(loss_u_tr)
            weights.append(weight)
        if args.model=='mt':
            acc1_t_tr.append(prec1_t_tr)
            acc1_t_val.append(prec1_t_val)
            weights_cl.append(weight_cl)
        learning_rate.append(lr)

        # remember best prec@1 and save checkpoint
        if args.model == 'mt': 
            is_best = prec1_t_val > best_prec1
            if is_best:
                best_val_prec1_t = prec1_t_val
                best_val_prec1 = prec1_val
            print("Best val precision: %.3f"%best_val_prec1_t)
            best_prec1 = max(prec1_t_val, best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_val_prec1' : best_val_prec1,
                'acc1_tr': acc1_tr,
                'losses_tr': losses_tr,
                'losses_cl_tr': losses_cl_tr,
                'acc1_val': acc1_val,
                'losses_val': losses_val,
                'acc1_t_tr': acc1_t_tr,
                'acc1_t_val': acc1_t_val,
                'state_dict_teacher': model_teacher.state_dict(),
                'weights_cl' : weights_cl,
                'learning_rate' : learning_rate,
            }
        elif args.model == 'mixmatch':
            is_best = prec1_val > best_prec1
            if is_best:
                best_val_prec1 = prec1_val
            print("Best val precision: %.3f"%best_val_prec1)
            best_prec1 = max(prec1_val, best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'best_val_prec1' : best_val_prec1,
                'acc1_tr': acc1_tr,
                'losses_tr': losses_tr,
                'acc1_val': acc1_val,
                'losses_val': losses_val,
                'learning_rate' : learning_rate,
            }
        else:
            is_best = prec1_val > best_prec1
            if is_best:
                best_val_prec1 = prec1_val
            print("Best val precision: %.3f"%best_val_prec1)
            best_prec1 = max(prec1_val, best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'acc1_tr': acc1_tr,
                'losses_tr': losses_tr,
                'losses_cl_tr': losses_cl_tr,
                'acc1_val': acc1_val,
                'losses_val': losses_val,
                'weights_cl' : weights_cl,
                'learning_rate' : learning_rate,
            }

        save_checkpoint(dict_checkpoint, is_best, args.arch.lower()+str(args.n_labeled), dirname=ckpt_dir)

        #step = args.val_iteration * (epoch+1)

        writer.add_scalar('loss/train_loss', loss_tr, (epoch+1)) 
        writer.add_scalar('loss/valid_loss', loss_val, (epoch+1))

        writer.add_scalar('accuracy/train_acc', prec1_tr, (epoch+1))
        writer.add_scalar('accuracy/val_acc', prec1_val, (epoch+1))
        if args.model=='mt':
            writer.add_scalar('accuracy/val_t_acc', prec1_t_val, (epoch+1))
        if args.model=='mixmatch':
            writer.add_scalar('accuracy/val_t_acc', prec1_ema_val, (epoch+1))

        # append logger file
        logger.append([loss_tr, loss_val, prec1_tr, prec1_val])

    logger.close()
    writer.close()

    print('Best acc:')
    print(best_prec1)

    print('Mean acc:')
    print(np.mean(acc1_val[-20:]))


def train_sup(label_loader, model, criterions, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    criterion, _, _, criterion_l1 = criterions

    end = time.time()

    label_iter = iter(label_loader)     
    for i in range(len(label_iter)):
        (input,input1), target = next(label_iter)
        # measure data loading time
        data_time.update(time.time() - end)
        sl = input.shape
        batch_size = sl[0]
        if use_cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)
        # compute output
        output = model(input)
        
        loss_ce = criterion(output, target.squeeze(1)) / float(batch_size)
        
        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1

        # measure accuracy and record loss
        prec1, _ = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss_ce.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(label_iter), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
    
    return top1.avg , losses.avg

def train_pi(label_loader, unlabel_loader, num_classes, model, criterions, optimizer, epoch, args, weight_pi=20.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_pi = AverageMeter()
    top1 = AverageMeter()
    weights_cl = AverageMeter()

    # switch to train mode
    model.train()

    criterion, criterion_mse, _, criterion_l1 = criterions
    
    end = time.time()

    label_iter = iter(label_loader)     
    unlabel_iter = iter(unlabel_loader)     
    len_iter = len(unlabel_iter)
    for i in range(len_iter):
        # set weights for the consistency loss
        weight_cl = cal_consistency_weight(epoch*len_iter+i, end_ep=(args.epochs//2)*len_iter, end_w=1.0)
        
        try:
            (input,input1), target = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)     
            (input,input1), target  = next(label_iter)

        (input_ul, input1_ul), _ = next(unlabel_iter)
        sl = input.shape
        su = input_ul.shape
        batch_size = sl[0] + su[0]
        # measure data loading time
        data_time.update(time.time() - end)

        input, input1, target = input.cuda(), input1.cuda(), target.cuda(non_blocking=True)
        input_ul, input1_ul = input_ul.cuda(), input1_ul.cuda()
        input_concat_var = torch.cat([input, input_ul])
        input1_concat_var = torch.cat([input1, input1_ul])
       
        # compute output
        output = model(input_concat_var)
        with torch.no_grad():
            output1 = model(input1_concat_var)

        output_label = output[:sl[0]]
        #pred = F.softmax(output, 1) # consistency loss on logit is better 
        #pred1 = F.softmax(output1, 1)
        loss_ce = criterion(output_label, target.squeeze(1)) / float(sl[0])
        loss_pi = criterion_mse(output, output1) / float(num_classes * batch_size)

        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1 + weight_cl * weight_pi * loss_pi

        # measure accuracy and record loss
        prec1, _ = accuracy(output_label.data, target, topk=(1, 3))
        losses.update(loss_ce.item(), input.size(0))
        losses_pi.update(loss_pi.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        weights_cl.update(weight_cl, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LossPi {loss_pi.val:.4f} ({loss_pi.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len_iter, batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_pi=losses_pi,
                   top1=top1))
    
    return top1.avg , losses.avg, losses_pi.avg, weights_cl.avg

def train_mt(label_loader, unlabel_loader, num_classes, model, model_teacher, criterions, optimizer, epoch, args, ema_const=0.95, weight_mt=8.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cl = AverageMeter()
    top1 = AverageMeter()
    top1_t = AverageMeter()
    weights_cl = AverageMeter()

    # switch to train mode
    model.train()
    model_teacher.train()

    criterion, criterion_mse, _, criterion_l1 = criterions
    
    end = time.time()

    label_iter = iter(label_loader)     
    unlabel_iter = iter(unlabel_loader)     
    len_iter = len(unlabel_iter)
    for i in range(len_iter):
        # set weights for the consistency loss
        global_step = epoch * len_iter + i
        weight_cl = cal_consistency_weight(global_step, end_ep=(args.epochs//2)*len_iter, end_w=1.0)
        
        try:
            (input,input1), target = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)     
            (input,input1), target  = next(label_iter)
        (input_ul, input1_ul), _ = next(unlabel_iter)
        sl = input.shape
        su = input_ul.shape
        batch_size = sl[0] + su[0]
        # measure data loading time
        data_time.update(time.time() - end)

        input, input1, target = input.cuda(), input1.cuda(), target.cuda(non_blocking=True)
        input_ul, input1_ul = input_ul.cuda(), input1_ul.cuda()
        input_concat_var = torch.cat([input, input_ul])
        input1_concat_var = torch.cat([input1, input1_ul])

        # compute output
        output = model(input_concat_var)
        with torch.no_grad():
            output1 = model_teacher(input1_concat_var)

        output_label = output[:sl[0]]
        output1_label = output1[:sl[0]]
        #pred = F.softmax(output, 1)
        #pred1 = F.softmax(output1, 1)
        loss_ce = criterion(output_label, target.squeeze(1)) /float(sl[0])
        loss_cl = criterion_mse(output, output1) /float(num_classes * batch_size)

        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1 + weight_cl * weight_mt * loss_cl

        # measure accuracy and record loss
        prec1, _ = accuracy(output_label.data, target, topk=(1, 3))
        prec1_t, _ = accuracy(output1_label.data, target, topk=(1, 3))
        losses.update(loss_ce.item(), input.size(0))
        losses_cl.update(loss_cl.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top1_t.update(prec1_t.item(), input.size(0))
        weights_cl.update(weight_cl, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, model_teacher, ema_const, global_step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LossCL {loss_cl.val:.4f} ({loss_cl.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'PrecT@1 {top1_t.val:.3f} ({top1_t.avg:.3f})'.format(
                   epoch, i, len_iter, batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_cl=losses_cl,
                   top1=top1, top1_t=top1_t))
    
    return top1.avg , losses.avg, losses_cl.avg, top1_t.avg, weights_cl.avg


def train_mixmatch(label_loader, unlabel_loader, num_classes, model, optimizer, ema_optimizer, epoch, args):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    top1 = AverageMeter()
    weights = AverageMeter()
    nu = 2
    end = time.time()

    label_iter = iter(label_loader)     
    unlabel_iter = iter(unlabel_loader)

    model.train()
    for i in range(args.val_iteration):
        
        try:
            (input, _), target = next(label_iter)
        except:
            label_iter = iter(label_loader)     
            (input, _), target  = next(label_iter)
        try:
            (input_ul, input1_ul), _ = next(unlabel_iter)
        except:
            unlabel_iter = iter(unlabel_loader)
            (input_ul, input1_ul), _ = next(unlabel_iter)

        bs = input.size(0)
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda(non_blocking=True)
        input_ul, input1_ul = input_ul.cuda(), input1_ul.cuda()
        
        with torch.no_grad():
            # compute guess label
            logits = model(torch.cat([input_ul, input1_ul], dim=0))
            p = torch.nn.functional.softmax(
                logits, dim=-1).view(nu, -1, logits.shape[1])
            p_target = p.mean(dim=0).pow(1./args.T)
            p_target /= p_target.sum(dim=1, keepdim=True)
            guess = p_target.detach_()

            assert input.shape[0] == input_ul.shape[0]

            # mixup
            target_in_onehot = torch.zeros(bs, num_classes).float().cuda().scatter_(1, target.view(-1, 1), 1)
            mixed_input, mixed_target = mixup(torch.cat([input] + [input_ul, input1_ul], dim=0),
                                        torch.cat([target_in_onehot] + [guess] * nu, dim=0),
                                        beta = args.beta)
            # reshape to (nu+1, bs, w, h, c)
            mixed_input = mixed_input.reshape([nu + 1] + list(input.shape))
            # reshape to (nu+1, bs)
            mixed_target = mixed_target.reshape([nu + 1] + list(target_in_onehot.shape))
            input_x, input_u = mixed_input[0], mixed_input[1:]
            target_x, target_u = mixed_target[0], mixed_target[1:]

        model.train()
        batches = interleave([input_x, input_u[0], input_u[1]], bs)
        logits = [model(batches[0])]
        for batchi in batches[1:]:
            logits.append(model(batchi))
        logits = interleave(logits, bs)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], 0)

        # loss
        # cross entropy loss for soft label
        loss_xe = torch.mean(torch.sum(-target_x * F.log_softmax(logits_x, dim=-1), dim=1))
        # L2 loss
        loss_l2u = F.mse_loss(F.softmax(logits_u, dim=-1), target_u.reshape(nu * bs, num_classes))
        # weight for unlabeled loss with warmup
        w_match = args.lambda_u * linear_rampup(epoch + i/args.val_iteration, args.epochs)
        loss = loss_xe + w_match * loss_l2u

        # measure accuracy and record loss
        prec1, _ = accuracy(logits_x, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        losses_x.update(loss_xe.item(), input.size(0))
        losses_u.update(loss_l2u.item(), input.size(0))

        top1.update(prec1.item(), input.size(0))
        weights.update(w_match, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                  'Loss_u {loss_u.val:.4f} ({loss_u.avg:.4f})\t'
                  'Ws {ws.val:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, args.val_iteration, batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_x=losses_x, loss_u=losses_u,
                   ws=weights, top1=top1))
    
    ema_optimizer.step(bn=True)
    return top1.avg, losses.avg, losses_x.avg, losses_u.avg, weights.avg


def validate(val_loader, model, criterions, args, mode = 'valid'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    CxrDataset.eval()

    criterion, _, _, _ = criterions

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            sl = input.shape
            batch_size = sl[0]
            input, target = input.cuda(), target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            softmax = torch.nn.LogSoftmax(dim=1)(output)
            loss = criterion(output, target.squeeze(1)) / float(batch_size)
 
            # measure accuracy and record loss
            prec1, _ = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
 
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
 
            if i % args.print_freq == 0:
                if mode == 'test':
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1))
                else:
                    print('{0}: [{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                           mode, i, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1))

    print(' ****** Prec@1 {top1.avg:.3f} Loss {loss.avg:.3f} '
          .format(top1=top1, loss=losses))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dirname='.'):
    fpath = os.path.join(dirname, filename + '_latest.pth.tar')
    torch.save(state, fpath)
    if is_best:
        bpath = os.path.join(dirname, filename + '_best.pth.tar')
        shutil.copyfile(fpath, bpath)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [150, 225, 300] epochs"""
    
    boundary = [args.epochs//2,args.epochs//4*3,args.epochs]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    print('Learning rate: %f'%lr)
    #print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adjust_learning_rate_adam(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""
    
    boundary = [args.epochs//5*4]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    print('Learning rate: %f'%lr)
    #print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep)/float(end_ep - init_ep)
        #weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
    #print('Consistency weight: %f'%weight_cl)
    return weight_cl

def cal_reg_l1(model, criterion_l1):
    reg_loss = 0
    np = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
        np += param.nelement()
    reg_loss = reg_loss / np
    return reg_loss
 
def update_ema_variables(model, model_teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param.data)

class WeightEMA(object):
    def __init__(self, model, ema_model, args, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = DenseNet121(num_classes=3).cuda()  # TODO
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)


if __name__ == '__main__':
    main()