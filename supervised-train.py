import argparse
import os
import shutil
import time
import random

import numpy as np 
from tqdm import tqdm

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
from dataset import STANFORD_CXR_BASE, CxrDataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Supervised Training')
# Optimization options
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints 
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')                    
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Validation iteration')                        

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    agrs.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0

def main():

    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # data
    num_classes = 3
    train_set = CxrDataset(STANFORD_CXR_BASE, "~/cxr-jingyi/ViewPosition/Stanford_train_small.csv")
    val_set = CxrDataset(STANFORD_CXR_BASE, "~/cxr-jingyi/ViewPosition/Stanford_valid.csv")

    trainloader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # model
    print("==> creating model")

    def create_model(num_classes, ema=False):
        model = DenseNet121(num_classes)
        model = torch.nn.DataParallel(model).cuda()
        
        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(num_classes=num_classes, ema=False)
    cudnn.benchmark = True
    print('Ttoal params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    # resume
    title = 'supervised-Stanford'
    if args.resume:
        # load checkpoints
        print('==> Resuming from checkpoints..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss','Valid Loss','Valid Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch+1, args.epochs, state['lr']))

        train_iteration, train_loss = train(trainloader, num_classes, model, optimizer, criterion, epoch, use_cuda)
        train_acc_iteration, _, train_acc = validate(trainloader, model, num_classes, criterion, epoch, use_cuda, mode='Train Stats')
        val_iteration, val_loss, val_acc = validate(val_loader, model, num_classes, criterion, epoch, use_cuda, mode='Valid Stats')

        #step = args.val_iteration * (epoch+1)

        writer.add_scalar('loss/train_loss', train_loss, (epoch+1)) 
        writer.add_scalar('loss/valid_loss', val_loss, (epoch+1))

        writer.add_scalar('accuracy/train_acc', train_acc, (epoch+1))
        writer.add_scalar('accuracy/val_acc', val_acc, (epoch+1))
        
        # append logger file
        logger.append([train_loss, val_loss, val_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, is_best)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(val_accs[-20:]))


def train(train_loader, num_classes, model, optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    t = tqdm(enumerate(train_loader), total=len(train_loader), desc='training')

    model.train()
    for batch_idx, (input, target) in t:
        if use_cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time()-end)
        # batch size
        batch_size = input.size(0)
        
        output = model(input)
        loss = criterion(output, target.squeeze(1))

        # record loss
        losses.update(loss.item(), input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg
                    )
        bar.next()
    bar.finish()

    return (batch_idx, losses.avg,)


def validate(valloader, model, num_classes, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = inputs.size(0)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze(1))

            # measure accuracy and record loss
            prec1,_ = accuracy(outputs, targets, topk=(1,3))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg
                        )
            bar.next()
        bar.finish()
    return (batch_idx, losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
