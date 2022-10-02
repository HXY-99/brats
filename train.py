from config import *
from metrics import *
from model.model import *
from tqdm import tqdm
from dataset import Dataset
from collections import OrderedDict
import pandas as pd
import os
import warnings
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from glob import glob
import torch.optim as optim
from loss import *
from torch.utils.data import DataLoader


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    ious = AverageMeter()
    f1s = AverageMeter()

    model.train()

    for i, (input_seg, input_edge, target_seg, traget_edge) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input_edge = input_edge.cuda()
        input_seg = input_seg.cuda()
        target_seg = target_seg.cuda()
        target_edge = traget_edge.cuda()

        out_seg, out_edge = model(input_seg, input_edge)
        loss_seg = criterion(out_seg, target_seg)
        loss_edge = Cross_entropy_loss(out_edge, target_edge)

        loss = loss_seg + 0.1 * loss_edge
        # loss = loss_seg
        iou = iou_score(out_seg, target_seg)
        f1 = f1_score(out_edge, target_edge)

        losses.update(loss.item(), input_edge.size(0))
        ious.update(iou, input_edge.size(0))
        f1s.update(f1, input_edge.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou_seg', ious.avg),
        ('dice', f1s.avg),
    ])

    return log


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    f1s = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (input_seg, input_edge, target_seg, target_edge) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input_edge = input_edge.cuda()
            input_seg = input_seg.cuda()
            target_seg = target_seg.cuda()
            target_edge = target_edge.cuda()

            out_seg, out_edge = model(input_seg, input_edge)
            loss_seg = criterion(out_seg, target_seg)
            loss_edge = Cross_entropy_loss(out_edge, target_edge)

            loss = loss_seg + 0.1 * loss_edge

            iou = iou_score(out_seg, target_seg)
            f1 = f1_score(out_edge, target_edge)

            losses.update(loss.item(), input_edge.size(0))
            ious.update(iou, input_edge.size(0))
            f1s.update(f1, input_edge.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('f1', f1s.avg),
    ])

    return log


def main():
    args = Args()

    if not os.path.exists('brats20_result/%s' % args.arch):
        os.makedirs('brats20_result/%s' % args.arch)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    if args.loss == 'BCEDiceLoss':
        criterion = BCEDiceLoss().cuda()
    else:
        criterion = nn.BCELoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    seg_img = glob(r'G:\code\BraTS2018\trainSeg\*')
    edge_img = glob(r'G:\code\BraTS2018\trainEdge\*')
    seg_mask = glob(r'G:\code\BraTS2018\trainSegMask\*')
    edge_mask = glob(r'G:\code\BraTS2018\tranEdgeMask\*')

    train_seg, val_seg, train_edge, val_edge, train_mask, val_mask, train_edge_mask, val_edge_mask = \
        train_test_split(seg_img, edge_img, seg_mask, edge_mask, test_size=0.2, random_state=42)

    model = whole_model()

    model = model.cuda()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, nesterov=args.nesterov)

    train_dataset = Dataset(train_seg, train_edge, train_mask, train_edge_mask)
    val_dataset = Dataset(val_seg, val_edge, val_mask, val_edge_mask)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'f1', 'val_loss', 'val_iou', 'val_f1'
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('loss %.4f  - iou %.4f - f1 %.4f - val_loss %.4f - val_iou %.4f - val_f1 %.4f'
              % (train_log['loss'], train_log['iou'], train_log['f1'],  val_log['loss'], val_log['iou'], val_log['f1']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            train_log['f1'],
            val_log['loss'],
            val_log['iou'],
            val_log['f1'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'f1', 'val_loss', 'val_iou', 'val_f1'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('result/%s/log.csv' % args.arch, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'result/%s/model.pth' % (args.arch))
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
