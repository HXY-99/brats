# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py
import warnings
import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
from dataset import BraTS
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from model.swintransformer import transformer_model
import torch.distributed as dist
from model.criterions import BCEDiceLoss

from torch.utils.data import DataLoader
from utils import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn


local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='', type=str)

parser.add_argument('--experiment', default='BraTS', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='BraTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='path to training set', type=str)

parser.add_argument('--train_dir', default='path for train data', type=str)

parser.add_argument('--valid_dir', default='path for valid data', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--dataset', default='BraTS', type=str)

parser.add_argument('--model_name', default='SwinTransformer', type=str)

# Training Information
parser.add_argument('--lr', default=0.0001, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='sigmoid_dice', type=str)

parser.add_argument('--num_class', default=1, type=int)

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=100, type=int)

parser.add_argument('--save_freq', default=10, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=False, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    model = transformer_model()

    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    criterion = BCEDiceLoss().cuda()
    # criterion = getattr(criterions, args.criterion)


    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    writer = SummaryWriter()

    if args.load:
        pretrained_dict = torch.load(r'upernet_swin_base_patch4_window7_512x512.pth')
        for k, v in pretrained_dict.items():
            model_dict = v

        model.load_state_dict(model_dict, strict=False)

        logging.info('Successfully loading checkpoint and training!')
    else:
        logging.info('re-training!!!')

    train_root = args.train_dir

    train_set = BraTS(train_root)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    num_gpu = (len(args.gpu)+1) // 2


    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    logging.info('Samples for train = {}'.format(len(train_loader)))
    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()

        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            output = model(x)

            loss = criterion(output, target)
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()


            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter:{}  loss: {:.5f}  ||'
                             .format(epoch, i, reduce_loss))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss:', reduce_loss, epoch)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        writer.close()

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    warnings.filterwarnings("ignore")
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
