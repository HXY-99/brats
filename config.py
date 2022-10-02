import argparse


def Args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=500)
    parse.add_argument('--arch', type=str, default='fusion')
    parse.add_argument('--name', type=str, default='model')
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--optimizer', type=str, default='Adam')
    parse.add_argument('--momentum', type=float, default=0.9)
    parse.add_argument('--weight_decay', type=float, default=1e-5)
    parse.add_argument('--early-stop', default=15, type=int, metavar='N')
    parse.add_argument('--loss', type=str, default='BCEDiceLoss')
    parse.add_argument('--aug', default=False)
    parse.add_argument('--dataset', default='BraTS')

    args = parse.parse_args()

    return args


class AverageMeter(object):

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