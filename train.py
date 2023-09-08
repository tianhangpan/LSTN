import argparse
import time
import multiprocessing

from utils.trainer import Trainer


parser = argparse.ArgumentParser(description='indoor residual regression pytorch')
parser.add_argument('dataset_path', metavar='DATASET', help='path to dataset')
parser.add_argument('save_path', metavar='SAVE', help='the path to save checkpoint')
parser.add_argument('device', metavar='DEVICE', type=str, help='cuda id, mps or cpu')
parser.add_argument('task', metavar='TASK', type=str, help='task id to use')
parser.add_argument('-vd', '--video_dataset_mode', metavar='VD', type=str, help='FDST dataset', default='None')
parser.add_argument('-m', '--mask_path', metavar='MASK', type=str, help='roi mask path', default='None')
parser.add_argument('-pp', '--pretrained_path', metavar='PP', type=str,
                    help='the path of pretrained model', default='None')
parser.add_argument('-ld', '--load', metavar='LOAD', default='None',
                    type=str, help='path to the checkpoint')
parser.add_argument('-cn', '--cluster_num', metavar='DM', type=int, help='num of clusters of val list', default=1)
parser.add_argument('-lr', '--learning_rate', metavar='LR', type=float, default=1e-5, help='learning rate')
parser.add_argument('-wd', '--weight_decay', metavar='WD', type=float, default=1e-4, help='weight decay')
parser.add_argument('-bs', '--batch_size', metavar='BS', type=int, default=16, help='batch size')
parser.add_argument('-pf', '--print_freq', metavar='PF', type=int, default=50, help='print frequency')
parser.add_argument('-ne', '--num_epoch', metavar='NE', type=int, default=400, help='num of epoch')
parser.add_argument('-nw', '--num_workers', metavar='NE', type=int, help='number of workers', default=0)
parser.add_argument('-lw', '--loss_w', metavar='W', type=int, help='W in loss', default=2)
parser.add_argument('-lh', '--loss_h', metavar='H', type=int, help='H in loss', default=2)
parser.add_argument('-l', '--lamda', metavar='LAMDA', type=float, help='lamda in loss', default=0.001)
parser.add_argument('-b', '--beta', metavar='BETA', type=float, help='beta in loss', default=30.)


if __name__ == '__main__':
    args = parser.parse_args()
    args.start_epoch = 0
    args.seed = time.time()
    args.num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else int(args.num_workers)

    trainer = Trainer(args)
    trainer.execute()
