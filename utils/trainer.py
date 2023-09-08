import torch
import torch.nn as nn
import os
from pathlib import Path
import functools

from utils.utils import Utils
from utils.time_estimator import TimeEstimator
from model.model import LSTN
from model.loss import Loss


class Trainer(Utils):
    def __init__(self, args):
        super().__init__()
        print('Initializing trainer ... ')
        self.args = args
        match self.args.device:
            case 'mps':
                self.device = torch.device('mps')
            case 'cpu':
                self.device = torch.device('cpu')
            case _:
                self.device = torch.device('cuda')
                os.environ['CUDA_VISIBLE_DEVICES'] = self.args.device
                torch.cuda.manual_seed(self.args.seed)
                torch.multiprocessing.set_sharing_strategy('file_system')
        self.best_pre = 1e9
        self.time_estimator = TimeEstimator()
        self.args.video_dataset_mode = None if self.args.video_dataset_mode == 'None' else self.args.video_dataset_mode
        self.args.pretrained_path = None if self.args.pretrained_path == 'None' else Path(self.args.pretrained_path)
        self.pretrain = False if self.args.pretrained_path else True
        self.args.load = None if self.args.load == 'None' else Path(self.args.load)
        self.args.save_path = Path(self.args.save_path) / self.args.task
        self.args.save_path.mkdir(exist_ok=True)

        if self.args.video_dataset_mode:
            self.train_list, self.val_list = Utils.get_fdst_list(Path(self.args.dataset_path),
                                                                 self.args.video_dataset_mode, True)
        else:
            train_path = Path(self.args.dataset_path) / 'train' / 'images'
            train_list = [str(path) for path in train_path.glob('*.jpg')]
            train_list.sort(key=functools.cmp_to_key(self.cmp))
            self.train_list, self.val_list = Utils.divide_train_list(train_list, self.args.cluster_num)

        self.mask = self.get_roi_mask(self.args.mask_path)

        self.model = LSTN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.criterion = None

        if (not self.pretrain) and self.args.pretrained_path.is_file():
            print(f'Loading pretrain model from {self.args.pretrained_path}')
            checkpoint = torch.load(self.args.pretrained_path)
            self.model.load_state_dict(checkpoint['model'])
            self.model.freeze_front_end()
            self.criterion = Loss(self.args.loss_w, self.args.loss_h, self.args.lamda, self.args.beta).to(self.device)
            print('done.')
        else:
            self.model.unfreeze_front_end()
            self.criterion = nn.MSELoss().to(self.device)

        if self.args.load and self.args.load.is_file():
            print(f'loading checkpoint from {self.args.load} ...')
            checkpoint = torch.load(self.args.load)
            assert self.pretrain == checkpoint['pretrain']
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pre = checkpoint['best_pre']
            print(f'done.')

        self.model = nn.DataParallel(self.model)
        print('Trainer initializing done.')

    def execute(self):
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            print(f'epoch: {epoch} train begin ... ')
            self.show_hyper_parameter(train_mode=True)

            self.time_estimator.mark()
            self.time_estimator.estimate(epoch, self.args.num_epoch)

            self.train(self.train_list, epoch)
            print('validation begins ...')
            precision = self.test(self.val_list)
            is_best = precision < self.best_pre
            self.best_pre = min(precision, self.best_pre)
            print(f' * best MAE: {self.best_pre:.5f}')

            map_list = [(list(range(0, 51)), '_0~50'), (list(range(51, 101)), '_51~100'),
                        (list(range(101, 151)), '_101~150'), (list(range(151, 201)), '_151~200'),
                        (list(range(201, 251)), '_201~250'), (list(range(251, 301)), '_251~300'),
                        (list(range(301, 351)), '_301~350'), (list(range(351, 401)), '_351~400'),
                        (list(range(401, 451)), '_401~450'), (list(range(451, 501)), '_451~500'),
                        (list(range(501, 551)), '_501~550'), (list(range(551, 601)), '_551~600'),
                        (list(range(601, 651)), '_601~650'), (list(range(651, 701)), '_651~700'),
                        (list(range(701, 751)), '_701~750'), (list(range(751, 801)), '_751~800')]
            postfix = ''
            for tup in map_list:
                if epoch in tup[0]:
                    postfix = tup[1]
                    break
            Utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'pretrain': self.pretrain,
                    'best_pre': self.best_pre,
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'dataset': self.args.dataset_path,
                    'w': self.args.loss_w,
                    'h': self.args.loss_h,
                    'lamda': self.args.lamda
                },
                is_best,
                self.args.task + postfix,
                str(self.args.save_path)
            )
