import torch
from torch.utils.data import DataLoader
import numpy as np
import functools
import shutil
import re
from pathlib import Path
from itertools import chain

from dataset.dataset import PresentDataset, PresentNextDataset


class Utils:
    def __init__(self):
        self.args = None
        self.device = None
        self.pretrain = None
        self.mask = None
        self.model = None
        self.time_estimator = None
        self.device = None
        self.criterion = None
        self.optimizer = None

    @staticmethod
    def save_checkpoint(state, is_best, task_id, save_path, filename='checkpoint.pth.tar'):
        torch.save(state, save_path + '/' + task_id + filename)
        if is_best:
            shutil.copyfile(save_path + '/' + task_id + filename, save_path + '/' + task_id + 'model_best.pth.tar')

    @staticmethod
    def cmp(x, y):
        a = int(re.findall(r'(\d+)\.\w+$', str(x))[0])
        b = int(re.findall(r'(\d+)\.\w+$', str(y))[0])
        return -1 if a < b else 1

    @staticmethod
    def divide_train_list(train_list, num):
        val_list = []
        new_train_list = []
        val_ele_len = len(train_list) // (10 * num)
        divide_points = [(len(train_list) * i) // num for i in range(num + 1)]
        for i in range(num):
            val_list.append(train_list[divide_points[i + 1] - val_ele_len: divide_points[i + 1]])
            new_train_list.append(train_list[divide_points[i]: divide_points[i + 1] - val_ele_len])
        return new_train_list, val_list

    @staticmethod
    def get_fdst_list(dataset_path: Path, mode, train=False):
        video_list = [list(range(i * 5 + 1, i * 5 + 6)) for i in range(20)]
        if mode.lower() != 'full':
            mode = int(mode)
            if mode in [17, 18]:
                video_list = video_list[16: 18]
            else:
                video_list = [video_list[mode]]
        if train:
            # train_list = [ls[:2] for ls in video_list]
            # val_list = [ls[2] for ls in video_list]
            # train_list = list(chain(*train_list))
            # train_path = dataset_path / 'train'
            # train_path_list = [list((train_path / str(video_number) / 'images').glob('*.jpg'))
            #                    for video_number in train_list]
            # val_path_list = [list((train_path / str(video_number) / 'images').glob('*.jpg'))
            #                  for video_number in val_list]
            # for ls in train_path_list:
            #     ls.sort(key=functools.cmp_to_key(Utils.cmp))
            # for ls in val_path_list:
            #     ls.sort(key=functools.cmp_to_key(Utils.cmp))

            train_list = [ls[:3] for ls in video_list]
            train_list = list(chain(*train_list))
            train_path = dataset_path / 'train'
            train_path_list = [list((train_path / str(video_number) / 'images').glob('*.jpg'))
                               for video_number in train_list]
            for ls in train_path_list:
                ls.sort(key=functools.cmp_to_key(Utils.cmp))
            video_len = len(train_path_list[0])
            divide_point = video_len * 9 // 10
            val_path_list = [e[divide_point:] for e in train_path_list]
            train_path_list = [e[:divide_point] for e in train_path_list]
            return train_path_list, val_path_list
        else:
            test_list = [ls[3:] for ls in video_list]
            test_list = list(chain(*test_list))
            test_path = dataset_path / 'test'
            test_path_list = [list((test_path / str(video_number) / 'images').glob('*.jpg'))
                              for video_number in test_list]
            for ls in test_path_list:
                ls.sort(key=functools.cmp_to_key(Utils.cmp))
            return test_path_list

    @staticmethod
    def get_roi_mask(mask_path):
        if mask_path != 'None':
            mask = np.load(mask_path)
            mask[mask <= 1e-4] = 0
        else:
            mask = None
        return mask

    def cal_mae_sum(self, res, gt):
        mae_sum = 0
        for i in range(res.shape[0]):
            mae = abs(res[i].data.sum() - gt[i].sum().type(torch.FloatTensor).to(self.device))
            mae_sum += mae
            # print(f'{mae} {res[i].data.sum()} {gt[i].sum().type(torch.FloatTensor).to(self.device)}')
        return mae_sum

    def show_hyper_parameter(self, train_mode=True):
        print(f'Dataset: {self.args.dataset_path}')
        print(f'Device: {self.device}')
        print(f'Pretrain: {self.pretrain}')
        if train_mode:
            print(f'Learning rate: {self.args.learning_rate}')
            print(f'Weight decay: {self.args.weight_decay}')
            print(f'W: {self.args.loss_w}, H: {self.args.loss_h}, lamda: {self.args.lamda}')
            print(f'Batch size: {self.args.batch_size}')

    def unpack_data(self, data, train_mode=True):
        if train_mode:
            img, target, mask = data
            next_target = None
        else:
            img, target, next_target, mask = data
            next_target = next_target.to(self.device)
        img = img.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)
        if train_mode:
            return img, target, mask
        else:
            return img, target, next_target, mask

    def train(self, train_list, epoch):
        data_loader = DataLoader(
            PresentDataset(train_list, self.mask, True),
            batch_size=self.args.batch_size if self.pretrain else self.args.batch_size+1,
            shuffle=True if self.pretrain else False,
            num_workers=self.args.num_workers
        )

        self.model.train()

        loss_sum = 0
        present_mae_sum = 0
        next_mae_sum = 0
        num_img = 0

        self.time_estimator.simple_mark()

        for i, data in enumerate(data_loader):
            img, target, mask = self.unpack_data(data)
            if self.pretrain:
                present_res = self.model(img, False)
                present_res *= mask
                target *= mask
                loss = self.criterion(present_res, target)
            else:
                present_res, next_res = self.model(img, True)
                present_res = present_res * mask
                next_res = next_res * mask
                target = target * mask
                loss = self.criterion(img, present_res, next_res, target)
                next_mae_sum += self.cal_mae_sum(next_res[:-1], target[1:])
                # cri = torch.nn.MSELoss()
                # loss = cri(next_res, next_target)

            loss_sum += loss
            present_mae_sum += self.cal_mae_sum(present_res, target)
            num_img += img.shape[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.args.print_freq == 0:
                loss_sum /= self.args.print_freq
                present_mae_sum /= num_img
                next_mae_sum /= num_img
                print(f'epoch {epoch:<3}: [{i + 1:>5}/{len(data_loader)}]batch loss: {loss_sum:<16.13f} ' +
                      f'present mae: {present_mae_sum:<8.5f} ' +
                      (f'next mae: {next_mae_sum:<8.5f} ' if not self.pretrain else '') +
                      f'time: {self.time_estimator.query_time_span()}s')
                loss_sum = 0
                present_mae_sum = 0
                next_mae_sum = 0
                num_img = 0
                self.time_estimator.simple_mark()

    def test(self, test_list):
        data_loader = DataLoader(
            PresentNextDataset(test_list, self.mask, False),
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        self.model.eval()

        present_mae_sum = 0
        present_rmse_sum = 0
        next_mae_sum = 0
        next_rmse_sum = 0

        self.time_estimator.simple_mark()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                print(f' {((i + 1) / len(data_loader)) * 100:.1f}% ...\r', end='')
                img, target, next_target, mask = self.unpack_data(data, train_mode=False)
                if self.pretrain:
                    present_res = self.model(img, False)
                    present_res *= mask
                    target *= mask
                    mae = self.cal_mae_sum(present_res, target)
                    present_mae_sum += mae
                    present_rmse_sum += mae ** 2
                else:
                    present_res, next_res = self.model(img, True)
                    present_res *= mask
                    next_res *= mask
                    target *= mask
                    next_target *= mask
                    present_mae = self.cal_mae_sum(present_res, target)
                    next_mae = self.cal_mae_sum(next_res, next_target)
                    present_mae_sum += present_mae
                    present_rmse_sum += present_mae ** 2
                    next_mae_sum += next_mae
                    next_rmse_sum += next_mae ** 2

            present_mae_sum /= len(data_loader)
            present_rmse_sum /= len(data_loader)
            present_rmse_sum **= .5
            next_mae_sum /= len(data_loader)
            next_rmse_sum /= len(data_loader)
            next_rmse_sum **= .5

            print(f' Present MAE: {present_mae_sum:.5f}')
            print(f' Present RMSE: {present_rmse_sum:.5f}')
            if not self.pretrain:
                print(f' Next MAE: {next_mae_sum:.5f}')
                print(f' Next RMSE: {next_rmse_sum:.5f}')
            print(f' Time: {self.time_estimator.query_time_span()}s')

        return present_mae_sum
