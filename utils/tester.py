import torch
import os
import functools
from pathlib import Path

from utils.utils import Utils
from model.model import LSTN
from utils.time_estimator import TimeEstimator


class Tester(Utils):
    def __init__(self, args):
        super().__init__()
        print('Initializing tester ... ')
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
        self.args.video_dataset_mode = None if self.args.video_dataset_mode == 'None' else self.args.video_dataset_mode
        self.args.pth_path = Path(self.args.pth_path)
        self.time_estimator = TimeEstimator()

        if self.args.video_dataset_mode:
            self.test_list = self.get_fdst_list(Path(self.args.dataset_path),
                                                self.args.video_dataset_mode, False)
        else:
            test_path = Path(self.args.dataset_path) / 'test' / 'images'
            self.test_list = [str(path) for path in test_path.glob('*.jpg')]
            self.test_list.sort(key=functools.cmp_to_key(self.cmp))
            self.test_list = [self.test_list]

        self.mask = self.get_roi_mask(self.args.mask_path)

        self.model = LSTN().to(self.device)

        if self.args.pth_path.is_file():
            print(f'Loading model from {self.args.pth_path}')
            checkpoint = torch.load(self.args.pth_path)
            self.pretrain = checkpoint['pretrain']
            self.model.load_state_dict(checkpoint['model'])
            print('done.')
        else:
            raise Exception('Fail to load pth file.')

        print('Tester initializing done.')

    def execute(self):
        print('test begins ... ')
        self.show_hyper_parameter(train_mode=False)
        self.test(self.test_list)
