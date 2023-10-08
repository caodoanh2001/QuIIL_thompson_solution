# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.DATASET_PATH = '/data3/doanhbc/thompson_data/'

        # bottom up features root path
        self.FEATURE_PATH = '/data3/doanhbc/thompson_data/vinvl_thompson_features'

        self.init_path()


    def init_path(self):

        self.IMG_FEAT_PATH = self.FEATURE_PATH

        self.QUESTION_PATH = {
            'train': self.DATASET_PATH + 'train_thompson_questions_0.json',
            'val': self.DATASET_PATH + 'valid_thompson_questions_0.json',
            'test': self.DATASET_PATH + 'valid_thompson_questions_0.json',
            'vg': self.DATASET_PATH + 'questions.json',
        }

        self.ANSWER_PATH = {
            'train': self.DATASET_PATH + 'train_thompson_annotations_0.json',
            'val': self.DATASET_PATH + 'valid_thompson_annotations_0.json',
            'vg': self.DATASET_PATH + 'annotations.json',
        }

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts_0/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print('Checking dataset ...')

        # for mode in self.IMG_FEAT_PATH:
        if not os.path.exists(self.IMG_FEAT_PATH):
            print(self.IMG_FEAT_PATH + 'NOT EXIST')
            exit(-1)

        print('Finished')
        print('')

