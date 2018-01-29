from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import numpy as np
import scipy.io
import os
import torch

import utils

import pdb

class JIGSAWS_Dataset(Dataset):
    def __init__(self, feature_dir, trail_list, feature_type,
                 encode_level, sample_rate=1, sample_aug=True,
                 normalization=None):
        super(JIGSAWS_Dataset, self).__init__()

        self.trail_list = trail_list

        if feature_type not in ['visual', 'sensor']:
            raise Exception('Invalid Feature Type')

        self.sample_rate = sample_rate
        self.sample_aug = sample_aug
        self.encode_level = encode_level

        self.all_feature = []
        self.all_gesture = []
        self.marks = []

        start_index = 0
        for idx in range(len(self.trail_list)):
            
            trail_name = self.trail_list[idx]
            data_file = os.path.join(feature_dir, trail_name + '.avi.mat')
            trail_data = scipy.io.loadmat(data_file)

            if feature_type == 'visual':
                trail_feature = trail_data['A']
            elif feature_type == 'sensor':
                trail_feature = trail_data['S'].T

            trail_gesture = trail_data['Y']
            trail_len = trail_gesture.shape[0]

            self.all_feature.append(trail_feature)
            self.all_gesture.append(trail_gesture)

            self.marks.append([start_index, start_index + trail_len])
            start_index += trail_len

        self.all_feature = np.concatenate(self.all_feature)
        self.all_gesture = np.concatenate(self.all_gesture)

        # Normalization
        if normalization is not None:

            if normalization[0] is None:
                self.feature_means = self.all_feature.mean(0)
            else:
                self.feature_means = normalization[0]

            if normalization[1] is None:
                self.feature_stds = self.all_feature.std(0)
            else:
                self.feature_stds = normalization[1]

            self.all_feature = self.all_feature - self.feature_means
            self.all_feature = self.all_feature / self.feature_stds
        else:
            self.feature_means = None
            self.feature_stds = None


    def __len__(self):
        if self.sample_aug:
            return len(self.trail_list) * self.sample_rate
        else:
            return len(self.trail_list)

    def __getitem__(self, idx):

        if self.sample_aug:
            trail_idx = idx // self.sample_rate
            sub_idx = idx % self.sample_rate
        else:
            trail_idx = idx
            sub_idx = 0 

        trail_name = self.trail_list[trail_idx]

        start = self.marks[trail_idx][0]
        end = self.marks[trail_idx][1]

        feature = self.all_feature[start:end,:]
        gesture = self.all_gesture[start:end,:]
        feature = feature[sub_idx::self.sample_rate]
        gesture = gesture[sub_idx::self.sample_rate]

        trail_len = gesture.shape[0]

        padded_len = int(np.ceil(trail_len / 
                          (2**self.encode_level)))*2**self.encode_level

        mask = np.zeros([padded_len, 1])
        mask[0:trail_len] = 1

        padded_feature = np.zeros([padded_len, feature.shape[1]])
        padded_feature[0:trail_len] = feature

        padded_gesture = np.zeros([padded_len, 1])-1
        padded_gesture[0:trail_len] = gesture

        return {'feature': padded_feature,
                'gesture': padded_gesture,
                'mask': mask,
                'name': trail_name}

    def get_means(self):
        return self.feature_means

    def get_stds(self):
        return self.feature_stds
