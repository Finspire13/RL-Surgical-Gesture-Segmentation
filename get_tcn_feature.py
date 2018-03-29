from __future__ import division
from __future__ import print_function

import os
import numpy as np
import gc
import torch 
import torch.nn as nn
from random import randrange
from torch.autograd import Variable
from tcn_model import EncoderDecoderNet
from my_dataset import RawFeatureDataset

from logger import Logger
import utils
import pdb

from config import (tcn_feature_dir, sample_rate, 
                    raw_feature_dir, dataset_name)


def extract_feature(model, dataset):

    packed_data = []

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=1, shuffle=False)
    model.eval()

    for i, data in enumerate(loader):

        feature = data['feature'].float()   # 1x2049x76
        feature = Variable(feature, volatile=True).cuda()

        gesture = data['gesture'].long()       # 1x2029x1
        gesture = gesture.view(-1).numpy()

        # Forward
        out = model.extract_feature(feature)
        out = out.squeeze().cpu().data.numpy()

        packed_data.append([out, gesture])

    return packed_data


def get_feature_by_split(model_params, feature_type, naming):

    # Get trail list
    cross_val_splits = utils.get_cross_val_splits()

    # Cross Validation
    for split_idx, split in enumerate(cross_val_splits):
        feature_dir = os.path.join(raw_feature_dir, split['name'])
        test_trail_list = split['test']
        train_trail_list = split['train']

        split_naming = naming + '_split_{}'.format(split_idx+1)

        trained_model_file = utils.get_tcn_model_file(split_naming)

        # Model
        model = EncoderDecoderNet(**model_params)
        model = model.cuda()
        model.load_state_dict(torch.load(trained_model_file))
        
        print('Extracting TCN Feature...')

        n_layers = len(model_params['encoder_params']['layer_sizes'])

        # Dataset
        train_dataset = RawFeatureDataset(dataset_name,
                                          feature_dir,
                                          train_trail_list,
                                          feature_type=feature_type,
                                          encode_level=n_layers,
                                          sample_rate=sample_rate,
                                          sample_aug=False,
                                          normalization=[None, None])

        test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
        test_dataset = RawFeatureDataset(dataset_name,
                                         feature_dir,
                                         test_trail_list,
                                         feature_type=feature_type,
                                         encode_level=n_layers,
                                         sample_rate=sample_rate,
                                         sample_aug=False,
                                         normalization=test_norm)

        
        train_packed_data = extract_feature(model, train_dataset)
        test_packed_data = extract_feature(model, test_dataset)

        train_data_file = 'tcn_feature_train_{}.npy'.format(split_naming)
        test_data_file = 'tcn_feature_test_{}.npy'.format(split_naming)

        np.save(os.path.join(tcn_feature_dir, train_data_file), 
                                                train_packed_data)
        np.save(os.path.join(tcn_feature_dir, test_data_file), 
                                                test_packed_data)



