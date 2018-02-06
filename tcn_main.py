from __future__ import division
from __future__ import print_function

import utils
import time
import os
import numpy as np
from tcn_train_test import cross_validate
from get_tcn_feature import get_feature_by_split
from config import tcn_params, tcn_run_num, result_dir

import pdb

def main():

    for feature_idx in range(2):

        feature_type = ['sensor', 'visual'][feature_idx]
        if feature_type == 'visual':
            feature_size = 128
        elif feature_type == 'sensor':
            feature_size = 14

        tcn_params['model_params']['encoder_params']['input_size'] = feature_size

        for run_idx in range(tcn_run_num):

            naming = '{}_run_{}'.format(feature_type, run_idx + 1)

            run_result = cross_validate(tcn_params['model_params'], 
                                        tcn_params['train_params'],
                                        feature_type, 
                                        naming)  #8x6

            get_feature_by_split(tcn_params['model_params'], 
                                 feature_type, naming)

            result_file = os.path.join(result_dir, 
                            'tcn_result_{}.npy'.format(naming))

            np.save(result_file, run_result)

            # print('Acc: ', result[0].mean())
            # print('Edit: ', result[1].mean())
            # print('F10: ', result[2].mean())
            # print('F25: ', result[3].mean())
            # print('F50: ', result[4].mean())
            # print('F75: ', result[5].mean())

if __name__ == '__main__':
    main()

