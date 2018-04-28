from __future__ import division
from __future__ import print_function

import utils
import time
import os
import numpy as np
from tcn_train_test import cross_validate
from get_tcn_feature import get_feature_by_split
from config import tcn_params, tcn_run_num, result_dir, dataset_name

import pdb

def main():

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--feature_type', type=str, default='sensor')
    args = parser.parse_args()
    if args.feature_type not in ['sensor', 'visual']:
        raise Exception('Invalid Feature Type')


    feature_type = args.feature_type

    if feature_type == 'visual':

        if dataset_name in ['JIGSAWS_K', 'JIGSAWS_N']:
            raise Exception('No Visual Data for this dataset!')
        elif dataset_name in ['JIGSAWS', 'GTEA', '50Salads_eval', 
                              '50Salads_mid']:
            feature_size = 128
        else:
            raise Exception('Invalid Dataset Name!') 

    elif feature_type == 'sensor':

        if dataset_name in ['JIGSAWS', 'JIGSAWS_K', 'JIGSAWS_N']:
            feature_size = 14
        elif dataset_name in ['50Salads_eval', '50Salads_mid']:
            feature_size = 30
        elif dataset_name == 'GTEA':
            raise Exception('No Sensor Data for this dataset!')
        else:
            raise Exception('Invalid Dataset Name!') 

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

