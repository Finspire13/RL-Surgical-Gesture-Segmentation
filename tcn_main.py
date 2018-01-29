from __future__ import division
from __future__ import print_function

import utils
import time
import os
import numpy as np
from tcn_train_test import cross_validate
from get_tcn_feature import get_feature_by_split
from config import result_dir, model_params, train_params, tcn_run_num

import pdb

def main():

    utils.set_up_dirs()
    utils.clean_up()

    # Set seed
    #utils.set_global_seeds(777, True)

    full_result = []

    for feature_type in ['sensor', 'visual']:

        if feature_type == 'visual':
            feature_size = 128
        elif feature_type == 'sensor':
            feature_size = 14

        model_params['encoder_params']['input_size'] = feature_size

        modality_result = []

        for run_idx in range(tcn_run_num):

            naming = '{}_run_{}'.format(feature_type, run_idx)

            run_result = cross_validate(model_params, train_params,
                                        feature_type, naming)
            get_feature_by_split(model_params, feature_type, naming)

            # print('Acc: ', result[0].mean())
            # print('Edit: ', result[1].mean())
            # print('F10: ', result[2].mean())
            # print('F25: ', result[3].mean())
            # print('F50: ', result[4].mean())
            # print('F75: ', result[5].mean())

            modality_result.append(run_result)

        modality_result = np.array(modality_result).mean(0)
        full_result.append(modality_result)

    result_file = os.path.join(result_dir, 'tcn_result.npy')
    np.save(result_file, np.array(full_result))

if __name__ == '__main__':
    main()

