import os
import numpy as np 
from config import *

import pdb

# This is an awkward solution since I dont know how to run multiple tf sessions..

def main():
    tcn_cmd = 'python3 tcn_main.py'
    trpo_train_cmd = 'python3 trpo_train.py '
    trpo_test_cmd = 'python3 trpo_test.py '
    cmd_args = '--feature_type {} --tcn_run_idx {} --split_idx {} --run_idx {}'

    os.system(tcn_cmd)

    for feature_type in ['sensor', 'visual']:
        for tcn_run_idx in range(1, 1 + tcn_run_num):
            for split_idx in range(1, 1 + split_num):
                for run_idx in range(1, 1 + trpo_train_run_num):

                    formatted_args = cmd_args.format(feature_type,
                                    tcn_run_idx, split_idx, run_idx)

                    os.system(trpo_train_cmd + formatted_args)
                    os.system(trpo_test_cmd + formatted_args)

    # Get Averaged Results: TCN
    for feature_type in ['sensor', 'visual']:

        tcn_result = np.zeros((tcn_run_num, split_num, 6))

        for tcn_run_idx in range(1, 1 + tcn_run_num):
            run_result_file = 'tcn_result_{}_run_{}.npy'.format(feature_type, 
                                                                tcn_run_idx)
            run_result_file = os.path.join(result_dir, run_result_file)
            tcn_result[tcn_run_idx-1,:,:] = np.load(run_result_file)

        tcn_result_file = 'tcn_avg_result_{}.npy'.format(feature_type)
        tcn_result_file = os.path.join(result_dir, tcn_result_file)
        #np.save(tcn_result_file, tcn_result.mean(0).mean(0))
        np.save(tcn_result_file, tcn_result)


    # Get Averaged Results: TRPO
    for feature_type in ['sensor', 'visual']:

        trpo_result = np.zeros((tcn_run_num, split_num,
                            trpo_train_run_num, trpo_test_run_num, 9))

        for tcn_run_idx in range(1, 1 + tcn_run_num):
            for split_idx in range(1, 1 + split_num):
                for run_idx in range(1, 1 + trpo_train_run_num):
                    run_result_file = 'result_{}_tcn_{}_split_{}_run_{}.npy'.format(
                                    feature_type, tcn_run_idx, split_idx, run_idx)
                    run_result_file = os.path.join(result_dir, run_result_file)

                    run_result = np.load(run_result_file)
                    run_result = run_result.mean(0)

                    trpo_result[tcn_run_idx-1,split_idx-1,run_idx-1,:,:] = run_result

        trpo_result_file = 'trpo_avg_result_{}.npy'.format(feature_type)
        trpo_result_file = os.path.join(result_dir, trpo_result_file)
        #np.save(trpo_result_file, trpo_result.mean(0).mean(0).mean(0).mean(0))
        np.save(trpo_result_file, trpo_result)


if __name__ == '__main__':
    main()