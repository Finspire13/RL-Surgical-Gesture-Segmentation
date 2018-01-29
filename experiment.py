import os
from config import tcn_run_num, trpo_train_run_num, split_num

import pdb

# This is an awkward solution since I dont know how to run multiple tf sessions..

def main():
    trpo_train_cmd = 'python3 trpo_train.py '
    trpo_test_cmd = 'python3 trpo_test.py '
    cmd_args = '--feature_type {} --tcn_run_idx {} --split_idx {} --run_idx {}'

    for feature_type in ['sensor', 'visual']:
        for tcn_run_idx in range(1, 1 + tcn_run_num):
            for split_idx in range(1, 1 + split_num):
                for run_idx in range(1, 1 + trpo_train_run_num):

                    formatted_args = cmd_args.format(feature_type,
                                    tcn_run_idx, split_idx, run_idx)

                    os.system(trpo_train_cmd + formatted_args)
                    os.system(trpo_test_cmd + formatted_args)

if __name__ == '__main__':
    main()