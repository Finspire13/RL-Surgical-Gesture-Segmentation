import os
import shutil
import numpy as np 
import json
from subprocess import Popen
import importlib
import math
import config # To trigue config update

# Config should be improved: so ugly!!!

import utils
import pdb


# This is an awkward solution since I dont know how to run multiple tf sessions..

def experiment_tcn():

    from config import result_dir, split_num, tcn_run_num, dataset_name
    
    # feature_types = ['sensor']

    feature_types = ['visual'] \
        if dataset_name in ['JIGSAWS_K', 'JIGSAWS_N', 'GTEA'] \
        else ['sensor', 'visual']

    ####################################################

    for feature_type in feature_types:

        tcn_cmd = 'python3 tcn_main.py --feature_type {}'.format(feature_type)
        
        Popen(tcn_cmd, shell=True).wait()
        #os.system(tcn_cmd)

        # Get Averaged Results: TCN
        template = 'tcn_result_{}_run_{}.npy'
    
        tcn_result = np.zeros((tcn_run_num, split_num, 6))
        for tcn_run_idx in range(1, 1 + tcn_run_num):
            run_result_file = template.format(feature_type, tcn_run_idx)
            run_result_file = os.path.join(result_dir, run_result_file)
            tcn_result[tcn_run_idx-1,:,:] = np.load(run_result_file)
            os.remove(run_result_file)

        tcn_result_file = 'tcn_avg_result_{}.npy'.format(feature_type)
        tcn_result_file = os.path.join(result_dir, tcn_result_file)
        #np.save(tcn_result_file, tcn_result.mean(0).mean(0))
        np.save(tcn_result_file, tcn_result)


def experiment_trpo(naming):

    from config import (result_dir, trpo_model_dir, graph_dir, 
                        dataset_name, split_num, tcn_run_num, 
                        trpo_test_run_num, trpo_train_run_num)

    # feature_types = ['sensor']

    feature_types = ['visual'] \
        if dataset_name in ['JIGSAWS_K', 'JIGSAWS_N', 'GTEA'] \
        else ['sensor', 'visual']

    ####################################################

    trpo_train_cmd = 'python3 trpo_train.py '
    trpo_test_cmd = 'python3 trpo_test.py '
    cmd_args = '--feature_type {} --tcn_run_idx {} --split_idx {} --run_idx {}'

    # Run TRPO in paralell    
    for tcn_run_idx in range(1, 1 + tcn_run_num):
        for split_idx in range(1, 1 + split_num):
            for run_idx in range(1, 1 + trpo_train_run_num):

                # Train
                processes = []
                for feature_type in feature_types:
                #for feature_type in ['sensor']:
                    formatted_args = cmd_args.format(
                        feature_type, tcn_run_idx, split_idx, run_idx)
                    full_cmd = trpo_train_cmd + formatted_args
                    processes.append(Popen(full_cmd, shell=True))

                # Output on shell is messed up, see baseline logs instead
                exitcodes = [p.wait() for p in processes]
                if sum(exitcodes) != 0:
                    raise Exception('Subprocess Error!')

                # Test
                processes = []
                for feature_type in feature_types:
                #for feature_type in ['sensor']:
                    formatted_args = cmd_args.format(
                        feature_type, tcn_run_idx, split_idx, run_idx)
                    full_cmd = trpo_test_cmd + formatted_args
                    processes.append(Popen(full_cmd, shell=True))

                # Output on shell is messed up, see baseline logs instead
                exitcodes = [p.wait() for p in processes]
                if sum(exitcodes) != 0:
                    raise Exception('Subprocess Error!')


    # Get Averaged Results: TRPO
    template = 'trpo_result_{}_tcn_{}_split_{}_run_{}.npy'
    for feature_type in feature_types:
        trpo_result = np.zeros((tcn_run_num, split_num,
                            trpo_train_run_num, trpo_test_run_num, 9))
        for tcn_run_idx in range(1, 1 + tcn_run_num):
            for split_idx in range(1, 1 + split_num):
                for run_idx in range(1, 1 + trpo_train_run_num):
                    run_result_file = template.format(
                        feature_type, tcn_run_idx, split_idx, run_idx)
                    run_result_file = os.path.join(result_dir, run_result_file)
                    run_result = np.load(run_result_file)
                    run_result = run_result.mean(0)
                    trpo_result[tcn_run_idx-1,split_idx-1,run_idx-1,:,:] = run_result
                    os.remove(run_result_file)

        trpo_result_file = 'trpo_avg_result_{}_{}.npy'.format(feature_type, naming)
        trpo_result_file = os.path.join(result_dir, trpo_result_file)
        #np.save(trpo_result_file, trpo_result.mean(0).mean(0).mean(0).mean(0))
        np.save(trpo_result_file, trpo_result)

    # Move all models into subfolder
    trpo_model_sub_dir = os.path.join(trpo_model_dir, naming)
    if not os.path.exists(trpo_model_sub_dir):
        os.makedirs(trpo_model_sub_dir)

    for item in os.listdir(trpo_model_dir):
        if item.startswith('trpo_model'):
            item = os.path.join(trpo_model_dir, item)
            shutil.move(item, trpo_model_sub_dir)

    # Move all graphs into subfolder
    graph_sub_dir = os.path.join(graph_dir, naming)
    if not os.path.exists(graph_sub_dir):
        os.makedirs(graph_sub_dir)

    for item in os.listdir(graph_dir):
        if item.startswith('barcode'):
            item = os.path.join(graph_dir, item)
            shutil.move(item, graph_sub_dir)


def update_config_file(keys, value):

    all_params = json.load(open('config.json'))

    # Update the dict with a list of keys
    temp = all_params
    for key in keys[:-1]: temp = temp[key]
    temp[keys[-1]] = value

    with open('config.json', 'w') as f:
        json.dump(all_params, f, indent=2)

    importlib.reload(config)


def main():
    
    # Experiment setup
    for name in ['JIGSAWS_K', 'JIGSAWS_N', 'JIGSAWS']:
        update_config_file(['dataset_name'], name)
        utils.set_up_dirs()
        utils.clean_up()
        experiment_tcn()
        experiment_trpo('full')


if __name__ == '__main__':
    main()

