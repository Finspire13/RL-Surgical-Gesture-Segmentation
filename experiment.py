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
    
    feature_types = ['sensor']

    # feature_types = ['visual'] if dataset_name == 'GTEA' \
    #                 else ['sensor', 'visual']

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

    feature_types = ['sensor']

    # feature_types = ['visual'] if dataset_name == 'GTEA' \
    #             else ['sensor', 'visual']

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

    # Set seed
    #utils.set_global_seeds(777, True)

    # alpha_dict = {'50Salads_eval': 0.20, 
    #               'GTEA': 0.375, 
    #               '50Salads_mid': 0.35}   # Visual

    alpha_dict = {'50Salads_eval': 0.15, 
                  '50Salads_mid': 0.40}   # Sensor

    for name in ['50Salads_eval', '50Salads_mid']:
    #for name in ['GTEA', '50Salads_mid']:
        update_config_file(['dataset_name'], name)
        # utils.set_up_dirs()
        # utils.clean_up()
        experiment_tcn()

        update_config_file([name, 'rl_params', 'reward_alpha'], alpha_dict[name])
        experiment_trpo('Test')

        # # for alpha in [0.1, 0.125, 0.15, 0.175, 0.20, 
        # #               0.225, 0.25, 0.275, 0.30, 0.35]:
        # for alpha in [0.375, 0.40, 0.45, 0.50]:
                      
            
            



if __name__ == '__main__':
    main()



# k_steps_dict = {'JIGSAWS':[4, 21], '50Salads_eval':[1, 20],      #19.77
#                 '50Salads_mid':[1, 5], 'GTEA':[1, 15]}           #4.61 14.99  


# reward_alpha_dict = {'JIGSAWS': 0.1, '50Salads_eval': 0.14,  # According to visual
#                      '50Salads_mid': 0.25, 'GTEA': 0.27}  