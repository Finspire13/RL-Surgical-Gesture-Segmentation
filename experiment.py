import os
import shutil
import numpy as np 
import json
from subprocess import Popen
from config import (result_dir, trpo_model_dir, graph_dir, split_num,
                    tcn_run_num, trpo_test_run_num, trpo_train_run_num)

import pdb

# This is an awkward solution since I dont know how to run multiple tf sessions..

def experiment_tcn():
    
    tcn_cmd = 'python3 tcn_main.py'
    os.system(tcn_cmd)

    # Get Averaged Results: TCN
    template = 'tcn_result_{}_run_{}.npy'
    for feature_type in ['sensor', 'visual']:
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

    trpo_train_cmd = 'python3 trpo_train.py '
    trpo_test_cmd = 'python3 trpo_test.py '
    cmd_args = '--feature_type {} --tcn_run_idx {} --split_idx {} --run_idx {}'

    # Run TRPO in paralell    
    for tcn_run_idx in range(1, 1 + tcn_run_num):
        for split_idx in range(1, 1 + split_num):
            for run_idx in range(1, 1 + trpo_train_run_num):

                # Train
                processes = []
                for feature_type in ['sensor', 'visual']:
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
                for feature_type in ['sensor', 'visual']:
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
    for feature_type in ['sensor', 'visual']:
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


def main():

    # experiment_tcn()

    from config import all_params

    all_params['rl_params']['env_mode'] = 'full'
    with open('config.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    experiment_trpo('full')


    all_params['rl_params']['env_mode'] = 'no_tcn'
    with open('config.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    experiment_trpo('no_tcn')


    all_params['rl_params']['env_mode'] = 'no_future'
    with open('config.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    experiment_trpo('no_future')


    all_params['rl_params']['env_mode'] = 'no_hint'
    with open('config.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    experiment_trpo('no_hint')


if __name__ == '__main__':
    main()