import os
import shutil
import numpy as np 
import json
from subprocess import Popen
import importlib

import config # To trigue config update

# Config should be improved: so ugly!!!

import utils
import pdb


# This is an awkward solution since I dont know how to run multiple tf sessions..

def experiment_tcn():

    from config import result_dir, split_num, tcn_run_num, dataset_name
    
    feature_types = ['visual']

    # feature_types = ['visual'] if dataset_name == 'GTEA' \
    #                 else ['sensor', 'visual']

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

    feature_types = ['visual']

    # feature_types = ['visual'] if dataset_name == 'GTEA' \
    #             else ['sensor', 'visual']

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


def update_config_file(key, value):

    from config import sample_rate

    all_params = json.load(open('config.json'))
    all_params[key] = value

    if key == 'dataset_name':

        if value not in ['50Salads_eval', '50Salads_mid', 
                         'GTEA', 'JIGSAWS']:
            raise Exception('Invalid Dataset Name!') 

        all_dirs = ['raw_feature_dir', 'split_info_dir', 'tcn_log_dir', 
                    'tcn_model_dir', 'tcn_feature_dir', 'result_dir',
                    'graph_dir', 'trpo_model_dir']

        split_num_dict = {'JIGSAWS':8, '50Salads_eval':5,   # Problem!!!
                          '50Salads_mid':5, 'GTEA':4}

        class_num_dict = {'JIGSAWS':10, '50Salads_eval':10,   # 10, 9, 17, 10
                          '50Salads_mid':18, 'GTEA':11}

        kernel_size_dict = {'JIGSAWS':51, '50Salads_eval':63, 
                            '50Salads_mid':63, 'GTEA':63}

        # Appending Dataset Subdir
        for d in all_dirs:
            all_params[d] = os.path.join(all_params['parent_dirs'][d], 
                                         all_params['dataset_name'])

        all_params['split_num'] = split_num_dict[value]
        all_params['gesture_class_num'] = class_num_dict[value]

        kernel_size = kernel_size_dict[value] // sample_rate

        model_params = all_params['tcn_params']['model_params']
        model_params['class_num'] = class_num_dict[value]
        model_params['encoder_params']['kernel_size'] = kernel_size
        model_params['decoder_params']['kernel_size'] = kernel_size
        all_params['tcn_params']['model_params'] = model_params

    with open('config.json', 'w') as f:
        json.dump(all_params, f, indent=2)

    importlib.reload(config)


def main():

    # Set seed
    #utils.set_global_seeds(777, True)

    for name in ['JIGSAWS', 'GTEA', '50Salads_eval', '50Salads_mid']:
        update_config_file('dataset_name', name)
        utils.set_up_dirs()
        utils.clean_up()
        experiment_tcn()

    # from config import all_params

    # all_params['rl_params']['env_mode'] = 'full'
    # with open('config.json', 'w') as f:
    #     json.dump(all_params, f, indent=2)
    # experiment_trpo('full')


    # all_params['rl_params']['env_mode'] = 'no_tcn'
    # with open('config.json', 'w') as f:
    #     json.dump(all_params, f, indent=2)
    # experiment_trpo('no_tcn')


    # all_params['rl_params']['env_mode'] = 'no_future'
    # with open('config.json', 'w') as f:
    #     json.dump(all_params, f, indent=2)
    # experiment_trpo('no_future')


    # all_params['rl_params']['env_mode'] = 'no_hint'
    # with open('config.json', 'w') as f:
    #     json.dump(all_params, f, indent=2)
    # experiment_trpo('no_hint')


    # for k in [1, 2, 4, 8, 16, 32]:

    #     all_params['rl_params']['env_mode'] = 'full'
    #     all_params['rl_params']['k_steps'] = [k]
    #     all_params['rl_params']['glimpse'] = [k]
    #     all_params['rl_params']['reward_alpha'] = 0

    #     with open('config.json', 'w') as f:
    #         json.dump(all_params, f, indent=2)

    #     experiment_trpo(str(k))


if __name__ == '__main__':
    main()