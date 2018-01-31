from mpi4py import MPI
import os.path as osp
import gym
import logging
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import tensorflow as tf

from my_dataset import FeatureDataset
from my_env import MyEnv
from config import *
import os
import utils
import numpy as np
import pdb


def test(feature_type, tcn_run_idx, split_idx, run_idx):

    train_file = 'tcn_feature_train_{}_run_{}_split_{}.npy'.format(
                              feature_type, tcn_run_idx, split_idx)
    train_file = os.path.join(tcn_feature_dir, train_file)
    
    train_dataset = FeatureDataset(train_file)

    transitions = utils.get_normalized_transition_matrix(train_dataset)
    durations = utils.get_duration_statistics(train_dataset)
    statistical_model = {'initials': transitions[-1,:-1],
                         'transitions': transitions[:-1,:-1], 
                         'durations':durations}


    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)


    test_file = 'tcn_feature_test_{}_run_{}_split_{}.npy'.format(
                              feature_type, tcn_run_idx, split_idx)
    test_file = os.path.join(tcn_feature_dir, test_file)

    temp_dataset = FeatureDataset(test_file, test_index=0)
    temp_raw_env = MyEnv(temp_dataset,
                    statistical_model,
                    k_steps=k_steps,
                    glimpse=glimpse,
                    reward_alpha=reward_alpha)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, 
                         ob_space=temp_raw_env.observation_space, 
                         ac_space=temp_raw_env.action_space,
                         hid_size=pi_hidden_size, 
                         num_hid_layers=pi_hidden_layer)

    temp_env = bench.Monitor(temp_raw_env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)))

    gym.logger.setLevel(logging.WARN)
    pi = policy_fn('pi', temp_env.observation_space, temp_env.action_space)


    # Restore model
    run_model_dir = 'trpo_model_{}_tcn_{}_split_{}_run_{}'.format(
                        feature_type, tcn_run_idx, split_idx, run_idx)
    run_model_dir = os.path.join(trpo_model_dir, run_model_dir)
    model_file = os.path.join(run_model_dir, 'model')

    tf.train.Saver().restore(sess, model_file)


    # Test
    test_num = len(FeatureDataset(test_file))

    result = np.zeros((test_num, trpo_test_run_num, 9))

    for i in range(test_num):

        test_dataset = FeatureDataset(test_file, test_index=i)
        raw_env = MyEnv(test_dataset,
                        statistical_model,
                        k_steps=k_steps,
                        glimpse=glimpse,
                        reward_alpha=reward_alpha)

        env = bench.Monitor(raw_env, logger.get_dir() and
                            osp.join(logger.get_dir(), str(rank)))


        episode_result = np.zeros((trpo_test_run_num, 9))

        for episode in range(trpo_test_run_num):
            obs, done = env.reset(), False
            while not done:
                obs, rew, done, _ = env.step(pi.act(True, obs)[0])

            hist = np.array(raw_env.full_act_hist)
            hist = hist[:,0].astype(int)
            
            episode_result[episode,0] = raw_env.get_accuracy()
            episode_result[episode,1] = raw_env.get_edit_score()
            episode_result[episode,2] = raw_env.get_overlap_f1(0.1)
            episode_result[episode,3] = raw_env.get_overlap_f1(0.25)
            episode_result[episode,4] = raw_env.get_overlap_f1(0.5)
            episode_result[episode,5] = raw_env.get_overlap_f1(0.75)
            episode_result[episode,6] = (hist==0).sum() / hist.size
            episode_result[episode,7] = (hist==1).sum() / hist.size
            episode_result[episode,8] = (hist==2).sum() / hist.size

            # Plot
            graph_file = 'barcode_{}_tcn_{}_split_{}_run_{}_seq_{}_ep_{}'.format(
                feature_type, tcn_run_idx, split_idx, run_idx, i, episode)
            graph_file = os.path.join(graph_dir, graph_file)
            utils.plot_barcode(gt=raw_env.label, pred=raw_env.result,
                               steps=raw_env.get_hist_step_sizes(),
                               show=False, save_file=graph_file)

        result[i,:,:] = episode_result

        print('Acc: ', episode_result.mean(0)[0])
        print('Edit: ', episode_result.mean(0)[1])
        print('F10: ', episode_result.mean(0)[2])
        print('F25: ', episode_result.mean(0)[3])
        print('F50: ', episode_result.mean(0)[4])
        print('F75: ', episode_result.mean(0)[5])
        print('K0: ', episode_result.mean(0)[6])
        print('K1: ', episode_result.mean(0)[7])
        print('K2: ', episode_result.mean(0)[8])

    result_file = 'trpo_result_{}_tcn_{}_split_{}_run_{}.npy'.format(
                        feature_type, tcn_run_idx, split_idx, run_idx)
    result_file = os.path.join(result_dir, result_file)

    np.save(result_file, result)  #(5, 10, 9)


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--feature_type', type=str, default='sensor')
    parser.add_argument('--tcn_run_idx', type=int, default=1)
    parser.add_argument('--split_idx', type=int, default=1)
    parser.add_argument('--run_idx', type=int, default=1)
    
    args = parser.parse_args()
    
    if args.feature_type not in ['sensor', 'visual']:
        raise Exception('Invalid Feature Type')

    test(feature_type=args.feature_type,
         tcn_run_idx=args.tcn_run_idx,
         split_idx=args.split_idx,
         run_idx=args.run_idx)



if __name__ == '__main__':
    main()

