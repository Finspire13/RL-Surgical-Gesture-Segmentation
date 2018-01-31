#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
#import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
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
from config import (tcn_feature_dir, rl_params, 
                    trpo_model_dir, gesture_class_num)
from random import randint
import os
import utils
import pdb

def train(seed, feature_type, tcn_run_idx, split_idx, run_idx):

    feature_train_template = 'tcn_feature_train_{}_run_{}_split_{}.npy'
    run_model_dir_template = 'trpo_model_{}_tcn_{}_split_{}_run_{}'

    train_file = feature_train_template.format(feature_type, 
                                      tcn_run_idx, split_idx)

    train_file = os.path.join(tcn_feature_dir, train_file)

    dataset = FeatureDataset(train_file)

    transitions = utils.get_normalized_transition_matrix(dataset)
    durations = utils.get_duration_statistics(dataset)
    statistical_model = {'initials': transitions[-1,:-1],
                         'transitions': transitions[:-1,:-1], 
                         'durations':durations}


    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = MyEnv(dataset,
                statistical_model,
                class_num=gesture_class_num,
                feature_num=rl_params['tcn_feature_num'],
                k_steps=rl_params['k_steps'],
                glimpse=rl_params['glimpse'],
                reward_alpha=rl_params['reward_alpha'],
                mode=rl_params['env_mode'])

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, 
                         ob_space=env.observation_space, 
                         ac_space=env.action_space,
                         hid_size=rl_params['pi_hidden_size'], 
                         num_hid_layers=rl_params['pi_hidden_layer'])

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, 
                   timesteps_per_batch=1024, max_kl=0.01, 
                   cg_iters=10, cg_damping=0.1,
                   max_timesteps=rl_params['trpo_num_timesteps'], 
                   gamma=rl_params['discount_factor'], 
                   lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()


    # Save Model
    run_model_dir = run_model_dir_template.format(feature_type, 
                                tcn_run_idx, split_idx, run_idx)
    run_model_dir = os.path.join(trpo_model_dir, run_model_dir)
    if not os.path.exists(run_model_dir):
        os.makedirs(run_model_dir)

    model_file = os.path.join(run_model_dir, 'model')

    saver = tf.train.Saver()
    saver.save(sess, model_file)
    
    sess.close()
    #tf.reset_default_graph()
    
    


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--feature_type', type=str, default='sensor')
    parser.add_argument('--tcn_run_idx', type=int, default=1)
    parser.add_argument('--split_idx', type=int, default=1)
    parser.add_argument('--run_idx', type=int, default=1)

    args = parser.parse_args()
    logger.configure()

    rng_seed = randint(0, 1000)
    print(rng_seed)

    if args.feature_type not in ['sensor', 'visual']:
        raise Exception('Invalid Feature Type')

    train(seed=rng_seed,
          feature_type=args.feature_type,
          tcn_run_idx=args.tcn_run_idx,
          split_idx=args.split_idx,
          run_idx=args.run_idx)

if __name__ == '__main__':
    main()

