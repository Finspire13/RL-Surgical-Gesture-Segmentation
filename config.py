from __future__ import division
from __future__ import print_function

import os

# Data parameters
gesture_class_num = 10
sample_rate = 1    # Feature file alrady sampled

#Files and dirs
raw_feature_dir = './raw_features'
split_info_dir = './splits'

tcn_log_dir = './tcn_log'
tcn_model_dir = './tcn_model'
tcn_feature_dir = './tcn_features'

result_dir = './result'
graph_dir = './graph'

trpo_model_dir = './trpo_model'

# TCN Model parameters
encoder_params = {'input_size': None,  # To be defined later
                  'layer_type': ['TempConv', 'Bi-LSTM'][0],
                  'layer_sizes': [64, 96, 128], 
                  'kernel_size': 51,
                  'norm_type': 'Channel'}

# mid_lstm_params = {'input_size': mid_decoder_input[i],
#                    'hidden_size': mid_decoder_input[i] // 2,
#                    'layer_num': 2}
mid_lstm_params = None

decoder_params = {'input_size': 128,
                  'layer_type': ['TempConv', 'Bi-LSTM'][0],
                  'layer_sizes': [96, 64, 64], 
                  'transposed_conv': True,
                  'kernel_size': 51,
                  'norm_type': 'Channel'}

model_params = {'class_num': gesture_class_num,
                'fc_size': 32,
                'encoder_params': encoder_params,
                'decoder_params': decoder_params, 
                'mid_lstm_params': mid_lstm_params}


# Training parameters
train_params = {'num_epochs': 300, 
                'learning_rate': 0.00001,     
                'batch_size': 1, 
                'weight_decay': 0.0001} 

# train_params = None


# RL data parameters
tcn_feature_num = 32

# RL env parameters
k_steps = [4, 21]
glimpse = [4, 21]
reward_alpha = 0.1

# RL policy parameters
pi_hidden_size = 64
pi_hidden_layer = 1

# RL training parameters
trpo_num_timesteps = 1e3 #5e5
discount_factor = 0.9


# Experiment Setup
tcn_run_num = 1
trpo_test_run_num = 1
trpo_train_run_num = 1
split_num = 8