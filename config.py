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

# Model parameters
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
train_params = {'num_epochs': 300, #300
                'learning_rate': 0.00001,     
                'batch_size': 1, 
                'weight_decay': 0.0001} 

# train_params = None


# Experiment Setup
tcn_run_num = 3