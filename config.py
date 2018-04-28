from __future__ import division
from __future__ import print_function

import json
import pdb
import math

all_params = json.load(open('config.json'))

dataset_name = all_params['dataset_name']

locals().update(all_params['experiment_setup'])
locals().update(all_params[dataset_name])


tcn_params['model_params']['encoder_params']['kernel_size'] //= sample_rate
tcn_params['model_params']['decoder_params']['kernel_size'] //= sample_rate

tcn_params['model_params']['mid_lstm_params'] = None

temp = []
for k in rl_params['k_steps']:
    temp.append(math.ceil(k / sample_rate))
rl_params['k_steps'] = temp

temp = []
for k in rl_params['glimpse']:
    temp.append(math.ceil(k / sample_rate))
rl_params['glimpse'] = temp


#tcn_params['train_params'] = None
