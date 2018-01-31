from __future__ import division
from __future__ import print_function

import json
import pdb

all_params = json.load(open('config.json'))

locals().update(all_params)

tcn_params['model_params']['mid_lstm_params'] = None
#tcn_params['train_params'] = None
