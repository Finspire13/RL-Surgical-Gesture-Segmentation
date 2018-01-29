import os
import numpy as np
from config import result_dir

import pdb

feature_type = ['sensor', 'visual'][0]

full_result = []

for split in range(1, 9):

    split_result = []

    for run in range(1, 6):
        result_file = result_file = 'result_{}_split_{}_run_{}.npy'.format(
                                                feature_type, split, run)
        result_file = os.path.join(result_dir, result_file)

        run_result = np.load(result_file)  #5,6,10

        # print(run_result.std(2).mean())

        run_result = run_result.mean(2)
        run_result = np.expand_dims(run_result, axis=2) #5,6,1

        split_result.append(run_result)

    split_result = np.concatenate(split_result, axis=2)  #5,6,5
    split_result = split_result.mean(2)    #5,6
 
    full_result.append(split_result)

full_result = np.concatenate(full_result, axis=0)  #39,6

print(full_result.shape)

print('Acc: ', full_result[:,0].mean())
print('Edit: ', full_result[:,1].mean())
print('F10: ', full_result[:,2].mean())
print('F25: ', full_result[:,3].mean())
print('F50: ', full_result[:,4].mean())
print('F75: ', full_result[:,5].mean())
print('K0: ', full_result[:,6].mean())
print('K1: ', full_result[:,7].mean())
print('K2: ', full_result[:,8].mean())
