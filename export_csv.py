import numpy as np
import os
import pdb

# To be modified

datasets = os.listdir('./result')

for dataset in datasets:
    file = open('summary_{}.csv'.format(dataset),'w')

    result_dir = os.path.join('./result', dataset)

    entries = os.listdir(result_dir)
    entries.sort()

    for entry in entries:
        
        result_file = os.path.join(result_dir, entry)

        if entry.startswith('trpo'):
            entry_result = np.load(result_file).mean(0).mean(0).mean(0).mean(0)
        elif entry.startswith('tcn'):
            entry_result = np.load(result_file).mean(0).mean(0)

        line = str(entry)
        for i in range(len(entry_result)):
            line += ','
            line += str(entry_result[i])
        line += '\n'

        file.write(line)