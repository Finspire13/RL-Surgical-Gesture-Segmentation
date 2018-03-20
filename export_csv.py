import numpy as np
import os

file = open('summary.csv','w')

entries = os.listdir('./result')
entries.sort()

for entry in entries:
    
    result_file = os.path.join('./result',entry)

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