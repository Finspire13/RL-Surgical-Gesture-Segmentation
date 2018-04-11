from __future__ import division
from __future__ import print_function

import os
import torch
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('AGG')    # Show Plot Disabled
import matplotlib.pyplot as plt
import random
import string


import pdb

################## Random ##########################

def set_global_seeds(seed, use_cudnn=True):
    torch.backends.cudnn.enabled = use_cudnn   # Too slow
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def generate_random_str(size, chs=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chs) for _ in range(size))

################### Trail List ######################

def get_all_trail():
    from config import split_info_dir
    all_trail_file = os.path.join(split_info_dir, 'all.txt')

    with open(all_trail_file) as file:
        trail_list = file.readlines()
    trail_list = [t.strip() for t in trail_list]

    return trail_list

def get_cross_val_splits():
    from config import split_info_dir

    split_dirs = os.listdir(split_info_dir)
    split_dirs = sorted([s for s in split_dirs if "Split" in s])

    cross_val_splits = []
    for split_dir in split_dirs:
        train_file = os.path.join(split_info_dir, split_dir, 'train.txt')
        test_file = os.path.join(split_info_dir, split_dir, 'test.txt')
        
        with open(train_file) as file:
            train_list = file.readlines()
        with open(test_file) as file:
            test_list = file.readlines()

        train_list = [t.strip() for t in train_list]
        test_list = [t.strip() for t in test_list]

        cross_val_splits.append({'train': train_list,
                                 'test': test_list,
                                 'name': split_dir})

    return cross_val_splits


################### Load File ######################

def get_tcn_model_file(naming):
    from config import tcn_model_dir
    tcn_model_file = os.path.join(tcn_model_dir, naming)
    if not os.path.exists(tcn_model_file):
        os.makedirs(tcn_model_file)
    tcn_model_file = os.path.join(tcn_model_file, 'tcn_model.pkl')
    return tcn_model_file

def get_tcn_log_sub_dir(naming):
    from config import tcn_log_dir
    sub_dir = os.path.join(tcn_log_dir, naming)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    return sub_dir

def clear_dir(dir):
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        try:
            shutil.rmtree(filepath)
        except NotADirectoryError:
            os.remove(filepath)

# To be improved
def set_up_dirs():
    from config import (result_dir, tcn_log_dir, tcn_model_dir, 
                        tcn_feature_dir, trpo_model_dir, graph_dir)

    for i in [result_dir, tcn_log_dir, tcn_model_dir, 
              tcn_feature_dir, trpo_model_dir, graph_dir]:
        if not os.path.exists(i):
            os.makedirs(i)

# To be improved
def clean_up():
    from config import (result_dir, tcn_log_dir, tcn_model_dir, 
                        tcn_feature_dir, trpo_model_dir, graph_dir)

    for i in [result_dir, tcn_log_dir, tcn_model_dir, 
              tcn_feature_dir, trpo_model_dir, graph_dir]:
        clear_dir(i)    


################## Gesture Statistics ####################

def get_class_counts(dataset):  # RAW
    from config import gesture_class_num
    class_num = gesture_class_num

    counts = [0 for i in range(class_num)]

    for data in dataset:
        gesture = data['gesture']
        gesture = gesture[gesture!=-1]

        for i in range(class_num):
            counts[i] += (gesture==i).sum()

    return counts

def get_class_weights(dataset):  # RAW
    from config import gesture_class_num
    class_num = gesture_class_num

    counts = get_class_counts(dataset)

    weights = [1/i for i in counts]
    w_sum = sum(weights)
    for i in range(class_num):
        weights[i] = weights[i] * class_num / w_sum

    return weights


def get_transition_matrix(dataset): # TCN
    from config import gesture_class_num

    class_num = gesture_class_num + 1  # Including Init
    matrix = np.zeros((class_num, class_num))  # 10: Init

    for data in dataset:
        gesture = data['label']

        last = class_num - 1  #init
        for i in range(len(gesture)):
            current = int(gesture[i])
            matrix[last][current] += 1
            last = current

    return matrix.astype(int)


def get_normalized_transition_matrix(dataset): # TCN
    from config import gesture_class_num

    class_num = gesture_class_num + 1   # Including Init
    matrix = get_transition_matrix(dataset).astype(float)

    for i in range(class_num):
        matrix[i][i] = 0
        matrix[i] = matrix[i] / (matrix[i].sum() + 1e-20)

    return matrix

def get_gesture_durations(datasets): # TCN   # Multiple dataset possible
    from config import gesture_class_num
    
    class_num = gesture_class_num
    durations = [[] for i in range(class_num)]

    if type(datasets) != list:
        raise Exception('Input should be put into an array!')

    for dataset in datasets:
        for data in dataset:
            gesture = data['label']

            count = 1
            for i in range(1, len(gesture)):
                if gesture[i-1] == gesture[i]:
                    count += 1
                else:
                    durations[gesture[i-1]].append(count)
                    count = 1

            durations[gesture[i-1]].append(count)

    return durations

def get_duration_statistics(dataset): # TCN

    durations = get_gesture_durations([dataset])

    mus = [np.array(i).mean() for i in durations]
    sigmas = [np.array(i).std() for i in durations]

    # Empty durations handled: Caution!!!
    mus = [0 if np.isnan(i) else i  for i in mus]
    sigmas = [1 if np.isnan(i) else i  for i in sigmas]

    return np.array([mus, sigmas])

def get_min_length(datasets):  # TCN          # Multiple dataset possible

    durations = get_gesture_durations(datasets)

    # Empty durations handled: Caution!!!
    durations = [i if i else [float('inf')]  for i in durations]

    mins = [np.array(i).min() for i in durations]
    min_min = np.array(mins).min()

    return float(min_min)

def get_min_mean_length(datasets):  # TCN     # Multiple dataset possible

    durations = get_gesture_durations(datasets)

    # Empty durations handled: Caution!!!
    durations = [i if i else [float('inf')]  for i in durations]

    means = [np.array(i).mean() for i in durations]
    min_mean = np.array(means).min()

    return min_mean


################## Visualization ####################

def visualize_result(result):
    result_string = []
    last = ''
    for i in range(result.size):
        label = str(get_reverse_mapped_gesture_label(result[i]))
        if label != last:
            result_string.append(label)
            last = label

    result_string = '-'.join(result_string)

    return result_string


def plot_trail(ls, pred=None, ys=None, show=True, save_file=None):

    fig = plt.figure()
    xs = np.arange(len(ls))
    plt.plot(xs, ls, 'b')
    if ys is not None:
        plt.plot(xs, ys, 'r')
    if pred is not None:
        plt.plot(xs, pred, 'g')
    if save_file is not None:
        fig.savefig(save_file)
    if show:
        plt.show()

    plt.close(fig)


def plot_barcode(gt=None, pred=None, visited_pos=None,
                 show=True, save_file=None):
    from config import gesture_class_num

    if gesture_class_num <= 10:
        color_map = plt.cm.tab10
    else:
        color_map = plt.cm.tab20

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, 
                interpolation='nearest', vmin=0, vmax=gesture_class_num-1)

    fig = plt.figure(figsize=(18, 4))

    # a horizontal barcode
    if gt is not None:
        ax1 = fig.add_axes([0, 0.65, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        ax1.imshow(gt.reshape((1, -1)), **barprops)

    if pred is not None:
        ax2 = fig.add_axes([0, 0.35, 1, 0.2], **axprops)
        ax2.set_title('Predicted')
        ax2.imshow(pred.reshape((1, -1)), **barprops)

    if visited_pos is not None:
        ax3 = fig.add_axes([0, 0.15, 1, 0.1], **axprops)
        ax3.set_title('Steps of Agent')
        ax3.set_xlim(0, len(gt))
        ax3.plot(visited_pos, np.ones_like(visited_pos), 'ro', markersize=1)

    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    if show:
        plt.show()

    plt.close(fig)


################## Metrics ####################

def get_result_string(result):
    from itertools import groupby

    result_string = ''
    for i in range(result.size):
        result_string += str(int(result[i]))  # No negtive allowed

    result_string = ''.join(i for i, _ in groupby(result_string))
    return result_string


# levenstein
def get_edit_score(out, gt):
    import editdistance

    if type(out) == list:
        tmp = [get_edit_score(out[i], gt[i]) for i in range(len(out))]
        return np.mean(tmp)
    else:
        gt_string = get_result_string(gt)
        out_string = get_result_string(out)
        max_len = max(len(gt_string), len(out_string))
        edit_score = 1 - editdistance.eval(gt_string, out_string) / max_len
        return edit_score * 100

def get_accuracy(out, gt):
    if type(out) == list:
        return np.mean(np.concatenate(out)==np.concatenate(gt)) * 100
    else:
        return np.mean(out==gt) * 100


################## Colin Lea ####################

from numba import jit, int64, boolean
@jit("float64(int64[:], int64[:], boolean)")
def levenstein_(p,y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1] 
            else:
                D[i,j] = min(D[i-1,j]+1,
                             D[i,j-1]+1,
                             D[i-1,j-1]+1)
    
    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col) ) * 100
    else:
        score = D[-1,-1]

    return score

def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split

def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals

def get_edit_score_colin(P, Y, norm=True, bg_class=None, **kwargs):
    if type(P) == list:
        tmp = [get_edit_score_colin(P[i], Y[i], norm, bg_class)
                 for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_ = segment_labels(P)
        Y_ = segment_labels(Y)
        if bg_class is not None:
            P_ = [c for c in P_ if c!=bg_class]
            Y_ = [c for c in Y_ if c!=bg_class]
        return levenstein_(P_, Y_, norm)

def get_accuracy_colin(P, Y, **kwargs):  # Average acc
    def acc_(p,y):
        return np.mean(p==y)*100
    if type(P) == list:
        return np.mean([np.mean(P[i]==Y[i]) for i in range(len(P))])*100
    else:
        return acc_(P,Y)


def get_overlap_f1_colin(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p,y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1


        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()
        
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)  #RuntimeWarning: invalid value encountered in double_scalars

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1*100

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)