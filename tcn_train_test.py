from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch 
import torch.nn as nn
from random import randrange
from torch.autograd import Variable
from tcn_model import EncoderDecoderNet
from my_dataset import JIGSAWS_Dataset

from logger import Logger
import utils
import pdb

from config import *


def train_model(model, 
                train_dataset, 
                val_dataset, 
                num_epochs,
                learning_rate,
                batch_size,
                weight_decay,
                loss_weights=None, 
                trained_model_file=None, 
                log_dir=None):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size, shuffle=True)
    model.train()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).cuda(),
                        ignore_index=-1)

    # Logger
    if log_dir is not None:
        logger = Logger(log_dir) 

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,
                                            weight_decay=weight_decay)


    step = 1
    for epoch in range(num_epochs):
        print(epoch)
        for i, data in enumerate(train_loader):

            feature = data['feature'].float()
            feature = Variable(feature).cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = Variable(gesture).cuda() 

            # Forward
            out = model(feature)
            flatten_out = out.view(-1, out.shape[-1])

            loss = criterion(input=flatten_out, target=gesture)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if log_dir is not None:
                logger.scalar_summary('loss', loss.data[0], step)

            step += 1


        if log_dir is not None:
            train_result = test_model(model, train_dataset)
            t_accuracy, t_edit_score, t_loss, _ = train_result

            val_result = test_model(model, val_dataset)
            v_accuracy, v_edit_score, v_loss, _ = val_result

            logger.scalar_summary('t_accuracy', t_accuracy, epoch)
            logger.scalar_summary('t_edit_score', t_edit_score, epoch)
            logger.scalar_summary('t_loss', t_loss, epoch)

            logger.scalar_summary('v_accuracy', v_accuracy, epoch)
            logger.scalar_summary('v_edit_score', v_edit_score, epoch)
            logger.scalar_summary('v_loss', v_loss, epoch)
            

        if trained_model_file is not None:
            torch.save(model.state_dict(), trained_model_file)


def test_model(model, test_dataset, loss_weights=None):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=1, shuffle=False)
    model.eval()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).cuda(),
                        ignore_index=-1)

    #Test the Model
    total_loss = 0
    preditions = []
    gts=[]

    for i, data in enumerate(test_loader):

        feature = data['feature'].float()
        feature = Variable(feature, volatile=True).cuda()

        gesture = data['gesture'].long()
        gesture = gesture.view(-1)
        gesture = Variable(gesture, volatile=True).cuda() 

        # Forward
        out = model(feature)
        out = out.squeeze(0)

        loss = criterion(input=out, target=gesture)

        total_loss += loss.data[0]

        pred = out.data.max(1)[1]

        trail_len = (gesture.data.cpu().numpy()!=-1).sum()
        gesture = gesture[:trail_len]
        pred = pred[:trail_len]

        # utils.plot_barcode(gesture.data.cpu().numpy(),
        #                    pred.cpu().numpy(), show=False,
        #                    save_file='./graphs/{}'.format(data['name'][0]))

        preditions.append(pred.cpu().numpy())
        gts.append(gesture.data.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    edit_score = utils.get_edit_score(preditions, gts)
    accuracy = utils.get_accuracy_colin(preditions, gts)
    #accuracy = utils.get_accuracy(preditions, gts)

    f_scores = []
    for overlap in [0.1, 0.25, 0.5, 0.75]:
        f_scores.append(utils.get_overlap_f1_colin(preditions, gts,
            n_classes=gesture_class_num, bg_class=None, overlap=overlap))

    model.train()
    return accuracy, edit_score, avg_loss, f_scores


######################### Main Process #########################

def cross_validate(model_params, train_params, feature_type, naming):

    # Get trail list
    cross_val_splits = utils.get_cross_val_splits()

    # Cross-Validation Result
    result = []

    # Cross Validation
    for split_idx, split in enumerate(cross_val_splits):
        feature_dir = os.path.join(raw_feature_dir, split['name'])
        test_trail_list = split['test']
        train_trail_list = split['train']

        split_naming = naming + '_split_{}'.format(split_idx+1)

        trained_model_file = utils.get_tcn_model_file(split_naming)
        log_dir = utils.get_tcn_log_sub_dir(split_naming)

        # Model
        model = EncoderDecoderNet(**model_params)
        model = model.cuda()

        print(model)

        n_layers = len(model_params['encoder_params']['layer_sizes'])

        # Dataset
        train_dataset = JIGSAWS_Dataset(feature_dir,
                                        train_trail_list,
                                        feature_type=feature_type,
                                        encode_level=n_layers,
                                        sample_rate=sample_rate,
                                        sample_aug=True,
                                        normalization=[None, None])

        test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
        test_dataset = JIGSAWS_Dataset(feature_dir,
                                       test_trail_list,
                                       feature_type=feature_type,
                                       encode_level=n_layers,
                                       sample_rate=sample_rate,
                                       sample_aug=False,
                                       normalization=test_norm)

        loss_weights = utils.get_class_weights(train_dataset)
        #loss_weights = None

        if train_params is not None:
            train_model(model, 
                        train_dataset,
                        test_dataset, 
                        **train_params,
                        loss_weights=loss_weights,
                        trained_model_file=trained_model_file,
                        log_dir=log_dir)
                        #log_dir=None)

        model.load_state_dict(torch.load(trained_model_file))

        acc, edit, _, f_scores = test_model(model, test_dataset, 
                                        loss_weights=loss_weights)

        result.append([acc, edit, f_scores[0], f_scores[1], 
                                  f_scores[2], f_scores[3]])


    result = np.array(result)

    return result