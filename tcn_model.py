from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import pdb

# This module should be tested carefully 

class LSTM_Layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 bi_dir=True, use_gru=True):
        super(LSTM_Layer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_dir = bi_dir
        self.use_gru = use_gru

        if self.use_gru:
            self.lstm = nn.GRU(input_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=bi_dir)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=bi_dir)

    def forward(self, x): # x: (batch,feature,seq)

        x = x.permute(0, 2, 1) 

        batch_size = x.size(0)
        x, _ = self.lstm(x, self.__get_init_state(batch_size)) # x: (batch,seq,hidden)
        
        x = x.permute(0, 2, 1) 
        
        return x

    def __get_init_state(self, batch_size):

        if self.bi_dir:
            nl_x_nd = 2 * self.num_layers
        else:
            nl_x_nd = 1 * self.num_layers

        h0 = torch.zeros(nl_x_nd, batch_size, self.hidden_size)
        h0 = h0.cuda()

        if self.use_gru:
            return h0
        else:
            c0 = torch.zeros(nl_x_nd, batch_size, self.hidden_size)
            c0 = c0.cuda()
            return (h0, c0)


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, x): #(batch, feature, seq)
        divider = torch.max(torch.max(torch.abs(x), dim=0)[0], dim=1)[0] + 1e-5
        divider = divider.unsqueeze(0).unsqueeze(2)
        divider = divider.repeat(x.size(0), 1, x.size(2))
        x = x / divider
        return x


class Encoder(nn.Module):
    def __init__(self, input_size,
                 layer_type, layer_sizes, 
                 kernel_size=None, norm_type=None,
                 downsample=True):
        super(Encoder, self).__init__()

        if layer_type not in ['TempConv', 'Bi-LSTM']:
            raise Exception('Invalid Layer Type')
        if layer_type == 'TempConv' and kernel_size is None:
            raise Exception('Kernel Size For TempConv Not Specified')

        self.output_size = layer_sizes[-1]

        module_list = []

        for layer in range(len(layer_sizes)):
            if layer == 0:
                in_chl = input_size
            else:
                in_chl = layer_sizes[layer-1]
            out_chl = layer_sizes[layer]

            if layer_type == 'TempConv':
                conv_pad = kernel_size // 2
                module_list.append(('conv_{}'.format(layer), 
                    nn.Conv1d(in_chl, out_chl, kernel_size, padding=conv_pad)))
            elif layer_type == 'Bi-LSTM':
                module_list.append(('lstm_{}'.format(layer), 
                    LSTM_Layer(in_chl, out_chl // 2, 1, bi_dir=True)))

            if norm_type == 'Channel':
                module_list.append(('cn_{}'.format(layer), 
                    ChannelNorm()))
            elif norm_type == 'Batch':
                module_list.append(('bn_{}'.format(layer), 
                    nn.BatchNorm1d(out_chl)))
            elif norm_type == 'Instance':
                module_list.append(('in_{}'.format(layer), 
                    nn.InstanceNorm1d(out_chl)))
            else:
                print('No Norm Used!')           

            if layer_type == 'TempConv':
                module_list.append(('relu_{}'.format(layer), 
                    nn.ReLU()))
            else:
                pass

            if downsample:
                module_list.append(('pool_{}'.format(layer), 
                    nn.MaxPool1d(kernel_size=2, stride=2)))

        self.module = nn.Sequential(OrderedDict(module_list))


    def forward(self, x): # x: (batch,feature, seq)
        return self.module(x)



class Decoder(nn.Module):
    def __init__(self, input_size,
                 layer_type, layer_sizes, 
                 kernel_size=None, transposed_conv=None,
                 norm_type=None):
        super(Decoder, self).__init__()

        if layer_type not in ['TempConv', 'Bi-LSTM']:
            raise Exception('Invalid Layer Type')
        if layer_type == 'TempConv' and kernel_size is None:
            raise Exception('Kernel Size For TempConv Not Specified')
        if layer_type == 'TempConv' and transposed_conv is None:
            raise Exception('If Use Transposed Conv Not Specified')

        self.output_size = layer_sizes[-1]

        module_list = []

        for layer in range(len(layer_sizes)):
            if layer == 0:
                in_chl = input_size
            else:
                in_chl = layer_sizes[layer-1]
            out_chl = layer_sizes[layer]

            module_list.append(('up_{}'.format(layer), 
                nn.Upsample(scale_factor=2)))

            if layer_type == 'TempConv':
                conv_pad = kernel_size // 2
                if transposed_conv:
                    module_list.append(('conv_{}'.format(layer), 
                        nn.ConvTranspose1d(in_chl, out_chl, kernel_size, 
                                                    padding=conv_pad)))
                else:
                    module_list.append(('conv_{}'.format(layer), 
                        nn.Conv1d(in_chl, out_chl, kernel_size,
                                            padding=conv_pad)))
            elif layer_type == 'Bi-LSTM':
                module_list.append(('lstm_{}'.format(layer), 
                    LSTM_Layer(in_chl, out_chl // 2, 1, bi_dir=True)))

            if norm_type == 'Channel':
                module_list.append(('cn_{}'.format(layer), 
                    ChannelNorm()))
            elif norm_type == 'Batch':
                module_list.append(('bn_{}'.format(layer), 
                    nn.BatchNorm1d(out_chl)))
            elif norm_type == 'Instance':
                module_list.append(('in_{}'.format(layer), 
                    nn.InstanceNorm1d(out_chl)))
            else:
                print('No Norm Used!')           

            if layer_type == 'TempConv':
                module_list.append(('relu_{}'.format(layer), 
                    nn.ReLU()))
            else:
                pass

        self.module = nn.Sequential(OrderedDict(module_list))


    def forward(self, x): # x: (batch,feature, seq)
        return self.module(x)



class EncoderDecoderNet(nn.Module):
    def __init__(self, class_num, fc_size,
                 encoder_params, 
                 decoder_params, 
                 mid_lstm_params=None):

        super(EncoderDecoderNet, self).__init__()

        self.encoder = Encoder(**encoder_params)

        self.middle_lstm = None
        if mid_lstm_params is not None:
            self.middle_lstm = LSTM_Layer(mid_lstm_params['input_size'],
                                      mid_lstm_params['hidden_size'],
                                      mid_lstm_params['layer_num'],
                                      bi_dir=True)  # batch_first
        
        self.decoder = Decoder(**decoder_params)

        self.fc1 = nn.Linear(self.decoder.output_size, fc_size)
        self.fc2 = nn.Linear(fc_size, class_num)


    def forward(self, x):
        
        x = F.relu(self.extract_feature(x))
        x = self.fc2(x)

        return x


    def extract_feature(self, x):

        x = x.permute(0, 2, 1)  

        x = self.encoder(x)
        if self.middle_lstm is not None:
            x = self.middle_lstm(x)

        x = self.decoder(x)
        x = x.permute(0, 2, 1)

        x = self.fc1(x)

        return x
