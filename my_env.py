from __future__ import division
from __future__ import print_function

import utils
from agent import Agent

from config import gesture_class_num, tcn_feature_num

import gym
from gym import spaces
import numpy as np
import random

import pdb


class MyEnv(gym.Env):
    def __init__(self,
                 dataset, 
                 statistical_model,
                 k_steps,
                 glimpse,
                 reward_alpha):  # glimpse should > 0 

        self.dataset = dataset
        self.k_steps = k_steps
        self.glimpse = glimpse
        self.reward_alpha = reward_alpha

        self.feature_num = tcn_feature_num

        self.agent = Agent(name='CleverChang',
                           state_num=gesture_class_num,
                           **statistical_model)


        self.action_num = len(self.k_steps) * gesture_class_num

        self.action_space = spaces.Discrete(self.action_num)

        self.observation_num = self.feature_num * (len(self.glimpse)+1) + \
                               2 * gesture_class_num
        
        # self.observation_num = self.feature_num * (len(self.glimpse)+1) 

        bounds = np.ones(self.observation_num) * np.inf             # To be improved
        self.observation_space = spaces.Box(-bounds, bounds)

        self.state = None


    def _reset(self):
        #data = self.dataset[random.randrange(len(self.dataset))]
        data = random.choice(self.dataset)

        self.label = data['label']
        self.feature = data['value'].astype('float')

        self.episode_len = self.feature.shape[0]

        self.position = 0
        self.result = np.zeros_like(self.label) - 1
        self.full_act_hist = [] # For Debug
        self.agent.reset()

        self.state = self._get_state()

        return self.state


    def _get_state(self):

        if self.position >= self.episode_len:
            raise Exception('Agent out of environment')

        state = [self.feature[self.position]]

        for g in self.glimpse:
            if self.position + g < self.episode_len:
                state.append(self.feature[self.position + g])
            else:
                state.append(np.zeros(self.feature_num))
        
        state.append(self.agent.get_state_vector())
        state.append(self.agent.get_hints_vector())

        state = np.concatenate(state)

        return state


    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        act_k = action // gesture_class_num
        act_opt = action % gesture_class_num

        if act_opt not in [i for i in range(gesture_class_num)]:
            raise Exception('Invalid act_opt!')

        k_step = self.k_steps[act_k]
        self.agent.opt(act_opt)
        self.agent.update_current_duration(k_step)

        self.result[self.position:self.position+k_step] = act_opt

        error = np.not_equal(self.result[self.position:self.position+k_step],
                    self.label[self.position:self.position+k_step]).sum()

        reward = self.reward_alpha * k_step - error

        self._update_full_act_hist(action, reward) # For Debug

        self.position += k_step
        if self.position >= self.episode_len:
            self.state = np.zeros(self.observation_num)
            done = True
        else:
            self.state = self._get_state()
            done = False

        if self.agent.current_state != act_opt:
            raise Exception('Inconsistant state!')
            
        return self.state, reward, done, {}

    def _update_full_act_hist(self, action, reward):
        act_k = action // gesture_class_num
        act_opt = action % gesture_class_num
        entry = []
        entry.append(act_k)
        entry.append(act_opt)
        entry.append(self.position)
        entry.append(reward)
        entry.append(self.agent.current_state)
        self.full_act_hist.append(entry)

    def get_accuracy(self):
        return utils.get_accuracy(self.result, self.label)

    def get_edit_score(self):
        return utils.get_edit_score(self.result, self.label)

    def get_overlap_f1(self, overlap):
        return utils.get_overlap_f1_colin(self.result, self.label,
                                          n_classes=gesture_class_num,
                                          overlap=overlap)