from __future__ import division
from __future__ import print_function

import utils
from agent import Agent

import gym
from gym import spaces
import numpy as np
import random

import pdb


class MyEnv(gym.Env):
    def __init__(self,
                 dataset, 
                 statistical_model,
                 class_num,
                 feature_num,
                 k_steps,
                 glimpse,
                 reward_alpha,
                 mode):  # glimpse should > 0 

        self.dataset = dataset
        self.k_steps = k_steps
        self.glimpse = glimpse
        self.reward_alpha = reward_alpha

        self.class_num = class_num
        self.feature_num = feature_num

        self.agent = Agent(name='CleverChang',
                           state_num=self.class_num,
                           **statistical_model)

        self.action_num = len(self.k_steps) * self.class_num
        self.action_space = spaces.Discrete(self.action_num)

        self.mode = mode

        if self.mode == 'full':
            self.observation_num = self.feature_num * (len(self.glimpse)+1) + \
                                                        2 * self.class_num
        elif self.mode == 'no_tcn':
            self.observation_num = 2 * self.class_num
        elif self.mode == 'no_future':
            self.observation_num = self.feature_num + 2 * self.class_num
        elif self.mode == 'no_hint':
            self.observation_num = self.feature_num * (len(self.glimpse)+1)
        else:
            raise Exception('Invalid Env Mode!')

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

        state = []

        if self.mode == 'full':

            state.append(self.feature[self.position])
            for g in self.glimpse:
                if self.position + g < self.episode_len:
                    state.append(self.feature[self.position + g])
                else:
                    state.append(np.zeros(self.feature_num))
                    
            state.append(self.agent.get_state_vector())
            state.append(self.agent.get_hints_vector())

        elif self.mode == 'no_tcn':

            state.append(self.agent.get_state_vector())
            state.append(self.agent.get_hints_vector())

        elif self.mode == 'no_future':

            state.append(self.feature[self.position])

            state.append(self.agent.get_state_vector())
            state.append(self.agent.get_hints_vector())

        elif self.mode == 'no_hint':

            state.append(self.feature[self.position])
            for g in self.glimpse:
                if self.position + g < self.episode_len:
                    state.append(self.feature[self.position + g])
                else:
                    state.append(np.zeros(self.feature_num))

        else:
            raise Exception('Invalid Env Mode!')

        state = np.concatenate(state)
        return state


    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        act_k = action // self.class_num
        act_opt = action % self.class_num

        if act_opt not in [i for i in range(self.class_num)]:
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
        act_k = action // self.class_num
        act_opt = action % self.class_num
        entry = []
        entry.append(act_k)
        entry.append(act_opt)
        entry.append(self.position)
        entry.append(reward)
        entry.append(self.agent.current_state)
        self.full_act_hist.append(entry)

    # For Plot and Debug
    def get_hist_step_sizes(self):

        steps = np.array(self.full_act_hist)
        steps = steps[:,0].astype(int)

        flat_steps = np.zeros_like(self.label)

        running_idx = 0
        for s in steps:
            k = self.k_steps[s]
            flat_steps[running_idx:running_idx+k] = s
            running_idx += k 

        return flat_steps

    def get_accuracy(self):
        return utils.get_accuracy(self.result, self.label)

    def get_edit_score(self):
        return utils.get_edit_score(self.result, self.label)

    def get_overlap_f1(self, overlap):
        return utils.get_overlap_f1_colin(self.result, self.label,
                                          n_classes=self.class_num,
                                          overlap=overlap)