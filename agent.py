from __future__ import division
from __future__ import print_function

import utils
import numpy as np
import scipy.stats

import pdb

class Agent(object):

    def __init__(self, name, 
                 state_num,
                 initials,
                 transitions, 
                 durations):

        self.name = name
        self.states = [i for i in range(state_num+1)]  # 10: init

        self.initials = initials
        self.transitions = transitions
        self.durations = durations

        self.current_duration = 0
        self.current_state = self.states[-1]  # init

    def update_current_duration(self, steps):
        self.current_duration += steps

    def opt(self, state):
        if state != self.current_state:
            self.current_duration = 0
            self.current_state = state
        else:
            pass

    def reset(self):
        self.current_duration = 0
        self.current_state = self.states[-1]

    def get_state_vector(self):  # one hot  10d
        state_vector = np.zeros(len(self.states)-1)
        if self.current_state == self.states[-1]:
            pass  # all zeros
        else:
            state_vector[self.current_state] = 1

        return state_vector

    def get_hints_vector(self):
        if self.current_state == self.states[-1]:
            hints = np.array(self.initials)
        else:
            mu = self.durations[0][self.current_state]
            sigma = self.durations[1][self.current_state]

            duration_dist = scipy.stats.norm(mu, sigma)

            # stay_prob = 1 - duration_dist.cdf(self.current_duration) 
            # hints = self.transitions[self.current_state,:] 
            # hints = hints * (1-stay_prob) 
            # hints[self.current_state] = stay_prob 

            stay_prob = duration_dist.sf(self.current_duration)

            hints = np.array(self.transitions[self.current_state,:])
            if hints.sum() == 0: # gesture 9 
                hints[self.current_state] = 1 # no transition, stay with prob 1
            else:
                hints = hints * (1-stay_prob)
                hints[self.current_state] = stay_prob

        if abs(hints.sum()-1) > 1e-10:
            raise Exception('Irrgular Hints!')

        return hints