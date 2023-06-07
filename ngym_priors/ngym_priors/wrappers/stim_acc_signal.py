#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:02:28 2020

@author: molano
"""

import neurogym as ngym
from gym import Wrapper
import numpy as np
from gym import spaces


class StimAccSignal(Wrapper):
    """

    """
    metadata = {
        'description': 'Change ground truth probability based on previous' +
        'outcome.',
        'paper_link': 'https://www.biorxiv.org/content/10.1101/433409v3',
        'paper_name': 'Response outcomes gate the impact of expectations ' +
        'on perceptual decisions'
    }

    def __init__(self, env):
        super().__init__(env)
        try:
            self.n_ch = len(self.unwrapped.choices)  # max num of choices
        except AttributeError:
            raise AttributeError('task must have attribute choices')
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        self.cum_sum = np.zeros((self.n_ch+1,))

    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        # make action
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            self.cum_sum = np.zeros((self.n_ch+1,))
        else:
            self.cum_sum += obs
        print(obs)
        print(self.cum_sum)
        print('---------')
        return self.cum_sum, reward, done, info
