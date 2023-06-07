#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Feb  2020

@author: jorgedelpozolerida

"""

import neurogym as ngym
import numpy as np
from neurogym.core import TrialWrapper
import warnings
from collections import deque


class ComputeMeanPerf(TrialWrapper):
    metadata = {
        'description': 'Change number of active choices every ' +
        'block_nch trials. Always less or equal than original number.',
        'paper_link': None,
        'paper_name': None
    }

    def __init__(self, env, perf_th=[0.7], perf_w=100, key=['above_perf_th'],
                 cond_on_coh=False):
        """
        block_nch: duration of each block containing a specific number
        of active choices
        prob_2: probability of having only two active choices per block
        """
        super().__init__(env)
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        self.max_nch = len(self.unwrapped.choices)  # Max number of choices
        self.perf_w = perf_w
        self.perf = deque(maxlen=perf_w)
        self.perf_th = perf_th
        self.key = key
        self.cond_on_coh = cond_on_coh

    def new_trial(self, **kwargs):
        for th, k in zip(self.perf_th, self.key):
            above_perf = np.mean(self.perf) > th\
                if len(self.perf) == self.perf_w else False
            kwargs.update({'above_perf_th_'+k: above_perf})
        self.env.new_trial(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            if not (self.cond_on_coh and info['coh'] != 0):
                self.perf.append(1*info['performance'])
            info['mean_perf'] = np.mean(self.perf)
        return obs, reward, done, info
