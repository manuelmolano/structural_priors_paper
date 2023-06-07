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


class PerfPhases(TrialWrapper):
    metadata = {
        'description': 'Change number of active choices every ' +
        'block_nch trials. Always less or equal than original number.',
        'paper_link': None,
        'paper_name': None
    }

    def __init__(self, env, start_ph=None, end_ph=None, step_ph=None, wait=100,
                 flag_key='above_perf_th'):
        """
        block_nch: duration of each block containing a specific number
        of active choices
        prob_2: probability of having only two active choices per block
        """
        super().__init__(env)

        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        self.end = end_ph or len(self.unwrapped.choices)
        self.step_ph = step_ph or 1
        self.phase = start_ph or 3
        self.wait = wait
        self.counter = 0
        self.flag_key = flag_key

    def new_trial(self, **kwargs):
        if self.flag_key not in kwargs.keys():
            warnings.warn('PerfPhases wrapper expects '+self.flag_key+' variable' +
                          ' from compute_mean_perf wrapper')
        self.counter += 1
        if self.flag_key in kwargs.keys() and kwargs[self.flag_key]:
            # We change number of active choices every 'block_nch'.
            if self.counter >= self.wait:
                self.phase = min(self.phase+self.step_ph, self.end)
                self.counter = 0
        kwargs.update({'phase': self.phase})
        self.env.new_trial(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            info['phase'] = self.phase
        return obs, reward, done, info
