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


class Variable_nch(TrialWrapper):
    metadata = {
        'description': 'Change number of active choices every ' +
        'block_nch trials. Always less or equal than original number.',
        'paper_link': None,
        'paper_name': None
    }

    def __init__(self, env, block_nch=100, blocks_probs=None, sorted_ch=True,
                 prob_12=None):
        """
        block_nch: duration of each block containing a specific number
        of active choices
        prob_2: probability of having only two active choices per block
        """
        super().__init__(env)

        assert isinstance(block_nch, int), 'block_nch must be integer'
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        self.block_nch = block_nch
        self.max_nch = len(self.unwrapped.choices)  # Max number of choices
        self.curr_max_nch = self.max_nch
        self.prob_12 = prob_12 if self.max_nch > 2 else 1
        self.sorted_ch = sorted_ch
        # uniform distr. across choices unless prob(n_ch=2) (prob_2) is specified
        if blocks_probs is not None:
            self.prob = blocks_probs[:self.max_nch-1]
            if np.sum(self.prob) == 0:
                self.prob = [1/(self.max_nch-1)]*(self.max_nch-1)
            else:
                self.prob = self.prob/np.sum(self.prob)
        else:
            self.prob = [1/(self.max_nch-1)]*(self.max_nch-1)
        self.curr_prob = self.prob
        # Initialize selected choices
        self.nch = self.max_nch
        self.get_sel_chs()
        self.block_dur = 0

    def new_trial(self, **kwargs):
        change_block = False
        above_th = True if 'above_perf_th_vnch' not in kwargs.keys()\
            else kwargs['above_perf_th_vnch']
        if 'ground_truth' in kwargs.keys():
            warnings.warn('Variable_nch wrapper will ignore passed ground truth')
            del kwargs['ground_truth']
        if 'phase' in kwargs.keys() and self.curr_max_nch != kwargs['phase']:
            self.curr_max_nch = kwargs['phase']
            self.curr_prob = self.prob[:self.curr_max_nch-1]
            self.curr_prob = self.curr_prob/np.sum(self.curr_prob)
            change_block = True
        self.block_dur += 1
        # We change number of active choices every 'block_nch'.
        if change_block or (above_th and self.block_dur > self.block_nch):
            self.block_dur = 0
            self.get_sel_chs()

        kwargs.update({'sel_chs': self.sel_chs})
        self.env.new_trial(**kwargs)

    def get_sel_chs(self):
        fx_12 = self.prob_12 is not None
        if fx_12 and self.unwrapped.rng.rand() < self.prob_12:
            self.nch = 2
            self.sel_chs = np.arange(self.nch)
        else:
            if self.sorted_ch:
                prb = self.curr_prob[1*fx_12:]
                self.nch = self.rng.choice(range(2+1*fx_12, self.curr_max_nch + 1),
                                           p=prb/np.sum(prb))
                self.sel_chs = np.arange(self.nch)
            else:
                self.nch = self.rng.choice(range(2, self.curr_max_nch + 1),
                                           p=self.curr_prob)
                self.sel_chs = sorted(self.rng.choice(range(self.max_nch),
                                                      self.nch, replace=False))
                while (fx_12 and set(self.sel_chs) == set(np.arange(2))):
                    self.sel_chs = sorted(self.rng.choice(range(self.max_nch),
                                                          self.nch, replace=False))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['nch'] = self.nch
        info['sel_chs'] = '-'.join([str(x+1) for x in self.sel_chs])
        return obs, reward, done, info
