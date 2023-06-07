#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:18:49 2020

@author: molano
"""
from neurogym.core import TrialWrapper
import numpy as np

class VariableMapping(TrialWrapper):
    """Change ground truth probability based on previous outcome.

    Args:
        probs: matrix of probabilities of the current choice conditioned
            on the previous. Shape, num-choices x num-choices
    """
    def __init__(self, env,  mapp_ch_prob=0.003, min_mapp_dur=30,
                 sess_end_prob=0.0025, min_sess_dur=120):
        super().__init__(env)
        try:
            self.n_ch = self.unwrapped.n_ch  # max num of choices
            self.curr_n_stims = self.stims.shape[1]
        except AttributeError:
            raise AttributeError('TrialHistory requires task to '
                                 'have attribute choices')
        self.mapp_ch_prob = mapp_ch_prob
        self.min_mapp_dur = min_mapp_dur
        self.sess_end_prob = sess_end_prob
        self.min_sess_dur = min_sess_dur
        self.mapp_start = 0
        self.sess_start = 0
        self.curr_mapping = np.arange(self.curr_n_stims)
        self.unwrapped.rng.shuffle(self.curr_mapping)
        self.mapping_id = '-'.join([str(int(x)+1) for x in self.curr_mapping])
        self.stims = self.unwrapped.stims

    def new_trial(self, **kwargs):
        block_change = False
        self.sess_end = False
        # change of number of stimuli?
        if 'sel_chs' in kwargs.keys() and len(kwargs['sel_chs']) != self.curr_n_stims:
            self.curr_n_stims = len(kwargs['sel_chs'])
            block_change = True
        else:
            mapp_dur = self.unwrapped.num_tr-self.mapp_start
            block_change = mapp_dur > self.min_mapp_dur and\
                self.unwrapped.rng.rand() < self.mapp_ch_prob
        # end of mapping block?
        if block_change:
            self.curr_mapping = np.arange(self.curr_n_stims)
            self.unwrapped.rng.shuffle(self.curr_mapping)
            self.mapping_id = '-'.join([str(int(x)+1) for x in self.curr_mapping])
            self.mapp_start = self.unwrapped.num_tr
        # end of session?
        sess_dur = self.unwrapped.num_tr-self.sess_start
        if sess_dur > self.min_sess_dur and\
           self.unwrapped.rng.rand() < self.sess_end_prob:
            self.stims = np.array(list(it.product([0, 1],
                                                  repeat=self.n_ch))).T == 1
            self.stims = self.stims[:, np.random.choice(self.stims.shape[1],
                                                        size=self.curr_n_stims,
                                                        replace=False)]
            self.sess_end = True
            self.sess_start = self.unwrapped.num_tr
            self.mapp_start = self.unwrapped.num_tr
        # Choose ground truth and update previous trial info
        kwargs.update({'mapping': self.curr_mapping, 'stims': self.stims})
        return self.env.new_trial(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['mapping'] = self.mapping_id
        info['sess_end'] = self.sess_end
        self.sess_end = False
        return obs, reward, done, info
