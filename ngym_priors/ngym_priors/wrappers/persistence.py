#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:48:16 2020

@author: manuel
"""
import neurogym as ngym
from neurogym.core import TrialWrapper
import numpy as np


class Persistence(TrialWrapper):
    """
    This wrapper imposes a persistence on the reward side such that it will tend
    to repeat the previous side with probability=prob.

    Parameters
    ----------
    env : neurogym.env
        Environment that will be wrapped
    prob : float, optional
        The probability of repeating the previous choice. The default is None.

    Raises
    ------
    AttributeError
        DESCRIPTION.

    Returns
    -------
    wrapped environment

    """
    metadata = {
        'description': 'Repeats previous reward side with probability=prob',
        'paper_link': '',
        'paper_name': ''
    }

    def __init__(self, env, probs=None):
        super().__init__(env)
        try:
            self.n_ch = len(self.unwrapped.choices)  # max num of choices
            self.curr_chs = self.unwrapped.choices
            self.curr_n_ch = self.n_ch
        except AttributeError:
            raise AttributeError('''SideBias requires task
                                 to have attribute choices''')
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        assert probs is not None, 'Please provide choices probabilities'
        self.probs = probs
        self.prev_trial = self.rng.choice(self.n_ch)  # random initialization
        self.curr_tr_mat = self.trans_probs

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        # Check if n_ch is passed and if it is different from previous value
        if 'sel_chs' in kwargs.keys() and\
           set(kwargs['sel_chs']) != set(self.curr_chs):
            self.curr_chs = kwargs['sel_chs']
            self.curr_n_ch = len(self.curr_chs)
            self.prev_trial = self.rng.choice(np.arange(self.curr_n_ch))
            self.curr_tr_mat = self.trans_probs
        # get ground truth
        probs_curr_blk = self.curr_tr_mat[self.prev_trial, :]
        ground_truth = self.unwrapped.rng.choice(self.curr_chs, p=probs_curr_blk)
        self.prev_trial = np.where(self.curr_chs == ground_truth)[0][0]
        kwargs.update({'ground_truth': ground_truth})
        self.env.new_trial(**kwargs)

    @property
    def trans_probs(self):
        '''
        if prob is float it creates the transition matrix
        if prob is already a matrix it normalizes the probabilities and extracts
        the subset corresponding to the current number of choices
        '''
        # build transition matrix
        # repeating context
        tr_mat = np.eye(self.curr_n_ch)*self.probs
        tr_mat[tr_mat == 0] = (1-self.probs)/(self.curr_n_ch-1)
        # get context id
        return tr_mat

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
