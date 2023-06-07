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


class LearnTransMatrix(Wrapper):
    """

    """
    metadata = {
        'description': 'Change ground truth probability based on previous' +
        'outcome.',
        'paper_link': 'https://www.biorxiv.org/content/10.1101/433409v3',
        'paper_name': 'Response outcomes gate the impact of expectations ' +
        'on perceptual decisions'
    }

    def __init__(self, env, lr=0.1, decay=0.01):
        super().__init__(env)
        try:
            self.n_ch = len(self.unwrapped.choices)  # max num of choices
        except AttributeError:
            raise AttributeError('task must have attribute choices')
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        self.trans_mat = np.ones((self.n_ch, self.n_ch))/self.n_ch
        self.prev_choice = -1
        self.lr = lr
        self.decay = decay
        self.obs_sh = self.env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.obs_sh+self.n_ch,),
                                            dtype=np.float32)
    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        # get perfect integrator action
        # make action
        obs, reward, done, info = self.env.step(action)
        if self.prev_choice != -1:
            probs = self.trans_mat[self.prev_choice, :]
        else:
            probs = np.ones((self.n_ch,))/self.n_ch
        assert np.abs(np.sum(probs) - 1) < 0.00001
        obs_prev_act_rew = np.concatenate((obs, probs))
        # if it's the end of trial, store action and reward, and update performance
        if info['new_trial']:
            # build observation from previous action and reward
            if self.prev_choice != -1 and action != 0:
                Qn = self.trans_mat[self.prev_choice, action-1]
                Qn_1 = Qn+self.lr*(1-Qn)  # from Sutton and Barto
                self.trans_mat[self.prev_choice, action-1] = Qn_1
            self.prev_choice = action-1
            # decay
            Qn = self.trans_mat
            Qn_1 = Qn+self.decay*(1-Qn)  # from Sutton and Barto
            self.trans_mat = Qn_1
            # normalize
            self.trans_mat = self.trans_mat/np.sum(self.trans_mat, axis=1)[:, None]
        return obs_prev_act_rew, reward, done, info
