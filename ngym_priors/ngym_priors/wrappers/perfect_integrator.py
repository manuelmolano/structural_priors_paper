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


class PerfectIntegrator(Wrapper):
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
        self.cum_sum = np.zeros((self.n_ch,))
        self.stim_indx = self.env.observation_space.name['stimulus']
        self.obs_sh = self.env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.obs_sh+self.n_ch+1,),
                                            dtype=np.float32)
    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        # get perfect integrator action
        if self.env.t_ind >= int(self.env.start_t['decision']/self.env.dt):
            act_io = np.argmax(self.cum_sum)+1
        else:
            act_io = 0
        # make action
        obs, reward_io, done, info = self.env.step(act_io)
        # if it's the end of trial, store action and reward, and update performance
        if info['new_trial']:
            self.cum_sum = np.zeros((self.n_ch,))
            reward = 1*(action == info['gt'] and action == act_io)
            info['performance'] = reward
            # build observation from previous action and reward
            onehot = np.zeros((self.n_ch,))
            onehot[int(act_io)-1] = 1
            obs_prev_act_rew = np.concatenate((np.zeros((self.obs_sh,)),
                                               onehot, np.array([reward_io])))
        else:
            self.cum_sum += obs[self.stim_indx]
            reward = 0
            obs_prev_act_rew = np.zeros((self.obs_sh+self.n_ch+1,))
        info['act_io'] = act_io
        info['reward_io'] = reward_io
        return obs_prev_act_rew, reward, done, info
