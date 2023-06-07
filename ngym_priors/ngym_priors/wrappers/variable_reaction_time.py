#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:07:21 2019

@author: molano
"""
import gym
import numpy as np


class VariableReactionTime(gym.Wrapper):  # TODO: Make this a trial wrapper instead?
    """Allow reaction time response.

    Modifies a given environment by allowing the network to act at
    any time after the fixation period.
    """
    metadata = {
        'description': 'Modifies a given environment by allowing the network' +
        ' to act at any time after the fixation period.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, urgency=0.0, stim_dur_limit=100):
        super().__init__(env)
        self.env = env
        self.urgency = urgency
        self.stim_dur_limit = max(self.env.dt+1, stim_dur_limit)
        self.tr_dur = 0

    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        dec = 'decision'
        stim = 'stimulus'
        assert stim in self.env.start_t.keys(),\
            'Reaction time wrapper requires a stimulus period'
        assert dec in self.env.start_t.keys(),\
            'Reaction time wrapper requires a decision period'
        if self.env.t_ind == 0:
            # get maximum stim duration
            limit = min(self.stim_dur_limit,
                        self.env.end_t[stim]-self.env.start_t[stim])
            # randomly choose minimum stim duration for current trial
            limit = self.env.rng.choice(np.arange(self.env.dt, limit))
            # get timestep
            self.min_stim_dur = int(limit/self.env.dt)
            # set start of decision period
            self.env.start_t[dec] =\
                self.env.start_t[stim]+self.min_stim_dur*self.env.dt
            # change ground truth accordingly
            self.env.gt[self.env.start_ind[stim]+self.min_stim_dur:
                        self.env.end_ind[stim]] =\
                self.env.gt[self.env.start_ind[dec]]
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            info['min_stim_dur'] = self.min_stim_dur
            info['tr_dur'] = self.tr_dur+1
            obs *= 0
        else:
            self.tr_dur = self.env.t_ind
        reward += self.urgency
        return obs, reward, done, info
