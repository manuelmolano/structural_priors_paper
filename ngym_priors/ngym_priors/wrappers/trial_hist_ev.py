#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:02:28 2020

@author: molano
"""

import neurogym as ngym
from neurogym.core import TrialWrapper
import numpy as np
from collections import deque


class TrialHistoryEvolution(TrialWrapper):
    """
    This wrapper imposes specific probability transition matrices that are
    characterized by probs (the probability of the most likely next choice).
    The number of transition matrices is specified by num_contexts. The
    transition matrices are reset with a probability specified by death_prob.

    Parameters
    ----------
    env : neurogym.env
        Environment that will be wrapped
    probs : float, optional
        The probability of the most likely next choice. The default is None.
    ctx_dur : int, optional
        Duration of the contexts (if ctx_ch_prob is None). Default is 200 (trials).
    num_contexts : int, optional
        Number of contexts experienced by each individual. The default is 3.
    death_prob : float, optional
        Probability of death by each individual. The default is 0.0001.
    ctx_ch_prob : float, optional
        Probability of context change. The default is None.
    balanced_probs : boolean, optional
        Indicates whether transtion matrices are balanced. The default is False.

    Raises
    ------
    AttributeError
        DESCRIPTION.

    Returns
    -------
    wrapped environment

    """
    metadata = {
        'description': 'Change ground truth probability based on previous' +
        'outcome.',
        'paper_link': 'https://www.biorxiv.org/content/10.1101/433409v3',
        'paper_name': 'Response outcomes gate the impact of expectations ' +
        'on perceptual decisions'
    }

    def __init__(self, env, probs=None, ctx_dur=200, num_contexts=3,
                 fix_2AFC=False, death_prob=0.0000001, ctx_ch_prob=None,
                 balanced_probs=False, predef_tr_mats=False):
        super().__init__(env)
        try:
            self.n_ch = len(self.unwrapped.choices)  # max num of choices
            self.curr_chs = self.unwrapped.choices
            self.curr_n_ch = self.n_ch
        except AttributeError:
            raise AttributeError('''TrialHistoryEvolution requires task
                                 to have attribute choices''')
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        assert probs is not None, 'Please provide choices probabilities'
        self.fix_2AFC = fix_2AFC  # imposes that chs 1 and 2 will follow 2AFC task
        self.probs = probs
        self.balanced_probs = balanced_probs
        self.num_contexts = num_contexts if not predef_tr_mats else 3
        self.predef_tr_mats = predef_tr_mats
        self.ctx_ch_prob = ctx_ch_prob
        if ctx_ch_prob is None:
            self.death_prob = death_prob*ctx_dur
        else:
            self.death_prob = death_prob/ctx_ch_prob
        self.curr_contexts = self.contexts
        self.curr_tr_mat = self.trans_probs
        assert self.curr_tr_mat.shape[1] == self.n_ch,\
            'The number of choices {:d}'.format(self.tr_mat.shape[1]) +\
            ' inferred from prob mismatchs {:d}'.format(self.n_ch) +\
            ' inferred from choices'
        self.ctx_dur = ctx_dur
        self.prev_trial = self.rng.choice(self.n_ch)  # random initialization

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        block_changed = False
        above_th = True if 'above_perf_th_trh' not in kwargs.keys()\
            else kwargs['above_perf_th_trh']
        # Check if n_ch is passed and if it is different from previous value
        if 'sel_chs' in kwargs.keys() and\
           set(kwargs['sel_chs']) != set(self.curr_chs):
            self.curr_chs = kwargs['sel_chs']
            self.curr_n_ch = len(self.curr_chs)
            self.prev_trial = self.rng.choice(np.arange(self.curr_n_ch))
            self.curr_contexts = self.contexts
            self.curr_tr_mat = self.trans_probs
            block_changed = True
        if above_th and not block_changed:
            # change rep. prob. every self.ctx_dur trials
            if self.ctx_ch_prob is None:
                change_block = self.unwrapped.num_tr % self.ctx_dur == 0
            else:
                change_block = self.unwrapped.rng.rand() < self.ctx_ch_prob
            if change_block:
                self.curr_tr_mat = self.trans_probs
        # get ground truth
        probs_curr_blk = self.curr_tr_mat[self.prev_trial, :]
        ground_truth = self.unwrapped.rng.choice(self.curr_chs, p=probs_curr_blk)
        self.prev_trial = np.where(self.curr_chs == ground_truth)[0][0]
        kwargs.update({'ground_truth': ground_truth,
                       'curr_block': self.curr_block})
        self.env.new_trial(**kwargs)

    @property
    def trans_probs(self):
        '''
        if prob is float it creates the transition matrix
        if prob is already a matrix it normalizes the probabilities and extracts
        the subset corresponding to the current number of choices
        '''
        if self.unwrapped.rng.rand() < self.death_prob:
            # new contexts
            self.curr_contexts = self.contexts
        # select context
        sel_cntxt = self.unwrapped.rng.choice(range(self.contexts.shape[0]))
        # build transition matrix
        context = self.curr_contexts[sel_cntxt]
        tr_mat = np.eye(self.curr_n_ch)*self.probs
        tr_mat[tr_mat == 0] = (1-self.probs)/(self.curr_n_ch-1)
        tr_mat = tr_mat[context, :]
        self.curr_block = sel_cntxt
        # get context id
        blk_id = np.zeros((self.n_ch))-1
        blk_id[np.array(self.curr_chs)] =\
            np.array(self.curr_chs)[np.array(context)]
        self.blk_id = '-'.join([str(int(x)+1) for x in blk_id])
        return tr_mat

    @property
    def contexts(self):
        self.new_generation = True
        num_ch = self.curr_n_ch-2 if self.fix_2AFC else self.curr_n_ch
        contexts = np.empty((self.num_contexts, self.curr_n_ch))
        if self.predef_tr_mats:
            # repeating context
            indx = np.arange(num_ch)
            contexts[0, :] = indx
            # clockwise context
            indx = np.append(np.arange(1, num_ch), 0)
            contexts[1, :] = indx
            # repeating context
            indx = np.insert(np.arange(0, num_ch-1), 0, num_ch-1)
            contexts[2, :] = indx
            contexts = np.unique(contexts, axis=0)
        else:
            for i_ctx in range(self.num_contexts):
                if self.balanced_probs:
                    indx = np.arange(num_ch)
                    self.unwrapped.rng.shuffle(indx)
                else:
                    indx = self.unwrapped.rng.choice(num_ch, size=(num_ch,))
                if self.fix_2AFC:
                    indx = [x+2 for x in indx]
                    indx_2afc = np.arange(2)
                    if i_ctx < self.num_contexts/2:
                        indx_2afc = np.flip(indx_2afc)
                    indx = list(indx_2afc)+indx
                contexts[i_ctx, :] = indx
        return contexts.astype(int)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['curr_block'] = self.blk_id
        info['new_generation'] = self.new_generation
        self.new_generation = False
        return obs, reward, done, info
