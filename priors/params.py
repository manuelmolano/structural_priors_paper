#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:46:44 2020

@author: molano
"""

import numpy as np

explore = {'seed': np.arange(8, 16),
           'alg': ['ACER'],
           'n_ch': [4, 8, 12]}

# other
experiment = 'sims_21'
general_params = {'seed': 8, 'alg': 'ACER', 'train_mode': 'RL',
                  'task': 'NAltPerceptualDecisionMaking-v0', 'n_lstm': 1024,
                  'rollout': 15, 'num_trials': 1000000, 'num_cpu': 20,
                  'run_time':48, 'num_thrds': 20}
#
algs = {'A2C': {}, 'ACER': {}, 'ACKTR': {}, 'PPO2': {'nminibatches': 4}}


# task
task_kwargs = {'NAltPerceptualDecisionMaking-v0': {'n_ch': 16, 'ob_nch': False, 'stim_scale': 0.5,
                                                   'zero_irrelevant_stim': True, 'rewards': {'abort': 0.},
                                                   'timing': { 'fixation': ('constant', 100),
                                                              'stimulus': ('truncated_exponential', [150, 100, 300]),
                                                              'decision': ('constant', 100)}}}

# wrappers
wrapps = {'TrialHistoryEv-v0': {'probs': 0.8, 'predef_tr_mats': True,
                                'ctx_ch_prob': 0.0125, 'death_prob': 0.00000001},
          'Variable_nch-v0': {'block_nch': 5000, 'prob_12': 0.01,
                              'sorted_ch': True},
          'PassAction-v0': {},
          'PassReward-v0': {},
          'MonitorExtended-v0': {'folder': '', 'sv_fig': False, 'sv_per': 100000,
                                 'fig_type': 'svg'}}  # XXX: monitor always last
test_kwargs = {'/test_2AFC/': {'test_retrain': 'test', 'sv_per': 100000,
                               'num_steps': 500,
                               'sv_values': False, 'rerun': False,
                               'wrappers': {'Variable_nch-v0': {'block_nch': 10**9,
                                                                'prob_12': 1}}}}

