#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: molano
"""

# anme of the experiment
experiment = 'vanilla_RNNs'
# general parameters. Seed and alg are input parameters
general_params = {'seed': None, 'alg': None, 'train_mode': 'RL',
                  'task': 'NAltPerceptualDecisionMaking-v0', 'n_lstm': 64,
                  'num_trials': 100000000, 'num_cpu': 20, 'rollout': 15,
                  'run_time': 48, 'num_thrds': 20}

# possible algos: A2C, ACER, ACKTR, PPO2
algs = {'A2C': {}, 'ACER': {}, 'ACKTR': {}, 'PPO2': {'nminibatches': 4}}


# task parameters
task_kwargs = {'NAltPerceptualDecisionMaking-v0': {'n_ch': None, 'ob_nch': False, 'stim_scale': 0.5,
                                                   'zero_irrelevant_stim': True, 'rewards': {'abort': 0.},
                                                   'timing': { 'fixation': ('constant', 100),
                                                              'stimulus': ('truncated_exponential', [150, 100, 300]),
                                                              'decision': ('constant', 100)}}}

# wrappers parameters
wrapps = {'TrialHistoryEv-v0': {'probs': 0.8, 'predef_tr_mats': True,
                                'ctx_ch_prob': 0.0125, 'death_prob': 0.00000001},
          'Variable_nch-v0': {'block_nch': 5000, 'prob_12': 0.01,
                              'sorted_ch': True},
          'PassAction-v0': {},
          'PassReward-v0': {},
          # 'BiasCorrection-v0': {'choice_w': 100},  # used for N=2, N=4, seed=1
          'MonitorExtended-v0': {'folder': '', 'sv_fig': False, 'sv_per': 100000,
                                 'fig_type': 'svg'}}  # XXX: monitor always last

# test parameters, parameters used during the testing of the RNNs 
# Note that retraining is also possible with: 'test_retrain': 'retrain'
test_kwargs = {'/test_2AFC/': {'test_retrain': 'test', 'sv_per': 100000,
                          'num_steps': 5000000, 'sv_values': False,
                          'rerun': False, 
                          'wrappers': {'Variable_nch-v0': {'block_nch': 10**9,
                                                            'prob_12': 1}}}}

