#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:15:15 2020

@author: molano
"""
from ngym_priors.wrappers.variable_reaction_time import VariableReactionTime
from ngym_priors.wrappers.variable_nch import Variable_nch
from ngym_priors.wrappers.trial_hist_ev import TrialHistoryEvolution
from ngym_priors.wrappers.variable_mapping import VariableMapping
from ngym_priors.wrappers.time_out import TimeOut
from ngym_priors.wrappers.dynamic_noise import DynamicNoise
from ngym_priors.wrappers.monitor_extended import MonitorExtended
from ngym_priors.wrappers.catch_trials import CatchTrials
from ngym_priors.wrappers.trial_hist import TrialHistory
from ngym_priors.wrappers.combine import Combine
from ngym_priors.wrappers.identity import Identity
from ngym_priors.wrappers.transfer_learning import TransferLearning
from ngym_priors.wrappers.perfect_integrator import PerfectIntegrator
from ngym_priors.wrappers.stim_acc_signal import StimAccSignal
from ngym_priors.wrappers.learn_trans_matrix import LearnTransMatrix
from ngym_priors.wrappers.compute_mean_perf import ComputeMeanPerf
from ngym_priors.wrappers.perf_phases import PerfPhases
from ngym_priors.wrappers.bias_correction import BiasCorrection
from ngym_priors.wrappers.persistence import Persistence
from ngym_priors.wrappers.pass_gt import PassGT

ALL_WRAPPERS = {'DynamicNoise-v0':
                'ngym_priors.wrappers.dynamic_noise:DynamicNoise',
                'TrialHistoryEv-v0':
                    'ngym_priors.wrappers.trial_hist_ev:TrialHistoryEvolution',
                'VariableMapping-v0':
                    'ngym_priors.wrappers.trial_hist_ev:VariableMapping',
                'Variable_nch-v0':
                    'ngym_priors.wrappers.variable_nch:Variable_nch',
                'TimeOut-v0':
                    'ngym_priors.wrappers.time_out:TimeOut',
                'VariableReactionTime-v0':
                    'ngym_priors.wrappers.variable_reaction_time:VariableReactionTime',
                'MonitorExtended-v0':
                    'ngym_priors.wrappers.monitor_extended:MonitorExtended',
                'CatchTrials-v0': 'ngym_priors.wrappers.catch_trials:CatchTrials',
                'TrialHistory-v0': 'ngym_priors.wrappers.trial_hist:TrialHistory',
                'Combine-v0': 'ngym_priors.wrappers.combine:Combine',
                'Identity-v0': 'ngym_priors.wrappers.identity:Identity',
                'TransferLearning-v0':
                    'ngym_priors.wrappers.transfer_learning:TransferLearning',
                'PerfectIntegrator-v0':
                    'ngym_priors.wrappers.perfect_integrator:PerfectIntegrator',
                'StimAccSignal-v0':
                    'ngym_priors.wrappers.stim_acc_signal:StimAccSignal',
                'LearnTransMatrix-v0':
                    'ngym_priors.wrappers.learn_trans_matrix:LearnTransMatrix',
                'ComputeMeanPerf-v0':
                    'ngym_priors.wrappers.compute_mean_perf:ComputeMeanPerf',
                'PerfPhases-v0':
                    'ngym_priors.wrappers.perf_phases:PerfPhases',
                'BiasCorrection-v0':
                    'ngym_priors.wrappers.bias_correction:BiasCorrection',
                'Persistence-v0':
                    'ngym_priors.wrappers.persistence:Persistence',
                'PassGT-v0':
                    'ngym_priors.wrappers.pass_gt:PassGT'}


def all_wrappers():
    return sorted(list(ALL_WRAPPERS.keys()))
