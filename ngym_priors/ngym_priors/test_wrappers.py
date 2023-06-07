"""Test wrappers."""

import numpy as np
import gym
import sys
import os
# from gym import spaces
# from gym.core import Wrapper
import matplotlib.pyplot as plt
from neurogym.wrappers import PassAction
from neurogym.wrappers import PassReward
sys.path.append(os.path.expanduser("~/ngym_priors"))
sys.path.append(os.path.expanduser("~/ngym_priors/wrappers"))
from ngym_priors.wrappers.variable_reaction_time import VariableReactionTime
from ngym_priors.wrappers.variable_nch import Variable_nch
from ngym_priors.wrappers.trial_hist_ev import TrialHistoryEvolution
from ngym_priors.wrappers.variable_mapping import VariableMapping
from ngym_priors.wrappers.time_out import TimeOut
from ngym_priors.wrappers.dynamic_noise import DynamicNoise
from ngym_priors.wrappers.perfect_integrator import PerfectIntegrator
from ngym_priors.wrappers.stim_acc_signal import StimAccSignal
from ngym_priors.wrappers.learn_trans_matrix import LearnTransMatrix
from ngym_priors.wrappers.compute_mean_perf import ComputeMeanPerf
from ngym_priors.wrappers.perf_phases import PerfPhases
from ngym_priors.wrappers.bias_correction import BiasCorrection


def test_passaction(env_name='PerceptualDecisionMaking-v0', num_steps=1000,
                    verbose=True):
    """
    Test pass-action wrapper.

    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and action (False)

    Returns
    -------
    None.

    """
    env = gym.make(env_name)
    env = PassAction(env)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        assert obs[-1] == action, 'Previous action is not part of observation'
        if verbose:
            print(obs)
            print(action)
            print('--------')

        if done:
            env.reset()


def test_passreward(env_name='PerceptualDecisionMaking-v0', num_steps=1000,
                    verbose=False):
    """
    Test pass-reward wrapper.

    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)

    Returns
    -------
    None.

    """
    env = gym.make(env_name)
    env = PassReward(env)
    obs = env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        assert obs[-1] == rew, 'Previous reward is not part of observation'
        if verbose:
            print(obs)
            print(rew)
            print('--------')
        if done:
            env.reset()


def test_biascorrection(env_name='NAltPerceptualDecisionMaking-v0', num_steps=1000,
                        verbose=True):
    """
    Test pass-reward wrapper.

    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 100, 'decision': 100},
                'n_ch': 4}
    env = gym.make(env_name, **env_args)
    env = BiasCorrection(env, choice_w=100)
    obs = env.reset()
    for stp in range(num_steps):
        action = 1  # env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if info['new_trial'] and verbose and info['performance'] == 1:
            print(rew)
            print('--------')
        if done:
            env.reset()


def test_perf_integrator(env='NAltPerceptualDecisionMaking-v0', num_steps=100,
                         verbose=True):
    """
    Test pass-reward wrapper.

    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 100, 'decision': 100},
                'n_ch': 4}
    env = gym.make(env, **env_args)
    env = PerfectIntegrator(env)
    env = PassAction(env)
    env = PassReward(env)
    obs = env.reset()
    if verbose:
        observations = []
        reward = []
        actions = []
        gt = []
        perf_int_act = []
        new_trials = []
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if verbose:
            observations.append(obs)
            actions.append(action)
            reward.append(rew)
            new_trials.append(info['new_trial'])
            gt.append(info['gt'])
            perf_int_act.append(info['act_io'])
        if done:
            env.reset()
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(actions, label='Actions')
        ax[1].plot(gt, '--', label='gt')
        ax[1].plot(perf_int_act, '--', label='act-perf-int')
        ax[1].set_xlim([-.5, len(actions)-0.5])
        ax[1].legend()
        ax[2].plot(reward)
        end_of_trial = np.where(new_trials)[0]
        for a in ax:
            ylim = a.get_ylim()
            for ch in end_of_trial:
                a.plot([ch, ch], ylim, '--c')
        ax[2].set_xlim([-.5, len(actions)-0.5])


def test_learn_trans_matrix(env='NAltPerceptualDecisionMaking-v0', num_steps=100,
                            verbose=True, n_ch=2, th=0.01):
    """
    Test pass-reward wrapper.

    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 300, 'decision': 100},
                'n_ch': n_ch}
    env = gym.make(env, **env_args)
    env = TrialHistoryEvolution(env, probs=0.9, predef_tr_mats=True,
                                num_contexts=1)    
    env = LearnTransMatrix(env)
    env = PassAction(env)
    env = PassReward(env)
    obs = env.reset()
    if verbose:
        observations = []
        reward = []
        actions = []
        gt = []
        new_trials = []
    obs_cum = np.zeros((n_ch,))
    for stp in range(num_steps):
        if (obs_cum - np.mean(obs_cum) > th).any():
            action = np.argmax(obs_cum - np.mean(obs_cum))+1
        else:
            action = 0
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            obs_cum = np.zeros((env_args['n_ch'],))
        else:
            obs_cum += obs[1:n_ch+1]
        if verbose:
            observations.append(obs)
            actions.append(action)
            reward.append(rew)
            new_trials.append(info['new_trial'])
            gt.append(info['gt'])
        if done:
            env.reset()
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(actions, label='Actions')
        ax[1].plot(gt, '--', label='gt')
        ax[1].set_xlim([-.5, len(actions)-0.5])
        ax[1].legend()
        ax[2].plot(reward)
        end_of_trial = np.where(new_trials)[0]
        for a in ax:
            ylim = a.get_ylim()
            for ch in end_of_trial:
                a.plot([ch, ch], ylim, '--c')
        ax[2].set_xlim([-.5, len(actions)-0.5])


def test_stim_acc_signal(env='NAltPerceptualDecisionMaking-v0', num_steps=100,
                         verbose=True):
    """
    Test pass-reward wrapper.

    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 300, 'decision': 100},
                'n_ch': 4}
    env = gym.make(env, **env_args)
    env = StimAccSignal(env)
    env = PassAction(env)
    env = PassReward(env)
    obs = env.reset()
    if verbose:
        observations = []
        reward = []
        actions = []
        gt = []
        new_trials = []
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if verbose:
            observations.append(obs)
            actions.append(action)
            reward.append(rew)
            new_trials.append(info['new_trial'])
            gt.append(info['gt'])
        if done:
            env.reset()
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(actions, label='Actions')
        ax[1].plot(gt, '--', label='gt')
        ax[1].set_xlim([-.5, len(actions)-0.5])
        ax[1].legend()
        ax[2].plot(reward)
        end_of_trial = np.where(new_trials)[0]
        for a in ax:
            ylim = a.get_ylim()
            for ch in end_of_trial:
                a.plot([ch, ch], ylim, '--c')
        ax[2].set_xlim([-.5, len(actions)-0.5])


def test_reactiontime(env_name='PerceptualDecisionMaking-v0', num_steps=10000,
                      urgency=-0.1, ths=[-.5, .5], stim_dur_limit=0,
                      verbose=True):
    """
    Test reaction-time wrapper.

    The reaction-time wrapper allows converting a fix duration task into a reaction
    time task. It also allows addding a fix (negative) quantity (urgency) to force
    the network to respond quickly.
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    urgency : float, optional
        float value added to the reward (-0.1)
    stim_dur_limit : int, optional
        allows setting a minimum duration after which the agent can choose
    verbose : boolean, optional
        whether to print observation and reward (False)
    ths : list, optional
        list containing the threholds to make a decision ([-.5, .5])

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 2000, 'decision': 200}}
    env = gym.make(env_name, **env_args)
    env = VariableReactionTime(env, urgency=urgency, stim_dur_limit=stim_dur_limit)
    env.reset()
    if verbose:
        observations = []
        obs_cum_mat = []
        actions = []
        new_trials = []
        reward = []
        min_stim_mat = []
    obs_cum = 0
    end_of_trial = False
    step = 0
    fix = env_args['timing']['fixation']
    for stp in range(num_steps):
        if obs_cum > ths[1]:
            action = 1
        elif obs_cum < ths[0]:
            action = 2
        else:
            action = 0
        end_of_trial = True if action != 0 else False
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            step = 0
            obs_cum = 0
            end_of_trial = False
            min_stim_mat.append(info['min_stim_dur'])
        else:
            step += 1
            assert not end_of_trial or step <= env.min_stim_dur+fix/env.dt,\
                'Trial still on after making a decision'
            obs_cum += obs[1] - obs[2]
            min_stim_mat.append(-1)
        if verbose:
            observations.append(obs)
            actions.append(action)
            obs_cum_mat.append(obs_cum)
            new_trials.append(info['new_trial'])
            reward.append(rew)
    if verbose:
        min_stim_mat = np.array(min_stim_mat)
        print('Min. stim. durations:')
        print(np.unique(min_stim_mat[min_stim_mat != -1], return_counts=1))
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(actions, label='Actions')
        ax[1].plot(new_trials, '--', label='New trial')
        ax[1].set_xlim([-.5, len(actions)-0.5])
        ax[1].legend()
        ax[2].plot(obs_cum_mat, label='cum. observation')
        ax[2].plot([0, len(obs_cum_mat)], [ths[1], ths[1]], '--', label='upper th')
        ax[2].plot([0, len(obs_cum_mat)], [ths[0], ths[0]], '--', label='lower th')
        ax[2].set_xlim([-.5, len(actions)-0.5])
        ax[3].plot(reward, label='reward')
        ax[3].set_xlim([-.5, len(actions)-0.5])
        ax[4].plot(min_stim_mat, label='stim-dur')
        ax[4].set_xlim([-.5, len(actions)-0.5])


def test_variablemapping(env='NAltConditionalVisuomotor-v0', verbose=True,
                         mapp_ch_prob=0.05, min_mapp_dur=10, def_act=1,
                         num_steps=2000, n_stims=4, n_ch=4, margin=2,
                         sess_end_prob=0.01, min_sess_dur=20):
    """
    Test variable-mapping wrapper.
    TODO: explain wrapper
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'NAltConditionalVisuomotor-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)
    mapp_ch_prob : float, optional
        probability of mapping change (0.1)
    min_mapp_dur : int, optional
         minimum number of trials for a mapping block (3)
    sess_end_prob: float, optional,
        probability of session to finish (0.0025)
    min_sess_dur: int, optional
        minimum number of trials for session (5)
    def_act : int, optional
        default action for the agent, if None an action will be randomly chosen (1)
    n_stims : int, optional
        number of stims (10)
    n_ch : int, optional
        number of channels (4)
    margin : float, optional
        margin allowed when comparing actual and expected mean block durations (2)

    Returns
    -------
    None.

    """
    env_args = {'n_stims': n_stims, 'n_ch': n_ch, 'timing': {'fixation': 100,
                                                             'stimulus': 200,
                                                             'delay': 200,
                                                             'decision': 200}}

    env = gym.make(env, **env_args)
    env = VariableMapping(env, mapp_ch_prob=mapp_ch_prob,
                          min_mapp_dur=min_mapp_dur, sess_end_prob=sess_end_prob,
                          min_sess_dur=min_sess_dur)
    env.reset()
    if verbose:
        observations = []
        reward = []
        actions = []
        gt = []
        new_trials = []
        mapping = []
        new_session = []
    prev_mapp = env.curr_mapping[env.trial['ground_truth']] + 1
    stims = env.stims.flatten()
    for stp in range(num_steps):
        action = def_act or env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            mapping.append(info['mapping'])
            assert (action == prev_mapp and rew == 1.) or action != prev_mapp
            prev_mapp = env.curr_mapping[env.trial['ground_truth']] + 1
            if info['sess_end']:
                new_session.append(1)
                assert (stims != env.stims.flatten()).any()
                stims = env.stims.flatten()
            else:
                new_session.append(0)
                assert (stims == env.stims.flatten()).all()
        if verbose:
            observations.append(obs)
            actions.append(action)
            reward.append(rew)
            new_trials.append(info['new_trial'])
            gt.append(info['gt'])
    mapping = [int(x.replace('-', '')) for x in mapping]
    mapp_ch = np.where(np.diff(mapping) != 0)[0]
    sess_ch = np.where(new_session)[0]
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(actions, label='Actions')
        ax[1].plot(gt, '--', label='gt')
        ax[1].set_xlim([-.5, len(actions)-0.5])
        ax[1].legend()
        ax[2].plot(reward)
        end_of_trial = np.where(new_trials)[0]
        for ch in end_of_trial:
            ax[2].plot([ch, ch], [0, 1], '--c')
        for ch in mapp_ch:
            ax[2].plot([end_of_trial[ch], end_of_trial[ch]], [0, 1], '--k')
        for ch in sess_ch:
            ax[2].plot([end_of_trial[ch], end_of_trial[ch]], [0, 1], '--m')
        ax[2].set_xlim([-.5, len(actions)-0.5])
    sess_durs = np.diff(sess_ch)
    assert (sess_durs > min_sess_dur).all()
    mean_sess_dur = np.mean(sess_durs)
    exp_sess_durs = min_sess_dur+1/sess_end_prob
    assert np.abs(mean_sess_dur-exp_sess_durs) < margin,\
        'Mean sess. dur.: '+str(mean_sess_dur)+', expected: '+str(1/sess_end_prob)
    mapp_blck_durs = np.diff(mapp_ch)
    assert (mapp_blck_durs > min_mapp_dur).all()
    mean_durs = np.mean(mapp_blck_durs)
    exp_durs = min_mapp_dur+1/mapp_ch_prob
    assert np.abs(mean_durs - exp_durs) < margin,\
        'Mean mapp. block durations: '+str(mean_durs)+', expected: '+str(exp_durs)
    sys.exit()


def test_noise(env='PerceptualDecisionMaking-v0', margin=0.01, perf_th=None,
               ev_incr=1., num_steps=1000, std_noise=0.1, verbose=True):
    """
    Test noise wrapper.

    The noise wrapper allows adding noise to the full observation received by the
    network. It also offers the option of fixxing a specific target performance
    that the wrapper will assure by modulating the magnitude of the noise added.
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)
    margin : float, optional
        margin allowed when comparing actual and expected performances (0.01)
    perf_th : float, optional
        target performance for the noise wrapper (0.7)
    ev_incr : float, optional
        factor to increase the evidence as trials passes (1.0)
    std_noise : float, optional
        standard deviation of gaussian noise added to the observation (1.0)

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 5000, 'decision': 200}}
    env = gym.make(env, **env_args)
    env = DynamicNoise(env, perf_th=perf_th, std_noise=std_noise, ev_incr=ev_incr)
    env.reset()
    if verbose:
        observations = []
        obs_cum_mat = []
        actions = []
        new_trials = []
        reward = []
        perf = []
        std_mat = []
    std_noise = 0
    for stp in range(num_steps):
        if np.random.rand() < std_noise:
            action = env.action_space.sample()
        else:
            action = env.gt_now
        obs, rew, done, info = env.step(action)
        if 'std_noise' in info:
            std_noise = info['std_noise']
        if verbose:
            if info['new_trial']:
                perf.append(info['performance'])
                std_mat.append(std_noise)
                obs_cum = 0
            else:
                obs_cum = obs[1] - obs[2]
            observations.append(obs)
            actions.append(action)
            obs_cum_mat.append(obs_cum)
            new_trials.append(info['new_trial'])
            reward.append(rew)
        if done:
            env.reset()
    actual_perf = np.mean(perf[-5000:])
    if verbose:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot([0, len(perf)], [perf_th, perf_th], '--')
        plt.plot(np.convolve(perf, np.ones((100,))/100, mode='valid'))
        plt.subplot(2, 1, 2)
        plt.plot(std_mat)
        _, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
        ax = ax.flatten()
        observations = np.array(observations)
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(actions, label='Actions')
        ax[1].plot(new_trials, '--', label='New trial')
        ax[1].set_xlim([-.5, len(actions)-0.5])
        ax[1].legend()
        ax[2].plot(obs_cum_mat, label='cum. observation')
        ax[2].set_xlim([-.5, len(actions)-0.5])
        ax[3].plot(reward, label='reward')
        ax[3].set_xlim([-.5, len(actions)-0.5])
    if perf_th is not None:
        assert np.abs(actual_perf-perf_th) < margin, 'Actual performance: ' +\
            str(actual_perf)+', expected: '+str(perf_th)


def test_timeout(env='NAltPerceptualDecisionMaking-v0', time_out=500,
                 num_steps=100, verbose=True):
    env_args = {'n_ch': 4,
                'timing': {'fixation': 100, 'stimulus': 200, 'decision': 200}}
    env = gym.make(env, **env_args)
    env = TimeOut(env, time_out=time_out)
    env.reset()
    reward = []
    observations = []
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if verbose:
            reward.append(rew)
            observations.append(obs)
        if done:
            env.reset()
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(reward, '--', label='reward')
        ax[1].set_xlim([-.5, len(reward)-0.5])
        ax[1].legend()


def test_concat_wrpprs_th_vch_pssr_pssa(env_name='NAltPerceptualDecisionMaking-v0',
                                        num_steps=1000, probs=0.8, num_blocks=16,
                                        verbose=True, num_ch=6, variable_nch=True,
                                        th=0.5, env_args={}):
    var_nch_block = 100
    var_nch_perf_th = 0.8
    tr_hist_block = 20
    tr_hist_perf_th = 0.5
    env_args['n_ch'] = num_ch
    env_args['zero_irrelevant_stim'] = True
    env_args['ob_histblock'] = False
    env = gym.make(env_name, **env_args)
    env = TrialHistoryEvolution(env, probs=probs, ctx_ch_prob=0.05,
                                predef_tr_mats=True, balanced_probs=True,
                                num_contexts=num_blocks)
    env = Variable_nch(env, block_nch=var_nch_block, prob_12=0.05, sorted_ch=True)
    env = PerfPhases(env, start_ph=3, step_ph=1, wait=100,
                     flag_key='above_perf_th_vnch')
    env = ComputeMeanPerf(env, perf_th=[var_nch_perf_th, tr_hist_perf_th],
                          perf_w=[var_nch_block, tr_hist_block],
                          key=['vnch', 'trh'],
                          cond_on_coh=[False, True])
    transitions = np.zeros((num_blocks, num_ch, num_ch))
    env = PassReward(env)
    env = PassAction(env)
    env.reset()
    num_tr_blks = np.zeros((num_blocks,))
    blk_id = []
    s_chs = []
    blk = []
    blk_stp = []
    gt = []
    nch = []
    obs_mat = []
    perf_vnch = []
    perf_trh = []
    phase = []
    prev_gt = 1
    obs_cum = np.zeros((num_ch,))
    for stp in range(num_steps):
        if (obs_cum - np.mean(obs_cum) > th).any():
            action = np.argmax(obs_cum - np.mean(obs_cum))+1
        else:
            action = 0
        obs, rew, done, info = env.step(action)
        obs_mat.append(obs)
        blk_stp.append(info['curr_block'])
        if done:
            env.reset()
        if info['new_trial'] and verbose:
            perf_vnch.append(info['mean_perf_'+str(var_nch_perf_th)+'_' +
                             str(var_nch_block)+'_vnch'])
            perf_trh.append(info['mean_perf_'+str(tr_hist_perf_th)+'_' +
                                 str(tr_hist_block)+'_trh'])
            phase.append(info['phase'])
            obs_cum = np.zeros((env_args['n_ch'],))
            # print(info['curr_block'])
            # print('-------------')
            blk.append(info['curr_block'])
            gt.append(info['gt'])
            sel_chs = list(info['sel_chs'].replace('-', ''))
            sel_chs = [int(x)-1 for x in sel_chs]
            blk_id, indx = check_blk_id(blk_id, info['curr_block'], num_blocks,
                                        sel_chs)
            s_chs.append(info['sel_chs'])
            nch.append(info['nch'])
            if len(nch) > 2 and 2*[nch[-1]] == nch[-3:-1] and\
               2*[blk[-1]] == blk[-3:-1] and\
               indx != -1:
                num_tr_blks[indx] += 1
                transitions[indx, prev_gt, info['gt']-1] += 1
                if prev_gt > info['nch'] or info['gt']-1 > info['nch']:
                    pass
            prev_gt = info['gt']-1
        else:
            obs_cum += obs[1:num_ch+1]
    if verbose:
        sel_choices, counts = np.unique(s_chs, return_counts=1)
        print('\nSelected choices and frequencies:')
        print(sel_choices)
        print(counts/np.sum(counts))
        blocks, counts = np.unique(blk, return_counts=1)
        print('\nTransition matrices and frequencies:')
        print(blocks)
        print(counts/np.sum(counts))
        tr_blks, counts = np.unique(np.array(blk)[np.array(s_chs) == '1-2'],
                                    return_counts=1)
        print('\n2AFC task transition matrices and frequencies:')
        print(tr_blks)
        print(counts/np.sum(counts))
        _, ax = plt.subplots(nrows=1, ncols=1)
        obs_mat = np.array(obs_mat)
        ax.imshow(obs_mat.T, aspect='auto')
        _, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        blk_int = [int(x.replace('-', '')) for x in blk]
        ax[0].plot(np.array(blk_int[:20000])/(10**(num_ch-1)), '-+',
                   label='tr-blck')
        ax[0].plot(nch[:20000], '-+', label='num choices')
        ax[0].plot(phase[:20000], '-+', label='phase')
        ax[1].plot(gt[:20000], '-+', label='correct side')
        ax[2].set_xlabel('Trials')
        ax[2].plot(perf_vnch[:20000], '-+',
                   label='performance vnch (w='+str(var_nch_block) +
                   ', th='+str(var_nch_perf_th)+')')
        ax[2].plot(perf_trh[:20000], '-+',
                   label='performance trh (w='+str(tr_hist_block) +
                   ', th='+str(tr_hist_perf_th)+')')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        num_cols_rows = int(np.sqrt(num_blocks))
        _, ax1 = plt.subplots(ncols=num_cols_rows, nrows=num_cols_rows)
        ax1 = ax1.flatten()
        _, ax2 = plt.subplots(ncols=num_cols_rows, nrows=num_cols_rows)
        ax2 = ax2.flatten()
        for ind_blk in range(len(blk_id)):
            norm_counts = transitions[ind_blk, :, :]
            ax1[ind_blk].imshow(norm_counts)
            ax1[ind_blk].set_title(str(blk_id[ind_blk]) +
                                   ' (N='+str(num_tr_blks[ind_blk])+')',
                                   fontsize=6)
            nxt_tr_counts = np.sum(norm_counts, axis=1).reshape((-1, 1))
            norm_counts = norm_counts / nxt_tr_counts
            ax2[ind_blk].imshow(norm_counts)
            ax2[ind_blk].set_title(str(blk_id[ind_blk]) +
                                   ' (N='+str(num_tr_blks[ind_blk])+')',
                                   fontsize=6)
    data = {'transitions': transitions, 'blk': blk, 'blk_id': blk_id, 'gt': gt,
            'nch': nch, 's_ch': s_chs, 'obs_mat': obs_mat, 'blk_stp': blk_stp}
    return data


def check_blk_id(blk_id_mat, curr_blk, num_blk, sel_chs):
    # translate transitions t.i.a. selected choices
    # curr_blk_indx = list(curr_blk.replace('-', ''))
    # curr_blk_indx = [sel_chs[int(x)-1] for x in curr_blk_indx]
    # curr_blk = '-'.join([str(x) for x in curr_blk_indx])
    if curr_blk in blk_id_mat:
        return blk_id_mat, np.argwhere(np.array(blk_id_mat) == curr_blk)
    elif len(blk_id_mat) < num_blk:
        blk_id_mat.append(curr_blk)
        return blk_id_mat, len(blk_id_mat)-1
    else:
        return blk_id_mat, -1


def test_trialhistEv(env_name, num_steps=10000, probs=0.8, num_blocks=2,
                     verbose=True, num_ch=4):
    env = gym.make(env_name, **{'n_ch': num_ch})
    env = TrialHistoryEvolution(env, probs=probs, ctx_dur=200, death_prob=0.001,
                                num_contexts=num_blocks, fix_2AFC=True,
                                balanced_probs=True)
    transitions = []
    env.reset()
    num_tr = 0
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        if info['new_trial'] and verbose:
            num_tr += 1
            # print(info['curr_block'])
            transitions.append(np.array([np.where(x == 0.8)[0][0]
                                         for x in env.curr_tr_mat[0, :, :]]))
        if info['new_generation'] and verbose:
            print('New generation')
            print(num_tr)
    plt.figure()
    plt.imshow(np.array(transitions), aspect='auto')


if __name__ == '__main__':
    plt.close('all')
    env_args = {'stim_scale': 10, 'timing': {'fixation': 100,
                                             'stimulus': 200,
                                             'decision': 200}}
    test_biascorrection()
    sys.exit()
    test_learn_trans_matrix()
    sys.exit()
    data = test_concat_wrpprs_th_vch_pssr_pssa(env_args=env_args)
    test_biascorrection()
    test_learn_trans_matrix()
    test_stim_acc_signal()
    test_perf_integrator()
    # test_identity('Nothing-v0', num_steps=5)

    test_timeout()
    test_reactiontime()
    test_noise()
    sys.exit()
    test_variablemapping()
    sys.exit()
    test_passreward()
    test_passaction()

    # test_trialhistEv('NAltPerceptualDecisionMaking-v0', num_steps=100000,
    #                  probs=0.8, num_blocks=3, verbose=True, num_ch=8)
