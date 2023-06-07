#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:33:28 2020

@author: martafradera
"""

import gym
import numpy as np
# from astropy.convolution import convolve
import matplotlib.pyplot as plt
import importlib
import sys
import os
import glob
sys.path.append(os.path.expanduser("~/gym"))
sys.path.append(os.path.expanduser("~/stable-baselines"))
sys.path.append(os.path.expanduser("~/neurogym"))
sys.path.append(os.path.expanduser("~/ngym_priors"))
import neurogym
import neurogym.utils.plotting as pl
from neurogym.wrappers import ALL_WRAPPERS
from ngym_priors.wrappers import ALL_WRAPPERS as all_wrpps_p
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

ALL_WRAPPERS.update(all_wrpps_p)


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in set(dict2).intersection(dict1))


def apply_wrapper(env, wrap_string, params):
    wrap_str = ALL_WRAPPERS[wrap_string]
    wrap_module = importlib.import_module(wrap_str.split(":")[0])
    wrap_method = getattr(wrap_module, wrap_str.split(":")[1])
    return wrap_method(env, **params)


def make_env(env_id, rank, seed=0, wrapps={}, **kwargs):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        print(kwargs)
        print(wrapps)
        for wrap in wrapps.keys():
            if not (wrap == 'MonitorExtended-v0' and rank != 0):
                env = apply_wrapper(env, wrap, wrapps[wrap])
        return env
    set_global_seeds(seed)
    return _init


def create_env(task, seed, sv_folder, task_kwargs, wrappers_kwargs, n_thrds=1):
    # task
    if n_thrds == 1:
        env = make_env(env_id=task, rank=0, seed=seed, wrapps=wrappers_kwargs,
                       **task_kwargs)()
    else:
        env = SubprocVecEnv([make_env(env_id=task, rank=i, seed=seed,
                                      wrapps=wrappers_kwargs, **task_kwargs)
                             for i in range(n_thrds)])

    return env


def extend_obs(ob, num_threads):
    sh = ob.shape
    return np.concatenate((ob, np.zeros((num_threads-sh[0], sh[1]))))


def run_env(env, num_steps=200, num_trials=None, def_act=None, model=None,
            num_threads=20, sv_values=False, sv_activity=True):
    ob = env.reset()
    if sv_values:
        observations = [ob]
        contexts = []
        coherence = []
        ob_cum = []
        state_mat = []
        rewards = []
        actions = []
        prev_action = []
        actions_end_of_trial = []
        gt = []
        perf = []
        prev_perf = []
        info_vals = {}
    ob_cum_temp = ob
    if num_trials is not None:
        num_steps = 1e5  # Overwrite num_steps value

    trial_count = 0
    _states = None
    done = False
    prev_act = -1
    prev_p = -1
    for stp in range(int(num_steps)):
        ob = np.reshape(ob, (1, ob.shape[0]))
        done = [done] + [False for _ in range(num_threads-1)]
        action, _states = model.predict(extend_obs(ob, num_threads),
                                        state=_states, mask=done)
        action = action[0]
        ob, rew, done, info = env.step(action)
        if done:
            env.reset()
        if sv_values:
            ob_cum_temp += ob
            ob_cum.append(ob_cum_temp.copy())
            if isinstance(info, (tuple, list)):
                info = info[0]
                ob_aux = ob[0]
                rew = rew[0]
                action = action[0]
            else:
                ob_aux = ob
            if sv_activity:
                state_mat.append(_states[0, :])
            observations.append(ob_aux)
            rewards.append(rew)
            actions.append(action)
            prev_action.append(prev_act)
            prev_perf.append(prev_p)
            contexts.append(info['curr_block'])
            coherence.append(info['coh'])
            if 'gt' in info.keys():
                gt.append(info['gt'])
            else:
                gt.append(0)
            for key in info:
                if key not in info_vals.keys():
                    info_vals[key] = [info[key]]
                else:
                    info_vals[key].append(info[key])
            if info['new_trial']:
                prev_act = action
                prev_p = info['performance']
                actions_end_of_trial.append(action)
                perf.append(info['performance'])
                ob_cum_temp = np.zeros_like(ob_cum_temp)
                trial_count += 1
                if num_trials is not None and trial_count >= num_trials:
                    break
            else:
                actions_end_of_trial.append(-1)
                perf.append(-1)
    env.close()
    print('DONE')
    if sv_values:
        if model is not None and len(state_mat) > 0:
            states = np.array(state_mat)
            # states = states[:, 0, :]
        else:
            states = None
        data = {
            'stimulus': np.array(observations[:-1]),
            'ob_cum': np.array(ob_cum),
            'reward': np.array(rewards),
            'choice': np.array(actions),
            'perf': np.array(perf),
            'prev_choice': np.array(prev_action),
            'prev_perf': np.array(prev_perf),
            'actions_end_of_trial': actions_end_of_trial,
            'gt': gt,
            'states': states,
            'info_vals': info_vals,
        }
        return data
    else:
        return {}


def get_algo(alg):
    if alg == "A2C":
        from stable_baselines import A2C as algo
    elif alg == "ACER":
        from stable_baselines import ACER as algo
    elif alg == "ACKTR":
        from stable_baselines import ACKTR as algo
    elif alg == "PPO2":
        from stable_baselines import PPO2 as algo
    return algo


def get_activity(folder, alg, sv_folder, model_name='model', sv_per=10000,
                 task='NAltPerceptualDecisionMaking-v0', num_steps=1000,
                 test_retrain='test', seed=0, sv_values=False, rerun=False,
                 wrappers=None, sv_activity=True, rmv_wrapps=None, name='',
                 learning_rate=7e-4):
    files = glob.glob(folder+'/*'+model_name+'*')
    if len(files) > 0 and not os.path.exists(sv_folder) or rerun:
        if not os.path.exists(sv_folder):
            os.makedirs(sv_folder)
        sorted_models, last_model = order_by_sufix(files)
        model_name = sorted_models[-1]
        algo = get_algo(alg)
        print('Loading model: ')
        print(folder+'/'+model_name)
        model = algo.load(folder+'/'+model_name, tensorboard_log=None,
                          custom_objects={'verbose': 0,
                                          'learning_rate': learning_rate})
        params = glob.glob(folder+'/*params*')
        assert len(params) == 1, params
        params = np.load(params[0], allow_pickle=1)
        task_kwargs = params['task_kwargs'].item()
        num_threads = params['n_thrds'].item()
        wrappers_kwargs = params['wrappers_kwargs'].item()
        if 'Monitor-v0' in wrappers_kwargs.keys():
            wrappers_kwargs['MonitorExtended-v0'] = wrappers_kwargs['Monitor-v0']
            del wrappers_kwargs['Monitor-v0']
        wrappers_kwargs['MonitorExtended-v0']['folder'] = sv_folder
        wrappers_kwargs['MonitorExtended-v0']['sv_per'] = sv_per
        if wrappers is not None:
            wrappers_kwargs.update(wrappers)
        if rmv_wrapps is not None:
            for wr in rmv_wrapps:
                del wrappers_kwargs[wr]
        if test_retrain == 'retrain':
            vars_ = {'alg': alg, 'seed': seed, 'task_kwargs': task_kwargs,
                     'wrappers_kwargs': wrappers_kwargs, 'num_steps': num_steps,
                     'sv_folder': sv_folder, 'n_thrds': num_threads,
                     'sorted_models': sorted_models, 'last_model': last_model}
            np.savez(sv_folder + '/params.npz', **vars_)

            env = create_env(task, seed, sv_folder, task_kwargs=task_kwargs,
                             n_thrds=num_threads, wrappers_kwargs=wrappers_kwargs)
            model.set_env(env)
            checkpoint_callback = CheckpointCallback(save_freq=5*sv_per,
                                                     save_path=sv_folder,
                                                     name_prefix='model')
            print('Running retrain')
            model.learn(total_timesteps=num_steps,
                        callback=checkpoint_callback)
            model.save(f"{sv_folder}/model_retrain")
            return []
        elif test_retrain == 'test':
            vars_ = {'alg': alg, 'seed': seed, 'task_kwargs': task_kwargs,
                     'wrappers_kwargs': wrappers_kwargs, 'num_steps': num_steps,
                     'sv_folder': sv_folder, 'n_thrds': num_threads,
                     'sorted_models': sorted_models, 'last_model': last_model}
            np.savez(sv_folder + '/params.npz', **vars_)

            env = create_env(task, seed, sv_folder, task_kwargs=task_kwargs,
                             n_thrds=1, wrappers_kwargs=wrappers_kwargs)
            print('Running test')
            data = run_env(env, model=model, num_steps=num_steps,
                           sv_activity=sv_activity, num_threads=num_threads,
                           sv_values=sv_values)
            data['env'] = env
            if sv_values:
                np.savez(sv_folder+'/data'+name+'.npz', **data)
            return data
    return []


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    flag = 'model.zip' in file_list
    file_list = [x for x in file_list if x != 'model.zip']
    sfx = [int(x[x.find('model_')+6:x.rfind('_')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    if flag:
        sorted_list.append('model.zip')
    return sorted_list, np.max(sfx)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        seed = 0
    elif len(sys.argv) == 2:
        seed = int(sys.argv[1])
    plt.close('all')
    test_retrain = ''
    # folder = '/home/molano/Dropbox/project_Barna/FOF_project/' +\
    #     'pretrained_RNNs_N16_final_models/seed_0/'
    # folder = '/home/manuel/priors_analysis/annaK/sims_21/' +\
    #     'alg_ACER_seed_0_n_ch_2_BiasCorr/'
    # folder = '/home/molano/Dropbox/project_Barna/FOF_project/networks/' +\
    #     'pretrained_RNNs_N2_fina_models/'
    # folder = '/home/molano/priors/AnnaKarenina_experiments/sims_21/' +\
    #     'alg_ACER_seed_1_n_ch_16/'
    folder = '/home/molano/priors/AnnaKarenina_experiments/sims_21_longer/' +\
        'alg_ACER_seed_0_n_ch_16/'
    sv_folder = folder + '/' + test_retrain+'/'
    test_params = {'test_retrain': 'test', 'sv_per': int(1e6),
                   'num_steps': int(1e4), 'seed': 3, 'sv_values': True,
                   'rerun': True,
                   'wrappers': {'Variable_nch-v0': {'block_nch': 10**9,
                                                    'prob_12': 1}}}
    sv_folder = folder + '/test_2AFC_activity/'
    if not os.path.exists(sv_folder):
        os.mkdir(sv_folder)
    # data = get_activity(folder=folder, alg='ACER', sv_folder=sv_folder,
    #                     **test_params)
    # data = np.load(sv_folder+'data.npz', allow_pickle=1)
    for inst in range(20):
        test_params['seed'] = inst+1234
        data = get_activity(folder=folder, alg='ACER', sv_folder=sv_folder,
                            name='_'+str(inst), **test_params)
    data = np.load(sv_folder+'data_0.npz', allow_pickle=1)
    f = pl.plot_env_1dbox(ob=data['stimulus'], actions=data['choice'],
                          gt=data['gt'], rewards=data['reward'],
                          performance=data['perf'], states=data['states'],
                          legend=True, ob_traces=['']*19)
