#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import itertools
import neurogym as ngym
from neurogym import spaces

class NAltConditionalVisuomotor(ngym.TrialEnv):
    """N-alternative conditional visuomotor task.

    N-alternative forced choice task in which the subject has
    infer an use a stimulus-action mapping.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        sigma: float, input noise level
        n_ch: Number of choices. (def: 3, int)
    """
    metadata = {
        'description': '''N-alternative forced choice task in which the subject has
        infer an use a stimulus-action mapping.''',
        'paper_link':
            'https://www.sciencedirect.com/science/article/pii/S0896627300806583',
        'paper_name': '''Neural Activity in the Primate Prefrontal Cortex
                         during Associative Learning''',
        'tags': ['conditional', 'n-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=0.1,
                 stim_scale=1., n_ch=10, n_stims=2):

        super().__init__(dt=dt)
        self.n_ch = n_ch
        self.choices = np.arange(n_stims)
        self.stims = np.array(list(itertools.product([0, 1], repeat=n_ch))).T == 1
        self.stims = self.stims[:, np.random.choice(self.stims.shape[1],
                                                    size=n_stims, replace=False)]
        assert isinstance(n_ch, int), 'n_ch must be integer'
        assert isinstance(n_stims, int), 'n_stims must be integer'
        assert n_stims > 1, 'n_stims must be at least 2'
        # The strength of evidence, modulated by stim_scale.
        self.cohs = np.array([100.])*stim_scale
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        self.timing = {
            'fixation': 500,
            'stimulus': 500,
            'delay': 1000,
            'decision': 500}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # Action and observation spaces
        name = {'fixation': 0, 'stimulus': range(1, n_ch + 1)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_ch + 1,),
                                            dtype=np.float32, name=name)
        self.mapping = np.arange(n_stims)
        self.unwrapped.rng.shuffle(self.mapping)
        name = {'fixation': 0, 'choice': range(1, n_stims + 1)}
        self.action_space = spaces.Discrete(n_stims + 1, name=name)

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            ground_truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            ob: observation
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        #  Controling whether ground_truth and/or choices is passed.
        if 'ground_truth' in kwargs.keys():
            ground_truth = kwargs['ground_truth']
        elif 'stims' in kwargs.keys():
            stims_temp = kwargs['stims']
            ground_truth = self.rng.choice(np.arange(kwargs['stims'].shape[1]))
        else:
            stims_temp = self.stims
            ground_truth = self.rng.choice(np.arange(self.stims.shape[1]))
        if 'mapping' in kwargs.keys():
            mapping_temp = kwargs['mapping']
        else:
            mapping_temp = self.mapping
        trial = {
            'ground_truth': ground_truth,
            'coh': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        self.add_ob(1, 'fixation', where='fixation')
        stim = np.ones(self.n_ch) * (1 - trial['coh']/100)/2
        stim[stims_temp[:, trial['ground_truth']]] = (1 + trial['coh']/100)/2
        self.add_ob(stim, 'stimulus', where='stimulus')

        #  Adding noise to stimulus observations
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        self.set_groundtruth(mapping_temp[ground_truth],
                             period='decision', where='choice')

        return trial

    def _step(self, action, **kwargs):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False

        obs = self.ob_now
        gt = self.gt_now

        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
        return obs, reward, False, {'new_trial': new_trial, 'gt': gt,
                                    'coh': self.trial['coh']}


if __name__ == '__main__':
    from neurogym.utils import plotting
    env = NAltConditionalVisuomotor(n_stim=2)
    env.reset()
    plotting.plot_env(env, def_act=1)
