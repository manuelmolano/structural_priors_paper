import importlib
from inspect import getmembers, isfunction, isclass
from pathlib import Path

import gym


def _get_envs(foldername=None, env_prefix=None, allow_list=None):
    """A helper function to get all environments in a folder.

    Example usage:
        _get_envs(foldername=None, env_prefix=None)
        _get_envs(foldername='contrib', env_prefix='contrib')

    The results still need to be manually cleaned up, so this is just a helper

    Args:
        foldername: str or None. If str, in the form of contrib, etc.
        env_prefix: str or None, if not None, add this prefix to all env ids
        allow_list: list of allowed env name, for manual curation
    """

    if env_prefix is None:
        env_prefix = ''
    else:
        if env_prefix[-1] != '.':
            env_prefix = env_prefix + '.'

    if allow_list is None:
        allow_list = list()

    # Root path of ngym_priors.envs folder
    env_root = Path(__file__).resolve().parent
    lib_root = 'ngym_priors.envs.'
    if foldername is not None:
        env_root = env_root / foldername
        lib_root = lib_root + foldername + '.'

    # Only take .py files
    files = [p for p in env_root.iterdir() if p.suffix == '.py']
    # Exclude files starting with '_'
    files = [f for f in files if f.name[0] != '_']
    filenames = [f.name[:-3] for f in files]  # remove .py suffix
    filenames = sorted(filenames)

    env_dict = {}
    for filename in filenames:
        # lib = 'ngym_priors.envs.collections.' + l
        lib = lib_root + filename
        module = importlib.import_module(lib)
        for name, val in getmembers(module):
            if name in allow_list:
                env_dict[env_prefix + name + '-v0'] = lib + ':' + name

    return env_dict


NATIVE_ALLOW_LIST = ['NAltPerceptualDecisionMaking', 'NAltConditionalVisuomotor']
ALL_ENVS = _get_envs(foldername=None, env_prefix=None,
                            allow_list=NATIVE_ALLOW_LIST)


def all_envs(tag=None, psychopy=False, contrib=False, collections=False):
    """Return a list of all envs in ngym_priors."""
    envs = ALL_ENVS.copy()
    env_list = sorted(list(envs.keys()))
    if tag is None:
        return env_list
    else:
        if not isinstance(tag, str):
            raise ValueError('tag must be str, but got ', type(tag))

        new_env_list = list()
        for env in env_list:
            from_, class_ = envs[env].split(':')
            imported = getattr(__import__(from_, fromlist=[class_]), class_)
            env_tag = imported.metadata.get('tags', [])
            if tag in env_tag:
                new_env_list.append(env)
        return new_env_list


def all_tags():
    return ['confidence', 'context dependent', 'continuous action space', 'delayed response', 'go-no-go',
            'motor', 'multidimensional action space', 'n-alternative', 'perceptual', 'reaction time',
            'steps action space', 'supervised', 'timing', 'two-alternative', 'value-based', 'working memory']


def _distance(s0, s1):
    # Copyright (c) 2018 luozhouyang
    if s0 is None:
        raise TypeError("Argument s0 is NoneType.")
    if s1 is None:
        raise TypeError("Argument s1 is NoneType.")
    if s0 == s1:
        return 0.0
    if len(s0) == 0:
        return len(s1)
    if len(s1) == 0:
        return len(s0)

    v0 = [0] * (len(s1) + 1)
    v1 = [0] * (len(s1) + 1)

    for i in range(len(v0)):
        v0[i] = i

    for i in range(len(s0)):
        v1[0] = i + 1
        for j in range(len(s1)):
            cost = 1
            if s0[i] == s1[j]:
                cost = 0
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len(s1)]


def make(id, **kwargs):
    try:
        gym.make(id, **kwargs)
    except gym.error.UnregisteredEnv:
        all_ids = [env.id for env in gym.envs.registry.all()]
        dists = [_distance(id, env_id) for env_id in all_ids]
        # Python argsort
        sort_inds = sorted(range(len(dists)), key=dists.__getitem__)
        env_guesses = [all_ids[sort_inds[i]] for i in range(5)]
        err_msg = 'No registered env with id: {}.\nDo you mean:\n'.format(id)
        for env_guess in env_guesses:
            err_msg += '    ' + env_guess + '\n'
        raise gym.error.UnregisteredEnv(err_msg)


_all_gym_envs = [env.id for env in gym.envs.registry.all()]


def register(id, **kwargs):
    if id not in _all_gym_envs:
        gym.envs.registration.register(id=id, **kwargs)


for env_id, entry_point in ALL_ENVS.items():
    register(id=env_id, entry_point=entry_point)