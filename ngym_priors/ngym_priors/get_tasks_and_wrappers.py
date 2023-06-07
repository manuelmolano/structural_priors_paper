from neurogym.envs import registration as reg
from reaction_time import ReactionTime
from variable_nch import Variable_nch
from trial_hist_ev import TrialHistoryEvolution
from variable_mapping import VariableMapping
from time_out import TimeOut
from noise import Noise
from inspect import getmembers
from pathlib import Path
import importlib

ALL_WRAPPERS = {'Noise-v0': 'neurogym.wrappers.noise:Noise',
                'TrialHistoryEv-v0':
                    'neurogym.wrappers.trial_hist_ev:TrialHistoryEvolution',
                'VariableMapping-v0':
                    'neurogym.wrappers.trial_hist_ev:VariableMapping',
                'Variable_nch-v0':
                    'neurogym.wrappers.variable_nch:Variable_nch',
                'TimeOut-v0':
                    'neurogym.wrappers.time_out:TimeOut'
                }


NATIVE_ALLOW_LIST = ['NAltPerceptualDecisionMaking', 'NAltConditionalVisuomotor']


def get_envs(foldername=None, env_prefix=None, allow_list=None):
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

    # Root path of neurogym.envs folder
    env_root = Path(__file__).resolve().parent
    lib_root = 'multiple_choice.tasks.'

    # Only take .py files
    files = [p for p in env_root.iterdir() if p.suffix == '.py']
    # Exclude files starting with '_'
    files = [f for f in files if f.name[0] != '_']
    filenames = [f.name[:-3] for f in files]  # remove .py suffix
    filenames = sorted(filenames)

    env_dict = {}
    for filename in filenames:
        # lib = 'neurogym.envs.collections.' + l
        lib = lib_root + filename
        module = importlib.import_module(lib)
        spec = importlib.util.spec_from_file_location(filename,
                                                      foldername+"/params.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, val in getmembers(module):
            if name in allow_list:
                env_dict[env_prefix + name + '-v0'] = lib + ':' + name

    return env_dict

env_dict = get_envs(foldername='/home/molano/multiple_choice/tasks',
                    allow_list=NATIVE_ALLOW_LIST)