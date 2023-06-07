# Recurrent networks endowed with structural priors explain suboptimal animal behavior

This is the repository where I plan to put all the code necessary to replicate the results of [Molano-Mazon et al. 2023 Current Biology](https://www.sciencedirect.com/science/article/abs/pii/S0960982222019819). 


Documentation: 
- [Installation](#Installation)
- [Tasks](#Tasks)
- [Wrappers](#Wrappers)
- [Examples](#Examples)
- [Authors](#Authors)

The tasks inherit all functionalities from the original [NeuroGym](https://github.com/neurogym/neurogym) toolkit and thus in corporate its flexibility to modify the tasks. The repository also incorporates several various modifier functions (wrappers) that allow easy configuration of new tasks. 


### Installation

The main dependencies are: 
Python version 3.7	https://www.python.org
Tensorflow version 1.13.1	T	https://www.tensorflow.org/
Stable baselines	https://github.com/hill-a/stable-baselines
Neurogym	https://neurogym.github.io/

You can install the tasks as follows (see also ):

    git clone https://github.com/manuelmolano/structural_priors_paper.git
    cd structural_priors_paper/ngym_priors
    pip install -e .
    

### Tasks
The environment used for the paper is: [nalt_perceptualdecisionmaking](https://github.com/manuelmolano/structural_priors_paper/blob/master/ngym_priors/ngym_priors/envs/nalt_perceptualdecisionmaking.py).

### Wrappers
[Wrappers](https://github.com/manuelmolano/structural_priors_paper/tree/master/ngym_priors/ngym_priors/wrappers) are short scripts that allow introducing modifications the original tasks. 
The main wrappers used are:
TrialHistoryEv: to define the contexts.
Variable_nch: to specify the number of choices.
MonitorExtended: to save the behavioral data
PassAction: to pass the previous action as an input (this wrapper is already present in the original NeuroGym toolbox)
PassReward:  to pass the previous reward as an input (this wrapper is already present in the original NeuroGym toolbox)

### Examples





### Authors
* Contact

    [Manuel Molano](https://github.com/manuelmolano) (manuelmolanomazon@gmail.com).
  
