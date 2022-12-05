#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:19:13 2020

@author: molano
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Affine2D
from matplotlib.patches import Rectangle
import mpl_toolkits.axisartist.floating_axes as floating_axes
import plotting_functions as pf
import helper_functions as hf
import process_rats_data as prd
from copy import deepcopy as deepc
import numpy as np
import matplotlib
import glob
import os
import seaborn as sns
import itertools
from neurogym.utils import plotting as pl
from argparse import Namespace as nspc
matplotlib.rcParams['font.size'] = 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# --- GLOBAL VARIABLES
model_cols = ['evidence',
              'L+1', 'L-1', 'L+2', 'L-2', 'L+3', 'L-3', 'L+4', 'L-4',
              'L+5', 'L-5', 'L+6-10', 'L-6-10',
              'T++1', 'T+-1', 'T-+1', 'T--1', 'T++2', 'T+-2', 'T-+2',
              'T--2', 'T++3', 'T+-3', 'T-+3', 'T--3', 'T++4', 'T+-4',
              'T-+4', 'T--4', 'T++5', 'T+-5', 'T-+5', 'T--5',
              'T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10',
              'ev-T++1', 'ev-T++2', 'ev-T++3', 'ev-T++4', 'ev-T++5', 'ev-T++6-10',
              'intercept']

aftercc_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                   'T+-1', 'T--1', 'T-+1',
                                                   'T+-2', 'T--2']]

afteree_cols = [x for x in model_cols if x not in ['L+1', 'L+2', 'L-2',
                                                   'T+-1', 'T++1', 'T-+1',
                                                   'T++2', 'T-+2',
                                                   'ev-T++1', 'ev-T++2']]

afterec_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                   'T+-1', 'T++1', 'T--1',
                                                   'T++2', 'T-+2',
                                                   'ev-T++1', 'ev-T++2']]

afterce_cols = [x for x in model_cols if x not in ['L+1', 'L+2', 'L-2',
                                                   'T++1', 'T--1', 'T-+1',
                                                   'T+-2', 'T--2',
                                                   'ev-T++1']]

plt.close('all')
MAIN_FOLDER = '/home/molano/priors/AnnaKarenina_experiments/'
SV_FOLDER =\
    '/home/molano/Dropbox/project_Barna/reset_paper/figures/figs_from_python/'

# MAIN_FOLDER = '/home/manuel/priors_analysis/annaK/'
# SV_FOLDER = '/home/manuel/priors_analysis/annaK/figs_paper/'


def sort_list(file_list, init_tag='model_', end_tag='_steps'):
    """
    Sort a file list by the order given by the values between init_tag and end_tag.

    Parameters
    ----------
    file_list : list
        list of files.
    init_tag : str, optional
        initial tag ('model_')
    end_tag : str, optional
        end tag ('_steps')

    Returns
    -------
    sorted_list : list
        sorted list.
    sfx_sorted : list
        values between tags ordered.

    """
    basenames = [os.path.basename(x) for x in file_list]
    sfx = [int(x[x.find(init_tag)+len(init_tag):x.rfind(end_tag)])
           for x in basenames]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    sfx_sorted = sorted(sfx)
    return sorted_list, sfx_sorted


def bin_values(xss, perfs, binning, mean_median='median'):
    """
    Compute mean for values in perfs binned following xss.

    Parameters
    ----------
    xss : list
        binning reference.
    perfs : list
        values to bin and average.
    binning : int
        binning.

    Returns
    -------
    mean : list
        averages.
    std : list
        standard deviations.
    xs : array
        indexes of bins.

    """
    xss = np.array(xss)
    perfs = np.array(perfs)
    xss_b = binning*((xss)/binning).astype(int)
    xs = np.unique(xss_b)
    if mean_median == 'mean':
        mean = [np.nanmean(perfs[xss_b == x]) for x in xs]
    else:
        mean = [np.nanmedian(perfs[xss_b == x]) for x in xs]
    std = [np.nanstd(perfs[xss_b == x])/np.sqrt(np.sum(xss_b == x)) for x in xs]
    xs = xs+binning/2
    return mean, std, xs

# --- FIG. EXPS


def plot_exp_psychoCurves(sv_folder, font=8, **plt_kwargs):
    """
    Plot psycho-curves for original dataset (hermoso-Mendizabal et al.).

    Note that all rats will be put together and the psychometric curves will be
    construct for this "meta-rat".

    Parameters
    ----------
    folder : str, optional
        where to find the data ('/home/molano/priors/rats/data_Ainhoa/Rat*/')
    psy_size : float, optional
        width of panels (height will be adjusted so panels are square) (0.13)
    fig_factor : float, optional
        factor to adjust panels height so panels are square (1)
    top_row : float, optional
        height position of panels (0.85)
    font : float, optional
        fontsize of labels (8)
    **plt_kwargs : dict
        extra plot properties.

    Returns
    -------
    None.

    """
    def plot_fit(probs, ax, color):
        popt, pcov = hf.curve_fit(hf.probit_lapse_rates, xs, probs)
        y = hf.probit_lapse_rates(x, popt[0], popt[1], popt[2], popt[3])
        ax.plot(x, y, color=color)

    f, ax = plt.subplots(ncols=2, figsize=(3, 1.5))
    pf.plot_dashed_lines(ax=ax[0])
    pf.plot_dashed_lines(ax=ax[1])
    x = np.linspace(-1, 1, 20)
    plt_opts = {'linestyle': '', 'Marker': '.'}
    xs = 2*(np.array([0, 0.25, 0.375, 0.500, 0.625, 0.750, 1.000])-0.5)
    probRep = [0.1541, 0.2839, 0.4559, 0.6523, 0.8074, 0.8974, 0.9601]
    stdprobRep = [0.0047, 0.0059, 0.0066, 0.0040, 0.0026, 0.0020, 0.0013]
    probAlt = [0.0659, 0.1403, 0.2438, 0.4109, 0.6027, 0.7500, 0.9089]
    stdprobAlt = [0.0017, 0.0024, 0.0029, 0.0042, 0.0064, 0.0058, 0.0038]
    probRepAE = [0.0991, 0.1969, 0.3322, 0.5280, 0.7062, 0.8278, 0.9144]
    stdprobRepAE = [0.0040, 0.0053, 0.0063, 0.0084, 0.0116, 0.0098, 0.0072]
    probAltAE = [0.1014, 0.2233, 0.3499, 0.5401, 0.7238, 0.8403, 0.9143]
    stdprobAltAE = [0.0077, 0.0103, 0.0117, 0.0080, 0.0058, 0.0047, 0.0036]
    plot_fit(probs=probRep, ax=ax[0], color=hf.azul)
    ax[0].errorbar(xs, probRep, stdprobRep, color=hf.azul, label='Rep. block',
                   **plt_opts)
    plot_fit(probs=probAlt, ax=ax[0], color=hf.rojo)
    ax[0].errorbar(xs, probAlt, stdprobAlt, color=hf.rojo, label='Alt. block',
                   **plt_opts)
    plot_fit(probs=probRepAE, ax=ax[1], color=hf.azul)
    ax[1].errorbar(xs, probRepAE, stdprobRepAE, color=hf.azul, **plt_opts)
    plot_fit(probs=probAltAE, ax=ax[1], color=hf.rojo)
    ax[1].errorbar(xs, probAltAE, stdprobAltAE, color=hf.rojo, **plt_opts)
    # ax[0].legend()
    ax[0].set_title('After correct', fontsize=font)
    ax[1].set_title('After error', fontsize=font)
    ax[0].set_ylabel('Repeating probability', fontsize=font)
    ax[0].set_xlabel('Rep. Stim. Evidence', fontsize=font)
    ax[1].set_xlabel('Rep. Stim. Evidence', fontsize=font)
    ax[0].set_yticks([0, 0.5, 1])
    ax[0].set_yticklabels(['0', '0.5', '1'])
    ax[1].set_yticks([])
    pf.rm_top_right_lines(ax[0])
    pf.rm_top_right_lines(ax[1])
    pf.sv_fig(f=f, name='psychoCurves', sv_folder=sv_folder)


def plot_exp_reset_index(main_folder='/home/molano/priors/rats', top_row=0.15,
                         plt_io=True, font=8, ax_RI_exp=None, ax_kernel_exp=None,
                         ax_2d_plot=None, **exp_sel):
    """
    Plot reset index for experiments and ideal observer.

    Parameters
    ----------
    main_folder : str, optional
        where to find the data ('/home/molano/priors/rats')
    plt_io : bool, optional
        whether to plot reset index for ideal obersver (False)
    top_row : float, optional
        height position of panels (0.85)
    font : float, optional
        fontsize of labels (8)
    ax_RI_exp : ax, optional
        ax to plot the reset index (None)
    ax_kernel_exp : ax, optional
        ax to plot the kernels (None)
    ax_2d_plot : ax, optional
        ax to plot the after correct VS error transition contribution (None)

    exp_sel : dic
        example with all experiments:
            exps_selected = {'exps': ['/80_20/', '/95_5/', '/low_volatility/',
                              '/high_volatility/', '/uncorrelated/',
                              '/silent_80_20/'],
                     'names': ['Freq.', '.8', '.95', 'LV', 'HV',
                               'Unc.', 'Sil.', 'I. Obs.'],
                     'exp_for_krnls': '.8'}

    Returns
    -------
    None.

    """
    exps_selected = {'exps': ['/80_20/', '/95_5/', '/uncorrelated/',
                              '/silent_80_20/', '/silent_95_05/'],
                     'names': ['Freq.', '.8', '.95', '.5', 'S0.8', 'S0.95',
                               'I. Obs.'],
                     'exp_for_krnls': '.8'}

    exps_selected.update(exp_sel)
    if ax_kernel_exp is None:
        ax_kernel_exp = plt.axes((0.77, top_row+0.85, 0.25, 0.6))
    if ax_RI_exp is None:
        ax_RI_exp = plt.axes((0.77, top_row+0.01, 0.24, 0.6))
    exp_for_krnls = exps_selected['exp_for_krnls']
    exps = exps_selected['exps']
    names = exps_selected['names']  # '95-5', 'H. Vol.', 'Silent',
    # '/silent/' '/low_volatility/' '/uncorrelated/' '/95_5/' '/high_volatility/'
    num_exps = np.arange(len(names)-1+1*plt_io)
    exps_selected['xticklabels'] = names  # ['Group '+str(x+1) for x in num_exps]
    f_temp, ax_temp = plt.subplots(nrows=1, ncols=1)
    xticklabels = exps_selected['xticklabels']
    if plt_io:
        xticklabels.append('Reverse')
    ax_kernel_exp.plot([1, 6], [0, 0], '--k', lw=0.5)
    # axs_glm_krnls = [[ax_temp]]
    for i_exp, exp in enumerate(exps):
        folder = main_folder+exp
        ax_krnl = ax_kernel_exp if names[i_exp+1] == exp_for_krnls else ax_temp
        col = None
        prd.results_frm_matlab_glm(main_folder=folder, ax_inset=ax_RI_exp,
                                   x=i_exp+1, ax_tr=[ax_krnl], color=col,
                                   name=names[i_exp+1], plt_ind_trcs=True,
                                   ax_2d_plot=ax_2d_plot)
    if 'Freq.' in names:
        ax_krnl = ax_kernel_exp if exp_for_krnls == 'Freq.' else ax_temp
        folder = '/home/molano/priors/rats/data_Ainhoa/'  # Ainhoa's data
        prd.glm_krnls(main_folder=folder, tag='mat', x=0, ax_inset=ax_RI_exp,
                      axs_glm_krnls=[[ax_krnl]], color=None, name='Freq.',
                      tags_mat=[['T++']], plt_ind_trcs=True)
    # PLOT RESET INDEX FOR IDEAL OBSERVER
    if plt_io:
        plot_io(ax=ax_RI_exp, i_exp=i_exp+2)

    pf.rm_top_right_lines(ax=ax_RI_exp)
    # ax_RI_exp.set_ylim([-0.1, 1.1])
    ax_RI_exp.set_xticks(num_exps)
    ax_RI_exp.set_xticklabels(xticklabels, fontsize=font)
    ax_RI_exp.set_ylabel('Reset Index', fontsize=font)
    ax_RI_exp.axhline(y=0, linestyle='--', color='k', lw=0.5)
    ax_RI_exp.axhline(y=1, linestyle='--', color='k', lw=0.5)
    # ax_kernel_exp.legend()
    pf.rm_top_right_lines(ax=ax_kernel_exp)
    ax_kernel_exp.set_ylabel('GLM weight', fontsize=font)
    ax_kernel_exp.set_xlabel('Trial lag', fontsize=font)
    pf.xtcks_krnls(xs=[6], ax=ax_kernel_exp)
    ax_kernel_exp.invert_xaxis()
    return ax_RI_exp, ax_kernel_exp


def plot_io(ax, i_exp):
    """
    Plot reset index in ax for ideal obersver.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    i_exp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    folder_io = MAIN_FOLDER+'/optimal_observer/'
    file = folder_io + '/data_ACER_optimal_.npz'
    data = np.load(file, allow_pickle=1)
    vals = np.array([str(float(v)) for v in data['val_mat']])
    glm_ac_cond = data['glm_mats_ac'][vals == '2.0']
    glm_ae_cond = data['glm_mats_ae'][vals == '2.0']
    a_glm_ac_cond = hf.get_average(glm_ac_cond)
    a_glm_ae_cond = hf.get_average(glm_ae_cond)
    regressors = ['T++']
    id_ob_plt_opts = {'color': (.5, .5, .5), 'markersize': 3, 'marker': 'o'}
    f_temp, ax_temp = plt.subplots()
    pf.plot_kernels(a_glm_ac_cond, a_glm_ae_cond, ax=[ax_temp],
                    ax_inset=ax, regressors=regressors,
                    inset_xs=i_exp, **id_ob_plt_opts)
    plt.close(f_temp)


def fig_exps(figsize=(8.5, 2), plt_io=False, plot_exp_psych=False, **plt_kwargs):
    """
    Plot figure 1 (experiments) for cosyne abstract.

    Parameters
    ----------
    figsize : tuple, optional
        size of figure ((6, 8))
    plt_io: bool, optional
        whether to plot reset index for ideal observer (False)
    **plt_kwargs : dict
        dict containing info for plotting.
        plt_opts = {'lw': 1., 'alpha': , 'colors': , 'font': , 'font_inset': }
        it can also contain any key accepted by plt.plot

    Returns
    -------
    None.

    """
    plt_opts = {'lw': 1., 'alpha': 1., 'colors': 0.5+np.zeros((20, 3)), 'font': 9,
                'font_inset': 7}
    plt_opts.update(plt_kwargs)
    plt_op = nspc(**plt_opts)
    del plt_opts['font_inset'], plt_opts['colors'], plt_opts['font']
    f_glm_tr, ax_tr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax_tr.remove()
    fig_factor = figsize[0]/figsize[1]
    top_row = 0.0
    # PLOT PSYCHOMETRIC CURVES FOR EXPERIMENTS
    if plot_exp_psych:
        plot_exp_psychoCurves(fig_factor=fig_factor, top_row=top_row,
                              font=plt_op.font, **plt_opts)
    # PLOT RESET INDEX FOR DIFFERENT EXPERIMENTS
    ax_RI_exp, ax_kernel_exp = plot_exp_reset_index(top_row=top_row, plt_io=plt_io,
                                                    font=plt_op.font)
    # ax_RI_exp.set_xlim([-0.5, 1.5])
    f_glm_tr.savefig(SV_FOLDER+'/fig_1_from_python.svg', dpi=400,
                     bbox_inches='tight')
    f_glm_tr.savefig(SV_FOLDER+'/fig_1_from_python.png', dpi=400,
                     bbox_inches='tight')
    f_glm_tr.savefig(SV_FOLDER+'/fig_1_from_python.pdf', dpi=400,
                     bbox_inches='tight')

# --- FIG. N2


def filter_mat(mat, mean_prf, val_mat, sel_vals, perf_th):
    """
    Filter array t.i.a. performance in mean_prf and values in sel_val.

    Parameters
    ----------
    mat : array
        array to filter.
    mean_prf : array
        performance associated with each element in mat.
    val_mat : array
        value associate with each element in mat..
    sel_vals : list
        values to select.
    perf_th : float
        minimum performance to select.

    Returns
    -------
    mat_filtered : TYPE
        DESCRIPTION.

    """
    mat_filtered = np.array([b for b, p, v in zip(mat, mean_prf, val_mat)
                             if p >= perf_th and v in sel_vals])
    return mat_filtered


def plot_perf_insts(perf_mat, offsets, max_offset, spacing, ax, color,
                    range_=None, **plt_opts):
    """
    Plot performances in perf_mat.

    Parameters
    ----------
    perf_mat : array
        performances to plot.
    offsets : list
        list with offsets to subtract from x axis.
    max_offset : str
        maximum offset (in case offset is None).
    spacing : int
        spacing used to subsample performances.
    ax : axis
        axis where to plot.
    color: tuple
        color of traces.
    range_ : list, optional
        interval to plot the traces (None)
    **plt_opts : dict
        plotting properties.

    Returns
    -------
    xss : list
        list containing the xs values for each experiment.
    perfs : list
        list containing performance trace for each experiment.

    """
    perfs = []
    xss = []
    for ind_b, p_mat in enumerate(perf_mat):
        if len(p_mat) != 0:
            offset = offsets[ind_b] if offsets[ind_b] is not None else max_offset
            xs = np.arange(p_mat.shape[0])*spacing-offset
            lmt_start = 0 if range_ is None else range_[0]
            lmt_end = p_mat.shape[0] if range_ is None else range_[1]
            ax.plot(xs[lmt_start:lmt_end], p_mat[lmt_start:lmt_end],
                    color=color, **plt_opts)
            xss = xss+xs[lmt_start:lmt_end].tolist()
            perfs = perfs+p_mat[lmt_start:lmt_end].tolist()
    return xss, perfs


def plot_nets_perf(main_folder, sel_vals=['2'], perf_th=0.6, plt_zoom=True,
                   prf_panel=[0.37, 0.44, 0.61, 0.3], spacing=10000, ax_perfs=None,
                   align=False, y_lim=[0.7, 0.85], binning=100000, x_margin=100000,
                   plot_pi=True, plot_mean=True, **plt_kwargs):
    """
    Plot performance of RNNs.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    spacing : int, optional
        spacing used to subsample performances vectors (10000)
    prf_panel : list, optional
        position and size of main performance panel ([0.37, 0.44, 0.61, 0.3])
    psy_size : float, optional
        width of panels (height will be adjusted so panels are square) (0.16)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    plt_opts = {'lw': 1., 'alpha': 1., 'colors': 0.5+np.zeros((20, 3)), 'font': 8,
                'font_inset': 7}
    plt_opts.update(plt_kwargs)
    colors = plt_opts['colors']
    font = plt_opts['font']
    del plt_opts['font_inset'], plt_opts['colors'], plt_opts['font']
    # perf panel
    if ax_perfs is None:
        ax_perfs = plt.axes(prf_panel)
    # file = main_folder + '/data_n_ch_2_ctx_ch_prob_0.0125__.npz'
    # file = main_folder + '/data_n_ch_2_ctx_ch_prob_0.0125__per_100K.npz'
    file = main_folder + '/data_ACER__.npz'
    data = np.load(file, allow_pickle=1)
    # filter experiments by performance and max number of choices used for training
    perf_mat_all_exps = np.array([g for g, v in zip(data['perf_mats'],
                                                    data['val_mat'])
                                  if v in sel_vals and len(g) > 0])

    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    perf_mat_cond = filter_mat(mat=data['perf_mats'], mean_prf=mean_prf,
                               perf_th=perf_th, val_mat=data['val_mat'],
                               sel_vals=sel_vals)
    # get perfect integrator performance (perf_mats are computed w.r.t. p.i.)
    perf_pi = filter_mat(mat=data['perf_pi'], mean_prf=mean_prf, perf_th=perf_th,
                         val_mat=data['val_mat'], sel_vals=sel_vals)
    # perf_mat_cond = np.array([p+pi for p, pi in zip(perf_mat_cond, perf_pi)])
    # align perf. to the momment at which the network starts using the trans. hist.
    if align:
        offsets = filter_mat(mat=data['aha_mmt'], mean_prf=mean_prf,
                             perf_th=perf_th, val_mat=data['val_mat'],
                             sel_vals=sel_vals)
    else:
        offsets = np.zeros_like(perf_mat_cond)
    max_offset = np.max([o for o in offsets if o is not None])
    print('Percentage of networks with above perfect integrator performance')
    print(100*perf_mat_cond.shape[0]/perf_mat_all_exps.shape[0])
    plt_opts_insts = deepc(plt_opts)
    plt_opts_insts['alpha'] = plt_opts['alpha']*0.2
    plt_opts_insts['lw'] = plt_opts['lw']*0.5
    xss, perfs =\
        plot_perf_insts(perf_mat=perf_mat_cond, offsets=offsets, ax=ax_perfs,
                        max_offset=max_offset, spacing=spacing, color=colors[0, :],
                        **plt_opts_insts)  # range_=[0, 150],
    plt_opts['lw'] = 1.5*plt_opts['lw']
    max_x = np.max(xss)
    if plot_mean:
        mean, _, xs = bin_values(xss, perfs, binning, mean_median='mean')
        ax_perfs.plot(xs, mean, color=colors[0, :], **plt_opts)
    if plot_pi:
        ax_perfs.plot(ax_perfs.get_xlim(), [perf_pi[0], perf_pi[0]], '--k', lw=0.5)
    ax_perfs.set_xlim([-max_offset-x_margin, max_x+x_margin])
    ax_perfs.set_ylim(y_lim)
    ylim = ax_perfs.get_ylim()
    ax_perfs.set_yticks(np.round([ylim[0], (ylim[1]+ylim[0])/2, ylim[1]], 1))
    ax_perfs.set_ylabel('Accuracy', fontsize=font)
    ax_perfs.set_xlabel('Trials (M)', fontsize=font)
    pf.rm_top_right_lines(ax=ax_perfs)
    if align:
        ax_perfs.plot([0, 0], ax_perfs.get_ylim(), '--', lw=0.5,
                      color=(.7, .7, .7))
    else:
        ax_perfs.set_xticks(np.array([0, 1e6, 2e6, 3e6]))  # +[exp_x])
        ax_perfs.set_xticklabels(['0', '1', '2', '3'])

    # inset with zoom
    if plt_zoom:
        zoom_panel = prf_panel
        zoom_panel[1] = 0.1
        ax_perfs_zoom = plt.axes(zoom_panel)
        plt_opts_insts['alpha'] = 1
        plot_perf_insts(perf_mat=perf_mat_cond, offsets=offsets, ax=ax_perfs_zoom,
                        max_offset=max_offset, spacing=spacing, color=colors[0, :],
                        range_=[0, 70], **plt_opts_insts)
        # plot performances for RNNs trained with no rep/alt blocks
        plot_nets_perf(main_folder=MAIN_FOLDER+'sims_21_2AFC_noblocks/',
                       sel_vals=['2'], perf_th=0.6, plt_zoom=False,
                       ax_perfs=ax_perfs_zoom, plot_pi=False, plot_mean=False,
                       **{'colors': np.zeros((20, 3))+np.array((0., 0., 1)),
                          'alpha': 2})  # alpha is multiplied by 0.2 by default

        ax_perfs_zoom.axhline(y=perf_pi[0], linestyle='--', color='k', lw=0.5)
        ax_perfs_zoom.axhline(y=0.775, linestyle='--', color='c', lw=0.5)
        ylim = ax_perfs_zoom.get_ylim()
        ax_perfs_zoom.set_yticks(np.round([ylim[0], ylim[1]], 1))
        ax_perfs_zoom.set_ylabel('Accuracy', fontsize=font)
        ax_perfs_zoom.set_xlabel('Trials (M)', fontsize=font)
        ax_perfs_zoom.set_xticks([1e5, 5e5])
        ax_perfs_zoom.set_xticklabels(['0.1', '0.5'])
        ax_perfs_zoom.set_xlim([0, 58e4])
        pf.rm_top_right_lines(ax=ax_perfs_zoom)
    return ax_perfs


def plot_N_cond_perf_insts(perfs_cond, ncols=1, plot=False, ax=None,
                           n_mat=None, perc_tr=1, pr_p='1'):
    """
    Plot performances condition on N.

    This code is a simplified version of the one that can be found in
    helper_functions.process_data, where it is possible to separate performances
    also per block.

    Parameters
    ----------
    perfs_cond : dict
        dictionary containing performances.
    ncols : int, optional
        number of columns for figure (1)
    ax : axis, optional
        axis to plot (None)
    **plt_opts : dict
        plotting options.

    Returns
    -------
    None.

    """
    perfs_cond_all = {}
    for ind_b, p_cnd in enumerate(perfs_cond):
        if len(p_cnd) != 0:
            try:  # code modified to plot fig. 4 a-b, t.i.a. previous outcome
                n_ch = np.max([int(p[p.find('-')+1:p.rfind('-')])
                               for p in p_cnd.keys()])
            except ValueError as m:
                print(m)
                n_ch = np.max([int(p[:p.find('-')]) for p in p_cnd.keys()])
            if ax is None:
                f, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(4, 3))
            colors = sns.color_palette("mako", n_colors=n_ch-1)
            ns = range(2, n_ch+1)  # if n_mat is None else n_mat
            for n in ns:  # [2, 4, 8, 16]:
                starts = ['1', '2', str(n)] if n > 2 else ['1', '2']
                if str(n) not in perfs_cond_all.keys():
                    perfs_cond_all[str(n)] = {'perf': [], 'xss': []}
                for i_s, strt in enumerate(starts):
                    try:  # code modified to plot fig. 4 a-b, t.i.a. previous outc
                        key = 'pp_'+pr_p+'-'+str(n)+'-'+strt
                        perfs_cond_all[str(n)]['perf'] +=\
                            p_cnd[key]['m_perf'].tolist()
                        perfs_cond_all[str(n)]['xss'] += p_cnd[key]['indx']
                        if plot:
                            ax.plot(perc_tr*p_cnd[key]['indx'],
                                    p_cnd[key]['m_perf'], '.', color=colors[n-2],
                                    label=str(n), alpha=0.5, markersize=1)
                    except KeyError as m:
                        print(m)
                        key = str(n)+'-'+strt
                        perfs_cond_all[str(n)]['perf'] +=\
                            p_cnd[key]['m_perf'].tolist()
                        perfs_cond_all[str(n)]['xss'] += p_cnd[key]['indx']
                        if plot:
                            ax.plot(perc_tr*p_cnd[key]['indx'],
                                    p_cnd[key]['m_perf'], '.', color=colors[n-2],
                                    label=str(n), alpha=0.5, markersize=1)
    return perfs_cond_all


def plot_nets_N_cond_perf(main_folder, sel_vals=['2'], binning=300000, file=None,
                          prf_panel=[0.37, 0.44, 0.61, 0.3], ax_perfs=None,
                          n_mat=None, perc_tr=1, plt_clrbar=True, perf_th=-100,
                          plot_chance_ref=True, alpha=1., pr_p='0', ax_clbr=None,
                          **plt_kwargs):
    """
    Plot performance of RNNs conditioned on N.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    fig_factor : float, optional
        factor to adjust panels height so panels are square (1)
    spacing : int, optional
        spacing used to subsample performances vectors (10000)
    prf_panel : list, optional
        position and size of main performance panel ([0.37, 0.44, 0.61, 0.3])
    conv_w : int, optional
        trials back to use to define the rep/alt contexts (3)
    psy_size : float, optional
        width of panels (height will be adjusted so panels are square) (0.16)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    plt_opts = {'lw': 1., 'alpha': 1., 'font': 8, 'font_inset': 7}
    plt_opts.update(plt_kwargs)
    font = plt_opts['font']
    del plt_opts['font_inset'], plt_opts['font']
    # perf panel
    if ax_perfs is None:
        ax_perfs = plt.axes(prf_panel)
    # file = main_folder + '/data_n_ch_2_ctx_ch_prob_0.0125__.npz'
    # file = main_folder + '/data_n_ch_2_ctx_ch_prob_0.0125__per_100K.npz'
    if file is None:
        file = main_folder +\
            '/data_ACER*n_ch_16__perf_cond_on_prev_outc_prev_tr.npz'
    data = np.load(file, allow_pickle=1)
    # filter experiments by performance and max number of choices used for training
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    perfs_cond = filter_mat(mat=data['perfs_cond'], mean_prf=mean_prf,
                            perf_th=perf_th, val_mat=data['val_mat'],
                            sel_vals=sel_vals)
    # align perf. to the momment at which the network starts using the trans. hist.
    perfs_cond_all = plot_N_cond_perf_insts(perfs_cond=perfs_cond, ax=ax_perfs,
                                            n_mat=n_mat, perc_tr=perc_tr,
                                            pr_p=pr_p)

    n_ch = np.max([int(p) for p in perfs_cond_all.keys()])
    colors = sns.color_palette("mako", n_colors=n_ch-1)
    ns = range(2, n_ch+1)  # , 2) if n_mat is None else n_mat
    for n in ns:  # range(2, n_ch+1, 2):
        xss = perfs_cond_all[str(n)]['xss']
        perfs = np.array(perfs_cond_all[str(n)]['perf'])+1/n
        mean, std, xs = bin_values(xss, perfs, binning)
        ax_perfs.errorbar(perc_tr*xs, mean, std, color=colors[n-2], lw=1,
                          label=str(n), alpha=alpha)
        if plot_chance_ref:
            ax_perfs.axhline(y=1/n, color=colors[n-2], linestyle='--', lw=0.5)

    ylim = ax_perfs.get_ylim()
    ax_perfs.set_yticks(np.round([ylim[0], (ylim[1]+ylim[0])/2, ylim[1]], 1))
    ax_perfs.set_ylabel('Accuracy', fontsize=font)
    ax_perfs.set_xlabel('Number of trials (x $10^6$)', fontsize=font)
    pf.rm_top_right_lines(ax=ax_perfs)
    ax_perfs.set_xticks(np.array([0, 1e6, 2e6]))  # +[exp_x])
    ax_perfs.set_xticklabels(['0', '1', '2'])
    ax_perfs.set_yticks(np.array([0, 0.3, 0.6]))  # +[exp_x])
    ax_perfs.set_yticklabels(['0', '0.3', '0.6'])
    # construct cmap
    if plt_clrbar:
        my_cmap =\
            ListedColormap(sns.color_palette("mako", n_colors=n_ch-1).as_hex())
        if ax_clbr is None:
            clbr_panel = prf_panel.copy()
            clbr_panel[0] += prf_panel[2]
            clbr_panel[2] = prf_panel[2]/20
            clbr_panel[3] = prf_panel[3]/1.5
            ax_clbr = plt.axes(clbr_panel)
        ax_clbr.imshow(np.linspace(1, n_ch, n_ch)[:, None], origin='lower',
                       cmap=my_cmap, aspect='auto')
        ax_clbr.set_yticks([0, n_ch/2-1, n_ch-1])
        ax_clbr.set_yticklabels(['2', '8', '16'])
        ax_clbr.set_xticks([])
        ax_clbr.set_title('N', fontsize=6)
        ax_clbr.yaxis.tick_right()
        ax_clbr.tick_params(labelsize=6)


def plot_blk_cond_perf_insts(perfs_cond, binning, ncols=1, sel_n=8, plot=False,
                             ax=None, starts=['1', '2']):
    """
    Plot performances condition on block.

    This code is a simplified version of the one that can be found in
    helper_functions.process_data, where it is possible to separate performances
    also per block.

    Parameters
    ----------
    perfs_cond : dict
        dictionary containing performances.
    ncols : int, optional
        number of columns for figure (1)
    ax : axis, optional
        axis to plot (None)
    **plt_opts : dict
        plotting options.

    Returns
    -------
    None.

    """
    perfs_cond_all = {}
    for ind_b, p_cnd in enumerate(perfs_cond):
        if len(p_cnd) != 0:
            # n_ch = np.max([int(p[:p.find('-')]) for p in p_cnd.keys()])
            if ax is None:
                f, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(4, 3))
            for i_s, strt in enumerate(starts):
                if strt not in perfs_cond_all.keys():
                    perfs_cond_all[strt] = {'perf': [], 'xss': []}
                for n in [sel_n]:  # range(3, n_ch+1):
                    perfs_cond_all[strt]['perf'] +=\
                        p_cnd[str(n)+'-'+strt]['m_perf'].tolist()
                    perfs_cond_all[strt]['xss'] += p_cnd[str(n)+'-'+strt]['indx']
            xss = p_cnd[str(sel_n)+'-'+starts[0]]['indx']
            perfs = p_cnd[str(sel_n)+'-'+starts[0]]['m_perf']
            mean_1, std_1, xs_1 = bin_values(xss, perfs, binning)
            xss = p_cnd[str(sel_n)+'-'+starts[1]]['indx']
            perfs = p_cnd[str(sel_n)+'-'+starts[1]]['m_perf']
            mean_2, std_2, xs_2 = bin_values(xss, perfs, binning)
            _, indx_1, indx_2 = np.intersect1d(xs_1, xs_2, return_indices=1)
            assert (xs_1[indx_1] == xs_2[indx_2]).all()
            mean_1 = np.array(mean_1)
            mean_2 = np.array(mean_2)
            ax.plot(mean_1[indx_1], mean_2[indx_2], '-', color='k', lw=0.25,
                    alpha=0.2)

    return perfs_cond_all


def plot_nets_blck_cond_perf(main_folder, sel_vals=['2'], binning=300000,
                             sel_ns=[8], prf_panel=[0.37, 0.44, 0.61, 0.3],
                             ax_perfs=None, **plt_kwargs):
    """
    Plot performance of RNNs conditioned on block.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    fig_factor : float, optional
        factor to adjust panels height so panels are square (1)
    spacing : int, optional
        spacing used to subsample performances vectors (10000)
    prf_panel : list, optional
        position and size of main performance panel ([0.37, 0.44, 0.61, 0.3])
    conv_w : int, optional
        trials back to use to define the rep/alt contexts (3)
    psy_size : float, optional
        width of panels (height will be adjusted so panels are square) (0.16)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    plt_opts = {'lw': 1., 'alpha': 1., 'colors': 0.5+np.zeros((20, 3)), 'font': 8,
                'font_inset': 7}
    plt_opts.update(plt_kwargs)
    # colors = plt_opts['colors']
    font = plt_opts['font']
    del plt_opts['font_inset'], plt_opts['colors'], plt_opts['font']
    # perf panel
    if ax_perfs is None:
        ax_perfs = plt.axes(prf_panel)
    # file = main_folder + '/data_n_ch_2_ctx_ch_prob_0.0125__.npz'
    # file = main_folder + '/data_n_ch_2_ctx_ch_prob_0.0125__per_100K.npz'
    file = main_folder + '/data_ACER__.npz'
    data = np.load(file, allow_pickle=1)
    # filter experiments by performance and max number of choices used for training
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    perfs_cond = filter_mat(mat=data['perfs_cond'], mean_prf=mean_prf,
                            perf_th=-100, val_mat=data['val_mat'],
                            sel_vals=sel_vals)
    # align perf. to the momment at which the network starts using the trans. hist.
    for n in sel_ns:
        print(n)
        starts = ['1', '2']
        perfs_cond_all = plot_blk_cond_perf_insts(perfs_cond=perfs_cond, sel_n=n,
                                                  binning=binning, ax=ax_perfs,
                                                  starts=starts)
        xss = perfs_cond_all[starts[0]]['xss']
        perfs = perfs_cond_all[starts[0]]['perf']
        mean_1, std_1, xs_1 = bin_values(xss, perfs, binning)
        xss = perfs_cond_all[starts[1]]['xss']
        perfs = perfs_cond_all[starts[1]]['perf']
        mean_2, std_2, xs_2 = bin_values(xss, perfs, binning)
        assert (xs_1 == xs_2).all()
        colors = sns.color_palette("rocket_r", n_colors=len(mean_1))
        n_stps = len(mean_1)
        for i_m in range(1, n_stps):
            ax_perfs.errorbar(x=mean_1[i_m-1:i_m+1], y=mean_2[i_m-1:i_m+1],
                              xerr=std_1[i_m-1:i_m+1], yerr=std_2[i_m-1:i_m+1],
                              color=colors[i_m], lw=1)
    ax_perfs.plot(ax_perfs.get_xlim(), ax_perfs.get_xlim(), '--k', lw=0.5)
    ax_perfs.set_ylabel('Accuracy CW block', fontsize=font)
    ax_perfs.set_xlabel('Accuracy Rep. block', fontsize=font)
    ax_perfs.set_xticks(np.array([0, 0.2, 0.4]))
    ax_perfs.set_xticklabels(['0', '0.2', '0.4'])
    ax_perfs.set_yticks(np.array([0, 0.2, 0.4]))
    ax_perfs.set_yticklabels(['0', '0.2', '0.4'])

    pf.rm_top_right_lines(ax=ax_perfs)
    my_cmap = ListedColormap(sns.color_palette("rocket_r",
                                               n_colors=n_stps).as_hex())
    clbr_panel = prf_panel.copy()
    clbr_panel[0] += prf_panel[2]
    clbr_panel[2] = prf_panel[2]/20
    clbr_panel[3] = prf_panel[3]/1.5
    ax_clbr = plt.axes(clbr_panel)
    ax_clbr.imshow(np.linspace(1, n_stps, n_stps)[:, None], origin='lower',
                   cmap=my_cmap, aspect='auto')
    ax_clbr.set_yticks([0, n_stps-1])
    ax_clbr.set_yticklabels([str(binning/1e6), str(binning*n_stps/1e6)])
    ax_clbr.tick_params(labelsize=6)
    ax_clbr.set_title('x$10^6$', fontsize=6)
    ax_clbr.set_xticks([])
    ax_clbr.yaxis.tick_right()


def test_diff_Ns():
    """
    Test plot_nets_blck_cond_perf with different Ns.

    Returns
    -------
    None.

    """
    exp = 'sims_21'
    f, ax = plt.subplots(nrows=3, ncols=5, figsize=(10, 8))
    ax = ax.flatten()
    main_folder = MAIN_FOLDER+'/'+exp+'/'
    for n in range(2, 17):
        plot_nets_blck_cond_perf(main_folder, sel_vals=['16'], binning=300000,
                                 prf_panel=[0.37, 0.44, 0.61, 0.3],
                                 ax_perfs=ax[n-2], sel_ns=[n])
        ax[n-2].set_title('N = '+str(n))
        if n != 12:
            ax[n-2].set_ylabel('')
            ax[n-2].set_xlabel('')
    pf.sv_fig(f=f, name='perf_blck_diff_ns', sv_folder=SV_FOLDER)


def plot_psycho_curves(main_folder, folder_examples='alg_ACER_seed_0_n_ch_2',
                       psych_panel=[0.37, 0.44, 0.3, 0.3], font_inset=7, ax_p=None,
                       fldrs_psychCurves=['200000000', '16000000', '8000000'],
                       spacing=10000, prev_tr_cong=None, step_v=None, step_h=-1):
    """
    Plot psychometric traces conditioned on context and previous outcome.

    Parameters
    ----------
    main_folder : str
        main folder.
    folder_examples : str, optional
        experiment folder ('alg_ACER_seed_0_n_ch_2')
    psych_panel : list, optional
        position and size of panel ([0.37, 0.44, 0.3, 0.3])
    font_inset : float, optional
        labels font (7)
    ax_p: axis, optinal
        axes to plot asterisks corresponding to the periods used to compute
        psycho-curves (None)

    Returns
    -------
    None.

    """
    if ax_p is not None:
        num_stps_trial = 3.3333*20  # num steps includes num threads (20 here)
        file = main_folder + '/data_ACER__.npz'
        data = np.load(file, allow_pickle=1)
        f_indx = np.array([f.find(folder_examples) != -1 for f in data['files']])
        perf = data['perf_mats'][f_indx][0]
        for ev in fldrs_psychCurves:
            idx = min(perf.shape[0]-1, int(float(ev)/(spacing*num_stps_trial)))
            ax_p.plot(idx*spacing, perf[idx], '*', color='k', markersize=4)
    if step_v is None:
        step_v = psych_panel[2]/2
    pnl_height = psych_panel[3]
    pnl_width = psych_panel[2]
    step_v = step_v if step_v > 0 else -pnl_height
    step_h = step_h if step_h > 0 else -pnl_width
    axs_2AFC = [plt.axes(psych_panel)]
    for i_a in range(1, len(fldrs_psychCurves)):
        psych_panel[0] += pnl_width+step_h
        psych_panel[1] += pnl_height+step_v
        axs_2AFC = np.insert(axs_2AFC, 0, plt.axes(psych_panel))
    # plot psychometric curves (late middle early)
    lbs = ['after error', 'after correct']
    colors = [hf.azul, hf.rojo]
    for ax, fld in zip(axs_2AFC, fldrs_psychCurves):
        plt.sca(ax)
        folder = main_folder+folder_examples+'/test_2AFC_all/_model_'+fld+'_steps'
        if not os.path.exists(folder+'/bhvr_data_all.npz'):
            pl.put_together_files(folder)
        data = hf.load_behavioral_data(folder+'/bhvr_data_all.npz')
        ch = data['choice']
        sig_ev = data['putative_ev']
        prf = data['performance']
        # transition blocks
        tr_block = data['tr_block']
        prev_perf = np.concatenate((np.array([0]), prf[:-1]))
        if prev_tr_cong is not None:
            reps = hf.get_repetitions(ch)
            prev_tr = np.concatenate((np.array([0]), reps[:-1]))
        for i_b, blk in enumerate(np.unique(tr_block)):
            if prev_tr_cong is not None:
                prev_tr_mask = prev_tr == prev_tr_cong
            else:
                prev_tr_mask = np.ones_like(prev_perf) == 1
            for ip, p in enumerate([0, 1]):
                alpha = 0.5 if p == 0 else 1
                lnstyl = '-'  # '--' if p == 0 else '-'
                plt_opts = {'color': colors[i_b], 'alpha': alpha, 'lw': 1,
                            'linestyle': lnstyl, 'markersize': 2}
                mask = hf.and_.reduce((prev_perf == p, tr_block == blk,
                                       prev_tr_mask))
                popt, pcov, ev_mask, repeat_mask =\
                    hf.bias_psychometric(choice=ch.copy(), ev=sig_ev.copy(),
                                         mask=mask, maxfev=100000)
                ev_mask = np.round(ev_mask, 2)  # this is to avoid rounding diffs
                hf.plot_psycho_curve(ev=ev_mask, choice=repeat_mask, popt=popt,
                                     ax=ax, label=lbs[ip], **plt_opts)
                ax.plot([-35, 35], [.5, .5], '--', lw=0.2, color=(.5, .5, .5))

        ax.legend().set_visible(False)
        ax.set_xticks([])
        ax.set_xlim([-35, 35])
        ax.set_title('')
        pf.rm_top_right_lines(ax=ax)
    # axs_2AFC[1].set_ylabel('Prob. of repeat', fontsize=font_inset+1)
    ax.set_xticks([-35, 0, 35])
    ax.set_xticklabels(['-1', '0', '1'])
    ax.set_xlabel('Rep. stim. evidence', fontsize=font_inset)
    return axs_2AFC


def plot_ri_across_training(main_folder, reset_panel=None, ax=None, sel_val='2',
                            perf_th=0.6, height=0.2, align=False, binning=100000,
                            margin=200000, ax_contrs=None, contr_th=0.1, num_ws=1,
                            plot_ind_traces=False, plot_ind_dots=True,
                            evs=None, ax_wgts_acr_tr=None, xtcks_ws=None,
                            **plt_kwargs):
    """
    Plot reset index (RI) for networks.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    sel_val : str, optional
        max. num. of choices for which to plot the RI ('2')
    prf_panel : list, optional
        position of perf. panel for reference ([0.37, 0.44])
    perf_th : float, optional
        threshold to filter networks by performance
        (note that perf. is w.r.t perfect integrator) (0.)
    height: float, optional
        height of reset panel (0.2)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    xtcks = ['T++'+x for x in ['2', '3', '4', '5', '6-10']]
    if xtcks_ws is None:
        xtcks_ws = ['T++1', 'T+-1'] + ['T++'+str(x) for x in range(2, num_ws+1)]
    num_stps_trial = 3.3333*20  # num of steps includes number of threads (20 here)
    n_stps_ws = 1
    plt_opts = {'lw': 1., 'alpha': 1., 'font': 8, 'marker': '.'}
    plt_opts.update(plt_kwargs)
    font = plt_opts['font']
    del plt_opts['font']
    if ax is None:
        ax_RI_acr_tr = plt.axes(reset_panel)
    else:
        ax_RI_acr_tr = ax
    if ax_contrs is not None:
        ax_contrs = plt.axes(ax_contrs)
        ac_c_mat = []
        ae_c_mat = []
    if ax_wgts_acr_tr is not None:
        krnls_ac = []
        krnls_ae = []
        plt_opts_ws = {'color': 'k', 'edgecolor': 'none'}

    file = main_folder + '/data_ACER__.npz'
    data_training = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data_training['perf_mats']]
    # get all offsets
    offsets = filter_mat(mat=data_training['aha_mmt'], mean_prf=mean_prf,
                         perf_th=-100, val_mat=data_training['val_mat'],
                         sel_vals=[sel_val])
    if not align:
        offsets = np.zeros_like(offsets)
    max_offset = np.max([o for o in offsets if o is not None])
    fs_training = filter_mat(mat=data_training['files'], mean_prf=mean_prf,
                             perf_th=-100, val_mat=data_training['val_mat'],
                             sel_vals=[sel_val])
    assert len(fs_training) == len(offsets)
    files = glob.glob(main_folder + '/*_model_*.npz')
    files, xs = sort_list(files)
    xs = np.array(xs)/num_stps_trial
    ris = []
    xss = []  # only stores xs vals when contribution is larger than a set th
    xss_all = []  # stores all xs values
    ri_traces = {k: {'trace': [], 'files': [], 'xss': []} for k in fs_training}
    for i_f, f in enumerate(files):
        print('xxxxxxxxxxxxxx')
        print(f)
        data = np.load(f, allow_pickle=1)
        mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
        # transform vals to float and then back to str
        perf_mat = data['perf_mats']
        # filter experiments by performance
        mean_prf = [np.mean(p[-10:]) for p in perf_mat]
        glm_ac = filter_mat(mat=data['glm_mats_ac'], mean_prf=mean_prf,
                            perf_th=perf_th, val_mat=data['val_mat'],
                            sel_vals=[sel_val])
        glm_ae = filter_mat(mat=data['glm_mats_ae'], mean_prf=mean_prf,
                            perf_th=perf_th, val_mat=data['val_mat'],
                            sel_vals=[sel_val])
        fs_testing = filter_mat(mat=data['files'], mean_prf=mean_prf,
                                perf_th=perf_th, val_mat=data['val_mat'],
                                sel_vals=[sel_val])
        f_temp, ax_temp = plt.subplots()
        # filter experiments by performance
        plt_opts['alpha'] = 0.2
        plt_opts['edgecolor'] = 'none'
        for ind_glm, glm_ac_tmp in enumerate(glm_ac):
            if len(glm_ac_tmp) != 0:
                # this assumes that all files in fs_testing are also in fs_training
                f_tst = fs_testing[ind_glm]
                offset = offsets[np.where(fs_training == f_tst)[0]]
                glm_ae_tmp = glm_ae[ind_glm]
                ws_ac = np.nanmean(glm_ac_tmp[-n_stps_ws:, :, :], axis=0)
                ws_ac = np.expand_dims(ws_ac, 0)
                ws_ae = np.nanmean(glm_ae_tmp[-n_stps_ws:, :, :], axis=0)
                ws_ae = np.expand_dims(ws_ae, 0)
                reset, krnl_ac, krnl_ae =\
                    pf.compute_reset_index(ws_ac, ws_ae, xtcks=xtcks,
                                           full_reset_index=False)
                contr_ac = np.abs(np.mean(krnl_ac))
                contr_ae = np.abs(np.mean(krnl_ae))
                _, krnl_ac, krnl_ae =\
                    pf.compute_reset_index(ws_ac, ws_ae, xtcks=xtcks_ws,
                                           full_reset_index=False)

                if contr_ac+contr_ae > contr_th:
                    if reset < -10 and False:
                        regressors = ['T++']
                        _, ax_temp = plt.subplots()
                        pf.plot_kernels(ws_ac, ws_ae, ax=[ax_temp],
                                        regressors=regressors)
                        ax_temp.set_title(str(reset)+' '+str(contr_ac)+' ' +
                                          str(contr_ae)+'  '+f)
                    ri_traces[f_tst]['trace'].append(reset)
                    ri_traces[f_tst]['files'].append(f)
                    ri_traces[f_tst]['xss'].append(xs[i_f]-offset)
                    if plot_ind_dots:
                        ax_RI_acr_tr.scatter(xs[i_f]-offset[0], reset, **plt_opts)
                    ris.append(reset)
                    xss.append(xs[i_f]-offset[0])
                xss_all.append(xs[i_f]-offset[0])
                if ax_contrs is not None:
                    plt_opts['color'] = hf.naranja
                    ax_contrs.scatter(xs[i_f]-offset, contr_ac, **plt_opts)
                    plt_opts['color'] = 'k'
                    ax_contrs.scatter(xs[i_f]-offset, contr_ae,  **plt_opts)
                    ac_c_mat.append(contr_ac)
                    ae_c_mat.append(contr_ae)
                if ax_wgts_acr_tr is not None:
                    # for i_w in range(num_ws):
                    #     plt_opts_ws['alpha'] = 1/(i_w+1)
                    #     plt_opts_ws['color'] = hf.naranja
                    #     ax_wgts_acr_tr.scatter(xs[i_f]-offset[0],
                    #                            krnl_ac[0, i_w],  **plt_opts_ws)
                    #     plt_opts_ws['color'] = 'k'
                    #     ax_wgts_acr_tr.scatter(xs[i_f]-offset[0],
                    #                            krnl_ae[0, i_w], **plt_opts_ws)

                    krnls_ac.append(krnl_ac[0, 0:num_ws])
                    krnls_ae.append(krnl_ae[0, 0:num_ws])

                if evs is not None:
                    if hf.get_tag('model', f) in evs['evs'] and\
                       os.path.basename(f_tst) == evs['exp']:
                        ax_RI_acr_tr.plot(xs[i_f]-offset, reset, '*', zorder=100,
                                          color='k', markersize=4)
        plt.close(f_temp)
    ri_max = {}
    for k in ri_traces.keys():
        if len(ri_traces[k]['trace']) > 0:
            ri_max[k] = {}
            ri_max[k]['f'] =\
                ri_traces[k]['files'][np.argmax(ri_traces[k]['trace'])]
            ri_max[k]['ri'] = np.max(ri_traces[k]['trace'])
            if plot_ind_traces:
                ax_RI_acr_tr.plot(ri_traces[k]['xss'], ri_traces[k]['trace'],
                                  '--', color=(.7, .7, .7), lw=0.5)
    # np.savez(SV_FOLDER+'/ri_max_'+sel_val+name+'.npz', **ri_max)
    max_x = np.max(xss)
    xss = np.array(xss)
    ris = np.array(ris)
    mean_, sem_, xs_ = bin_values(xss, ris, binning)
    plt_opts['alpha'] = 1
    plt_opts['color'] = (.4, .4, .4)
    del plt_opts['edgecolor']
    ax_RI_acr_tr.errorbar(xs_, mean_, sem_, zorder=10, **plt_opts)
    ax_RI_acr_tr.set_yticks([0, 1])
    ax_RI_acr_tr.set_ylim([-0.15, 1.05])
    ax_RI_acr_tr.set_xlim([-max_offset-margin, max_x+margin])
    if not align:
        ax_RI_acr_tr.set_xticks(np.array([0, 1e6, 2e6]))  # +[exp_x])
        ax_RI_acr_tr.set_xticklabels(['0', '1', '2'])
    else:
        ax_RI_acr_tr.set_xticks([0, 1.5e6])
        ax_RI_acr_tr.set_xticklabels(['0', '1.5'])
        # ax_RI_acr_tr.plot([0, 0], ax_RI_acr_tr.get_ylim(), '--c', lw=0.5)

    pf.rm_top_right_lines(ax=ax_RI_acr_tr)
    ax_RI_acr_tr.set_ylabel('Reset Index', fontsize=font)
    ax_RI_acr_tr.set_xlabel('Trial index (x $10^6$)', fontsize=font)
    xss_all = np.array(xss_all)
    if ax_contrs is not None:
        ac_c_mat = np.array(ac_c_mat)
        ae_c_mat = np.array(ae_c_mat)
        xp = np.linspace(np.min(xss_all), np.max(xss_all), 2)
        z = np.polyfit(xss_all, ac_c_mat, 1)
        p = np.poly1d(z)
        ax_contrs.plot(xp, p(xp), color=hf.naranja, lw=1)
        z = np.polyfit(xss_all, ae_c_mat, 1)
        p = np.poly1d(z)
        ax_contrs.plot(xp, p(xp), color='k', lw=1)
        pf.rm_top_right_lines(ax=ax_contrs)
        ax_contrs.set_ylabel('Trans. history contr.', fontsize=font)
        ax_contrs.set_xlabel('Trial index (x $10^6$)', fontsize=font)
        ax_contrs.set_xticks(np.array([0, 0.2, 0.4]))  # +[exp_x])
        ax_contrs.set_xticklabels(['0', '0.2', '0.4'])
        if not align:
            ax_contrs.set_xticks(np.array([0, 1e6, 2e6, 3e6]))  # +[exp_x])
            ax_contrs.set_xticklabels(['0', '1', '2', '3'])
        else:
            ax_contrs.set_xticks([0, 1.5e6])
            ax_contrs.set_xticklabels(['0', '1.5'])
    if ax_wgts_acr_tr is not None:
        # mean_, sem_, xs_ = bin_values(xss, ris, binning)
        # plt_opts['alpha'] = 0.5
        # ax_wgts_acr_tr.errorbar(xs_, mean_, sem_, zorder=10, **plt_opts)
        del plt_opts_ws['edgecolor']
        plt_opts_ws['lw'] = 1
        krnls_ac = np.array(krnls_ac)
        krnls_ae = np.array(krnls_ae)
        for i_wgt in range(num_ws):
            mean_, sem_, xs_ = bin_values(xss_all, krnls_ac[:, i_wgt], binning)
            plt_opts_ws['alpha'] = 1/(i_wgt+1)
            plt_opts_ws['color'] = hf.naranja
            ax_wgts_acr_tr.errorbar(xs_, mean_, sem_, zorder=10, **plt_opts_ws)
            mean_, sem_, xs_ = bin_values(xss_all, krnls_ae[:, i_wgt], binning)
            plt_opts_ws['color'] = 'k'
            ax_wgts_acr_tr.errorbar(xs_, mean_, sem_, zorder=10, **plt_opts_ws)


def plot_perf_across_training(main_folder, reset_panel=None, ax=None, sel_val='2',
                              binning=100000, plot_ind_dots=False,  margin=200000,
                              plot_ind_traces=False, perc_tr=1, perf_th=0.6,
                              offset=0, sel_xs=None, separate_rep_alt=True,
                              color=hf.azul, **plt_kwargs):
    """
    Plot reset index (RI) for networks.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    sel_val : str, optional
        max. num. of choices for which to plot the RI ('2')
    prf_panel : list, optional
        position of perf. panel for reference ([0.37, 0.44])
    perf_th : float, optional
        threshold to filter networks by performance
        (note that perf. is w.r.t perfect integrator) (0.)
    height: float, optional
        height of reset panel (0.2)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    num_stps_trial = 3.3333*20  # num of steps includes number of threads (20 here)
    plt_opts = {'lw': 1., 'alpha': 1., 'font': 8, 'marker': '.'}
    plt_opts.update(plt_kwargs)
    del plt_opts['font']
    if ax is None:
        ax_RI_acr_tr = plt.axes(reset_panel)
    else:
        ax_RI_acr_tr = ax
    # get all offsets
    files = glob.glob(main_folder + '/*_model_*.npz')
    files, xs = sort_list(files)
    xs = np.array(xs)/num_stps_trial
    pmm_rep = []
    if separate_rep_alt:
        pmm_alt = []
    xss = []  # stores xs vals
    for i_f, f in enumerate(files):
        data = np.load(f, allow_pickle=1)
        mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
        # transform vals to float and then back to str
        perf_mat = data['perf_mats']
        # filter experiments by performance
        mean_prf = [np.mean(p[-10:]) for p in perf_mat]
        perf = filter_mat(mat=data['perfs_cond'], mean_prf=mean_prf,
                          perf_th=perf_th, val_mat=data['val_mat'],
                          sel_vals=[sel_val])
        # filter experiments by performance
        # plt_opts['alpha'] = 0.2
        # plt_opts['edgecolor'] = 'none'
        for i_p, perf_tmp in enumerate(perf):
            if len(perf_tmp) != 0:
                # this assumes that all files in fs_testing are also in fs_training
                mean_rep = np.mean(perf_tmp['2-1']['m_perf'])
                mean_alt = np.mean(perf_tmp['2-2']['m_perf'])
                mean_rep = mean_rep if separate_rep_alt else (mean_rep+mean_alt)/2
                pmm_rep.append(mean_rep)
                if separate_rep_alt:
                    pmm_alt.append(mean_alt)
                xss.append(xs[i_f])
                # if plot_ind_dots:
                #     plt_opts['color'] = hf.azul
                #     ax_RI_acr_tr.scatter(xs[i_f]+np.random.randn()*0.001,
                #                          mean_rep, **plt_opts)
                #     if separate_rep_alt:
                #        plt_opts['color'] = hf.rojo
                #        ax_RI_acr_tr.scatter(xs[i_f]+np.random.randn()*0.001,
                #                             mean_alt, **plt_opts)

    xss = np.array(xss)
    pmm_rep = np.array(pmm_rep)
    mean_rep, sem_rep, xs_ = bin_values(xss, pmm_rep, binning)
    mean_rep = np.array(mean_rep)
    sem_rep = np.array(sem_rep)
    if separate_rep_alt:
        pmm_alt = np.array(pmm_alt)
        mean_alt, sem_alt, xs_ = bin_values(xss, pmm_alt, binning)
        mean_alt = np.array(mean_alt)
        sem_alt = np.array(sem_alt)
    xs_ = perc_tr*xs_
    if sel_xs is not None:
        indx = np.array([np.min(np.abs(x-sel_xs)) < 0.01 for x in xs_])
    # plt_opts['alpha'] = 1
    # del plt_opts['edgecolor']
    plt_opts['color'] = color
    ax_RI_acr_tr.errorbar(xs_[indx], mean_rep[indx], sem_rep[indx], zorder=10,
                          linestyle='-', **plt_opts)
    if separate_rep_alt:
        plt_opts['color'] = hf.rojo
        ax_RI_acr_tr.errorbar(xs_[indx], mean_alt[indx], sem_alt[indx], zorder=10,
                              linestyle='-', **plt_opts)

    # ax_RI_acr_tr.set_yticks([0, 1])
    # ax_RI_acr_tr.set_ylim([-0.15, 1.05])
    # ax_RI_acr_tr.set_xticks(np.array([0, 1e6, 2e6]))  # +[exp_x])
    # ax_RI_acr_tr.set_xticklabels(['0', '1', '2'])


def plot_kernels(file,  sel_vals=['2'], perf_th=0.6, regressors=['T++'],
                 ax_kernels=None, kernel_panel=[.05, .05, .37, .44],
                 sel_files=None, plot_mean=True, ax_krnls_ind_tr=None,
                 **plt_kwargs):
    """
    Plot reset index (RI) for networks.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    sel_vals : list, optional
        max. num. of choices for which to plot the RI (['2', '4', '8', '16'])
    kernel_panel : list, optional
        position of perf. panel for reference ([0.37, 0.44])
    perf_th : float, optional
        threshold to filter networks by performance
        (note that perf. is w.r.t perfect integrator) (0.)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    plt_opts = {'lw': 1., 'alpha': 1., 'font': 8}
    plt_opts.update(plt_kwargs)
    font = plt_opts['font']
    del plt_opts['font']
    if ax_kernels is None:
        ax_kernels = plt.axes(kernel_panel)
        ax_kernels.plot([1, 6], [0, 0], '--k', lw=0.5)
    if ax_krnls_ind_tr is None:
        ax_krnls_ind_tr = ax_kernels
    else:
        ax_krnls_ind_tr = plt.axes(ax_krnls_ind_tr)
        ax_krnls_ind_tr.plot([1, 6], [0, 0], '--k', lw=0.5)
    data = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    # transform vals to float and then back to str
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    glm_ac = filter_mat(mat=data['glm_mats_ac'], mean_prf=mean_prf,
                        perf_th=perf_th, val_mat=data['val_mat'],
                        sel_vals=sel_vals)
    glm_ae = filter_mat(mat=data['glm_mats_ae'], mean_prf=mean_prf,
                        perf_th=perf_th, val_mat=data['val_mat'],
                        sel_vals=sel_vals)
    files = filter_mat(mat=data['files'], mean_prf=mean_prf, perf_th=perf_th,
                       val_mat=data['val_mat'], sel_vals=sel_vals)
    sel_fs = files if sel_files is None else sel_files
    plt_opts_insts = deepc(plt_opts)
    if plot_mean and ax_krnls_ind_tr == ax_kernels:
        plt_opts_insts['alpha'] = 0.1
    elif ax_krnls_ind_tr != ax_kernels:
        plt_opts_insts['lw'] = 0.3
    plt_opts_insts['marker'] = ''
    for ind_glm, glm_ac_tmp in enumerate(glm_ac):
        if len(glm_ac_tmp) != 0 and files[ind_glm] in sel_fs:
            glm_ae_tmp = glm_ae[ind_glm]
            pf.plot_kernels(glm_ac_tmp, glm_ae_tmp, ax=[ax_krnls_ind_tr],
                            inset_xs=0, ax_inset=None, regressors=regressors,
                            **plt_opts_insts)
    if plot_mean:
        a_glm_ac = hf.get_average(glm_ac)
        a_glm_ae = hf.get_average(glm_ae)
        std_glm_ac = hf.get_std(glm_ac)
        std_glm_ae = hf.get_std(glm_ae)
        pf.plot_kernels(a_glm_ac, a_glm_ae, std_ac=std_glm_ac, std_ae=std_glm_ae,
                        ax=[ax_kernels], inset_xs=0, ax_inset=None,
                        regressors=regressors, **plt_opts)
        ax_kernels.invert_xaxis()
    pf.rm_top_right_lines(ax=ax_kernels)
    ax_kernels.set_ylabel('Transition weight', fontsize=font)
    ax_kernels.set_xlabel('Trial lag', fontsize=font)
    if ax_krnls_ind_tr != ax_kernels:
        ax_krnls_ind_tr.invert_xaxis()
        pf.rm_top_right_lines(ax=ax_krnls_ind_tr)
        ax_krnls_ind_tr.set_ylabel('Transition weight', fontsize=font)
        ax_krnls_ind_tr.set_xlabel('Trial lag', fontsize=font)

    return ax_kernels


def fig_N2(main_folder, figsize=(6.5, 3.5), spacing=10000, perf_th=0.6,
           start=50000, conv_w=3, plot_exp_psych=False, sel_vals=['2'],
           name='', align=False, y_lim=[0.7, 0.85], binning=100000,
           folder_examples='alg_ACER_seed_0_n_ch_2', kernel_panel=None,
           fldrs_psychCurves=['200000000', '16000000', '8000000'],
           **plt_kwargs):
    """
    Plot figure for cosyne abstract.

    Parameters
    ----------
    figsize : tuple, optional
        size of figure ((6, 8))
    spacing : int, optional
        window used to subsample the performance mat. It is needed to plot
        the smoothed performance (10000)
    perf_th : float, optional
        threshold to filter instances by their final performances. Note that
        performances are relative to that of a perfect integrator (0.)
    sel_vals : list, optional
        selected n-ch values (['2', '4', '8', '16'])
    **plt_kwargs : dict
        dict containing info for plotting.
        plt_opts = {'lw': 1., 'alpha': , 'colors': , 'font': , 'font_inset': }
        it can also contain any key accepted by plt.plot

    Returns
    -------
    None.

    """
    margin = 0.1
    plt_opts = {'lw': 1., 'alpha': 1., 'colors': 0.5+np.zeros((20, 3)), 'font': 8,
                'font_inset': 7}
    plt_opts.update(plt_kwargs)
    del plt_opts['font_inset'], plt_opts['colors'], plt_opts['font']
    f_glm_tr, ax_tr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax_tr.remove()
    fig_factor = figsize[0]/figsize[1]
    # PLOT NETWORKS PERFORMANCE ON 2AFC AND PSYCHO-CURVES
    psy_size = .095
    prf_panel = [margin, 0.6, 0.2, 0.35]
    ri_panel = prf_panel.copy()
    ri_panel[0] += prf_panel[2]+psy_size+0.2
    if kernel_panel is None:
        kernel_panel = ri_panel.copy()
        kernel_panel[1] = margin
    ax_perfs = plot_nets_perf(main_folder=main_folder, sel_vals=sel_vals,
                              prf_panel=prf_panel, y_lim=y_lim, **plt_opts)

    psych_panel = [margin+prf_panel[2]+0.1, margin, psy_size,
                   psy_size*fig_factor]
    plot_psycho_curves(main_folder=main_folder, psych_panel=psych_panel,
                       folder_examples=folder_examples, font_inset=7,
                       fldrs_psychCurves=fldrs_psychCurves, ax_p=ax_perfs)
    # ax_perfs.set_xticks([])
    # PLOT RESET INDEX ACROSS TRAINING
    ax = plt.axes(ri_panel)
    plt_opts['color'] = 'k'
    events = {'exp': folder_examples, 'evs': fldrs_psychCurves}
    plot_ri_across_training(main_folder=main_folder, ax=ax, align=align,
                            perf_th=perf_th, sel_val=sel_vals[0], binning=binning,
                            evs=events, plot_ind_traces=False, **plt_opts)
    # PLOT KERNELS
    plt.figure(f_glm_tr.number)
    file = main_folder + '/data_ACER_test_2AFC_.npz'
    ax_kernel =\
        plot_kernels(file=file,  sel_vals=sel_vals, kernel_panel=kernel_panel,
                     perf_th=perf_th, **plt_kwargs)
    ax_kernel.set_yticks([-1, 0, 1, 2])
    pf.sv_fig(f=f_glm_tr, name=name+'_from_python', sv_folder=SV_FOLDER)


def plot_Tpp_krnls_diff_nch(file, ax):
    """
    Plot T++ kernels for different n_ch.

    Parameters
    ----------
    file : str
        file with data.
    ax : axis
        where to plot.

    Returns
    -------
    None.

    """
    f_temp, ax_temp = plt.subplots()
    pos = ax_temp.get_position()
    sel_vals_mat = ['2', '4', '8', '12', '16']
    alpha_mat = np.linspace(0.2, 1, len(sel_vals_mat))
    for i_sv, s_v in enumerate(sel_vals_mat):
        plt_kwargs = {'alpha': alpha_mat[i_sv]}
        ax.plot([1, 6], [0, 0], '--k', lw=0.5)
        plot_kernels(file=file,  sel_vals=[s_v], ax_kernels=ax,
                     perf_th=0.6, regressors=['T++'], ax_krnls_ind_tr=pos,
                     **plt_kwargs)
    plt.close(f_temp)
    n_stps = 50
    pos = ax.get_position()
    ax_clbr = plt.axes([pos.x0+pos.width, pos.y0,
                        pos.width/20, pos.height/1.5])
    ax_clbr.imshow(np.linspace(n_stps, 1, n_stps)[:, None], origin='lower',
                   cmap='gray', aspect='auto')
    ax_clbr.set_yticks([0, n_stps-1])
    ax_clbr.set_yticklabels(['2', '16'])
    ax_clbr.tick_params(labelsize=6)
    ax_clbr.set_title('$N_{max}$', fontsize=6)
    ax_clbr.set_xticks([])
    ax_clbr.yaxis.tick_right()

# --- FIG. N16


def fig_N16(main_folder, figsize=(6, 6), spacing=10000, perf_th=0.6,
            start=50000, conv_w=3, plot_exp_psych=False, sel_vals=['2'],
            name='', align=False, y_lim=[0.7, 0.85],
            folder_examples='alg_ACER_seed_0_n_ch_2',
            binning=100000, fldrs_psychCurves=['200000000', '16000000', '8000000'],
            **plt_kwargs):
    """
    Plot figure for cosyne abstract.

    Parameters
    ----------
    figsize : tuple, optional
        size of figure ((6, 8))
    spacing : int, optional
        window used to subsample the performance mat. It is needed to plot
        the smoothed performance (10000)
    perf_th : float, optional
        threshold to filter instances by their final performances. Note that
        performances are relative to that of a perfect integrator (0.)
    sel_vals : list, optional
        selected n-ch values (['2', '4', '8', '16'])
    **plt_kwargs : dict
        dict containing info for plotting.
        plt_opts = {'lw': 1., 'alpha': , 'colors': , 'font': , 'font_inset': }
        it can also contain any key accepted by plt.plot

    Returns
    -------
    None.

    """
    margin = 0.12
    plt_opts = {'lw': 1., 'alpha': 1.}
    plt_opts.update(plt_kwargs)
    f_glm_tr, ax_tr = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax_tr.remove()
    # PLOT NETWORKS PERFORMANCE ON 2AFC AND PSYCHO-CURVES
    panels_size = 0.2
    # PERF PANEL
    prf_panel = [margin, 1-margin/2-panels_size, panels_size, panels_size]
    # plt.axes(prf_panel)
    # BLOCKS PERF PANEL
    panel_2d = prf_panel.copy()
    panel_2d[0] = prf_panel[0]+prf_panel[2]+margin
    # plt.axes(panel_2d)
    # PSYCHO CURVE PANEL
    psych_panel = prf_panel.copy()
    psych_panel[0] = panel_2d[0]+prf_panel[2]+margin
    # plt.axes(psych_panel)
    # MEAN KERNELS
    mean_kernel_panel = prf_panel.copy()
    mean_kernel_panel[1] -= prf_panel[3]+margin
    # plt.axes(mean_kernel_panel)
    # IND KERNELS PANEL
    kernels_panel = mean_kernel_panel.copy()
    kernels_panel[0] = mean_kernel_panel[0]+mean_kernel_panel[2]+margin
    # plt.axes(kernels_panel)
    # RI PANEL
    ri_panel = kernels_panel.copy()
    ri_panel[0] = kernels_panel[0]+kernels_panel[2]+margin
    # plt.axes(ri_panel)
    # CONTRS
    contrs_panel = mean_kernel_panel.copy()
    contrs_panel[1] -= mean_kernel_panel[3]+margin
    # plt.axes(contrs_panel)
    # KERNELS VS NCH
    krnl_vs_n_panel = contrs_panel.copy()
    krnl_vs_n_panel[0] = contrs_panel[0]+contrs_panel[2]+margin
    ax_krnl_vs_n = plt.axes(krnl_vs_n_panel)
    # RI VS NCH
    ri_vs_n_panel = krnl_vs_n_panel.copy()
    ri_vs_n_panel[0] = krnl_vs_n_panel[0]+krnl_vs_n_panel[2]+margin
    # plt.axes(ri_vs_n_panel)

    # PLOT PERFORMANCE CONDITIONED ON N
    plot_nets_N_cond_perf(main_folder=main_folder, sel_vals=sel_vals,
                          prf_panel=prf_panel, pr_p='1', **plt_opts)
    plot_nets_N_cond_perf(main_folder=main_folder, sel_vals=sel_vals,
                          prf_panel=panel_2d, pr_p='0', **plt_opts)

    # PLOT PERFORMANCE IN REPETING VS CLOCKWISE BLOCKS
    # plot_nets_blck_cond_perf(main_folder=main_folder, sel_vals=sel_vals,
    #                          prf_panel=panel_2d, **plt_opts)
    # PLOT PSYCHOMETRIC CURVES
    ax_psych = plot_psycho_curves(main_folder=main_folder, psych_panel=psych_panel,
                                  folder_examples=folder_examples, font_inset=7,
                                  fldrs_psychCurves=[fldrs_psychCurves[-1]])
    ax_psych[0].set_yticks([0, 1])
    ax_psych[0].set_xlabel('Rep. stim. evidence', fontsize=8)
    ax_psych[0].set_ylabel('Repeating Probability', fontsize=8)
    # PLOT RESET INDEX ACROSS TRAINING
    ax = plt.axes(ri_panel)
    plt_opts['color'] = 'k'
    # events = {'exp': folder_examples, 'evs': fldrs_psychCurves}
    plot_ri_across_training(main_folder=main_folder, ax=ax, align=align,
                            perf_th=perf_th, sel_val=sel_vals[0], binning=binning,
                            ax_contrs=contrs_panel, **plt_opts)
    # PLOT KERNELS
    plt.figure(f_glm_tr.number)
    file = main_folder + '/data_ACER_test_2AFC_.npz'
    ax_kernel = plot_kernels(file=file,  sel_vals=sel_vals, perf_th=perf_th,
                             ax_krnls_ind_tr=kernels_panel,
                             kernel_panel=mean_kernel_panel,
                             **plt_kwargs)
    ax_kernel.set_yticks([-1, 0, 1, 2])
    ax_kernel.set_ylim([-0.35, 3])

    # PLOT RESET INDEX FOR DIFFERENT Ns
    sel_vals = ['2', '4', '8', '12', '16']
    bxp_ops = {'lw': 0.4}
    ax_RI_diff_nch = plot_reset_index_diff_vals(main_folder=main_folder,
                                                sel_vals=sel_vals, perf_th=0.6,
                                                ax_RI_diff_nch=ri_vs_n_panel,
                                                bxp_ops=bxp_ops)
    ax_RI_diff_nch.set_xlabel('Max. no. choices $N_{max}$')
    ax_RI_diff_nch.set_xticks(np.arange(len(sel_vals)))
    ax_RI_diff_nch.set_xticklabels(sel_vals)
    ax_RI_diff_nch.axhline(y=0.87, linestyle='--', color='k', lw=0.5)

    # PLOT KERNELS FOR DIFFERENT Ns
    plot_Tpp_krnls_diff_nch(file=file, ax=ax_krnl_vs_n)

    pf.sv_fig(f=f_glm_tr, name=name+'_from_python', sv_folder=SV_FOLDER)


def plot_nets_reset_index(main_folder,  sel_vals=['2', '4', '8', '16'],
                          prf_panel=[0.37, 0.44], perf_th=0.6, height=0.2,
                          **plt_kwargs):
    """
    Plot reset index (RI) for networks.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    sel_vals : list, optional
        max. num. of choices for which to plot the RI (['2', '4', '8', '16'])
    prf_panel : list, optional
        position of perf. panel for reference ([0.37, 0.44])
    perf_th : float, optional
        threshold to filter networks by performance
        (note that perf. is w.r.t perfect integrator) (0.)
    height: float, optional
        height of reset panel (0.2)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    plt_opts = {'lw': 1., 'alpha': 1., 'font': 8}
    plt_opts.update(plt_kwargs)
    font = plt_opts['font']
    del plt_opts['font']
    reset_panel = [prf_panel[0]+0.15, prf_panel[1]-(height+.09), 0.46, height]
    ax_RI_diff_nch = plt.axes(reset_panel)
    plot_opts = {'lstyle_ac': '-', 'lstyle_ae': '--'}
    file = main_folder + '/data_ctx_ch_prob_0.0125_test_2AFC_400K_per.npz'
    data = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    # transform vals to float and then back to str
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    vals = np.array([str(float(v)) for v, p in zip(data['val_mat'], mean_prf)
                     if p >= perf_th and v in sel_vals])
    glm_ac = np.array([b for b, p, v in zip(data['glm_mats_ac'], mean_prf,
                                            data['val_mat'])
                       if p >= perf_th and v in sel_vals])
    glm_ae = np.array([b for b, p, v in zip(data['glm_mats_ae'], mean_prf,
                                            data['val_mat'])
                       if p >= perf_th and v in sel_vals])
    unq_vals = np.unique(vals)[np.array([1, 2, 3, 0])]
    f_temp, ax_temp = plt.subplots()
    n_stps_ws = 1
    opts = {k: x for k, x in plot_opts.items() if k.find('_a') == -1}
    opts['marker'] = '+'
    opts['color'] = 'k'
    # filter experiments by performance
    for i_v, val_str in enumerate(unq_vals):
        glm_ac_cond = glm_ac[vals == val_str]
        glm_ae_cond = glm_ae[vals == val_str]
        opts['alpha'] = 0.5
        opts['markersize'] = 4
        reset_mat = []
        for ind_glm, glm_ac_tmp in enumerate(glm_ac_cond):
            if len(glm_ac_tmp) != 0:
                glm_ae_tmp = glm_ae_cond[ind_glm]
                ws_ac = np.nanmean(glm_ac_tmp[-n_stps_ws:, :, :], axis=0)
                ws_ac = np.expand_dims(ws_ac, 0)
                ws_ae = np.nanmean(glm_ae_tmp[-n_stps_ws:, :, :], axis=0)
                ws_ae = np.expand_dims(ws_ae, 0)
                xtcks = ['T++'+x for x in ['2', '3', '4', '5', '6-10']]
                reset, _, _ = pf.compute_reset_index(ws_ac, ws_ae, xtcks=xtcks,
                                                     full_reset_index=False)
                ax_RI_diff_nch.plot(i_v+np.random.randn()*0.01, reset, **opts)
                reset_mat.append(reset)
        opts['alpha'] = 1
        opts['markersize'] = 8
        ax_RI_diff_nch.errorbar(i_v, np.mean(reset_mat),
                                np.std(reset_mat)/np.sqrt(len(reset_mat)),
                                zorder=10, **opts)

    folder = '/home/molano/priors/rats/80_20/'
    prd.results_frm_matlab_glm(main_folder=folder, ax_inset=ax_RI_diff_nch,
                               ax_tr=[ax_temp], color=(.5, .5, .5),
                               name='80-20', plt_ind_trcs=False,
                               plt_ind_indx=True, x=i_v+1)
    plt.close(f_temp)
    # ax_RI_diff_nch.plot([2.9, 3], [.67, .77], 'k', lw=0.5)
    ax_RI_diff_nch.set_yticks([0, 1])
    ax_RI_diff_nch.set_xticks(np.arange(i_v+2))
    ax_RI_diff_nch.set_xticklabels(sel_vals+['Exp'])
    pf.rm_top_right_lines(ax=ax_RI_diff_nch)
    ax_RI_diff_nch.set_ylabel('Reset Index', fontsize=font)
    ax_RI_diff_nch.set_xlabel('Maximum number of Choices', fontsize=font)


def plot_example_psych_curve(main_folder, start=50000, conv_w=3, height=0.2,
                             seed=8, psy_size=0.16, fig_factor=1,
                             ax_16AFC_psy2=None, prf_panel=[0.37, 0.44],
                             **plt_kwargs):
    """
    Plot psycho-curves for network pre-trained with nch=16.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    fig_factor : float, optional
        factor to adjust panels height so panels are square (1)
    spacing : int, optional
        spacing used to subsample performances vectors (10000)
    prf_panel : list, optional
        position and size of main performance panel ([0.37, 0.44])
    conv_w : int, optional
        trials back to use to define the rep/alt contexts (3)
    psy_size : float, optional
        width of panels (height will be adjusted so panels are square) (0.16)
    start : int, optional
        margin used to select trials to compute psycho-curves
        (this is basically to avoid using early trials in which the networks
         don't do the task) (50000)
    height: float, optional
        height of reset panel (0.2)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    lbs = ['after error', 'after correct']
    colors = [hf.azul, hf.rojo]
    plt_opts = {'lw': 1., 'alpha': 1., 'font_inset': 7}
    plt_opts.update(plt_kwargs)
    font_inset = plt_opts['font_inset']
    del plt_opts['font_inset']
    if ax_16AFC_psy2 is None:
        psych_panel = (prf_panel[0]-0.13, prf_panel[1]-(height+.09),  psy_size,
                       psy_size*fig_factor)
        ax_16AFC_psy2 = plt.axes(psych_panel)
    folder = main_folder+'alg_ACER_seed_'+str(seed) +\
        '_n_ch_16_ctx_ch_prob_0.0125/test_2AFC/'
    _ = pl.put_together_files(folder)
    data = hf.load_behavioral_data(folder+'/bhvr_data_all.npz')
    ch = data['choice'][start:]
    sig_ev = data['putative_ev'][start:]
    prf = data['performance'][start:]
    # performance history
    tr_block = data['tr_block'][start:]
    prev_perf = np.concatenate((np.array([0]), prf[:-1]))
    for i_b, blk in enumerate(np.unique(tr_block)):
        for ip, p in enumerate([0, 1]):
            alpha = 0.7 if p == 0 else 1
            lnstyl = '--' if p == 0 else '-'
            plt_opts = {'color': colors[i_b], 'alpha': alpha, 'linestyle': lnstyl,
                        'lw': 0.5, 'markersize': 2}
            mask = hf.and_(prev_perf == p, tr_block == blk)
            popt, pcov, ev_mask, repeat_mask =\
                hf.bias_psychometric(choice=ch.copy(), ev=sig_ev.copy(),
                                     mask=mask, maxfev=100000)
            ev_mask = np.round(ev_mask, 2)  # this is to avoid rounding diffs
            plt_opts['label'] = str(popt[1])+'  '+lbs[ip]
            hf.plot_psycho_curve(ev=ev_mask, choice=repeat_mask, popt=popt,
                                 ax=ax_16AFC_psy2, **plt_opts)
    ax_16AFC_psy2.plot([-35, 35], [.5, .5], '--', lw=0.2, color=(.5, .5, .5))
    ax_16AFC_psy2.set_xlabel('Rep. evidence', fontsize=font_inset)
    ax_16AFC_psy2.set_ylabel('Prob. of repeat', fontsize=font_inset)
    # ax_16AFC_psy2.set_xticks([])
    ax_16AFC_psy2.set_title('')
    ax_16AFC_psy2.set_xlim([-35, 35])
    ax_16AFC_psy2.set_xticklabels([-.5, 0, .5])
    ax_16AFC_psy2.set_xticklabels(['-1', '0', '1'])
    # ax_16AFC_psy2.legend()  # .set_visible(False)


def plot_krnls_rbnd(main_folder, sel_vals=['16'], perf_th=0.6, plt_ind_tr=False,
                    glm_rbnd_1='glm_mats_acc', glm_rbnd_2='glm_mats_aec',
                    color_1=hf.naranja,
                    color_2=np.clip(hf.rojo+0.3, a_min=0, a_max=1),
                    after_cols_1=aftercc_cols, after_cols_2=afterec_cols,
                    kernel_panel=[0.1, 0.1, 0.1, 0.1], ax_kernels=None,
                    **plt_kwargs):
    """
    Plot kernels for rebound for RNNs.

    Parameters
    ----------
    main_folder : TYPE
        DESCRIPTION.
    sel_vals : TYPE, optional
        DESCRIPTION. The default is ['2'].
    perf_th : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    None.

    """
    plt_opts = {'fntsz': 8, 'color_ac': color_1, 'color_ae': color_2,
                'marker': '.', 'lstyle_ac': '-', 'lstyle_ae': '-', 'lw': 1.,
                'alpha': 1.}
    plt_opts.update(plt_kwargs)
    font = plt_opts['fntsz']
    del plt_opts['fntsz']
    if ax_kernels is None:
        ax_kernels = plt.axes(kernel_panel)
        ax_kernels.plot([1, 6], [0, 0], '--k', lw=0.5)
    file = main_folder + '/data_ACER*n_ch_16*_test_2AFC_rbnd.npz'
    data = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    # transform vals to float and then back to str
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    glm_1 = np.array([b for b, p, v in zip(data[glm_rbnd_1], mean_prf,
                                           data['val_mat'])
                      if p >= perf_th and v in sel_vals])
    glm_2 = np.array([b for b, p, v in zip(data[glm_rbnd_2], mean_prf,
                                           data['val_mat'])
                      if p >= perf_th and v in sel_vals])

    regressors = ['T++']
    plt_opts_insts = deepc(plt_opts)
    plt_opts_insts['alpha'] = 0.2
    all_krnls_1 = []
    all_krnls_2 = []
    for ind_glm, glm_1_tmp in enumerate(glm_1):
        if len(glm_1_tmp) != 0:
            glm_2_tmp = glm_2[ind_glm]
            if not plt_ind_tr:
                f_temp, ax_ind = plt.subplots()
            else:
                ax_ind = ax_kernels
            _, krnl_1, krnl_2, x_1, x_2 =\
                pf.plot_kernels(glm_1_tmp[None], glm_2_tmp[None],
                                ax=[ax_ind], inset_xs=0,
                                regressors=regressors, ac_cols=after_cols_1,
                                ae_cols=after_cols_2, **plt_opts_insts)

            all_krnls_1.append(krnl_1)
            all_krnls_2.append(krnl_2)
            if not plt_ind_tr:
                plt.close(f_temp)
    plt_opts['alpha'] = 1
    plt_opts['lw'] = 1
    mean_krnl_1 = np.mean(np.array(all_krnls_1), axis=0)
    mean_krnl_2 = np.mean(np.array(all_krnls_2), axis=0)
    std_krnl_1 = np.std(np.array(all_krnls_1), axis=0)
    std_krnl_2 = np.std(np.array(all_krnls_2), axis=0)
    opts = pf.get_opts_krnls(plt_opts, tag='_ac')
    ax_kernels.errorbar(x_1, mean_krnl_1, std_krnl_1, **opts)
    opts = pf.get_opts_krnls(plt_opts, tag='_ae')
    ax_kernels.errorbar(x_2, mean_krnl_2, std_krnl_2, **opts)
    pf.xtcks_krnls(xs=x_1, ax=ax_kernels)
    pf.rm_top_right_lines(ax=ax_kernels)
    ax_kernels.set_ylabel('GLM weight', fontsize=font)
    ax_kernels.set_xlabel('Trial lag', fontsize=font)
    return ax_kernels

# --- FIG REBOUND


def fig_rebound(folder_nets,
                folder_exps='/home/molano/priors/rats/data_Ainhoa/Rat*/',
                font=8):
    """
    Plot figure 6 (rebound).

    Parameters
    ----------
    folder : TYPE, optional
        DESCRIPTION. The default is '/home/molano/priors/rats/data_Ainhoa/Rat*/'.

    Returns
    -------
    None.

    """
    margin = 0.05
    bottom = 0.05
    height = 0.9
    exp_panel = [margin, bottom, 0.4, height]
    f_krnls = plt.figure(figsize=(4, 1.5))
    ax_exps = plt.axes(exp_panel)
    ax_exps.plot([1, 6], [0, 0], '--k', lw=0.5)
    # PLOT KERNELS FOR EXPERIMENTS
    folder = '/home/molano/priors/rats/data_Ainhoa/'  # Ainhoa's data
    prd.glm_krnls_rbnd(main_folder=folder, ax_glm_rbnd_krnls=ax_exps, tag='mat',
                       plt_ind_trcs=False, tags_mat=[['T++']])
    ax_exps.invert_xaxis()
    pf.rm_top_right_lines(ax=ax_exps)
    ax_exps.set_ylabel('GLM weight', fontsize=font)
    ax_exps.set_xlabel('Trial lag', fontsize=font)
    ax_exps.legend()
    # PLOT KERNELS FOR RNNS
    # plot kernels for cc and ec
    ax_rnns = plot_krnls_rbnd(main_folder, sel_vals=['16'], perf_th=0.,
                              kernel_panel=[exp_panel[2]+4*margin, bottom,
                                            exp_panel[2], height],
                              **{'fntsz': font})
    # plot kernels for ce and ee
    plot_krnls_rbnd(main_folder, sel_vals=['16'], perf_th=0.,
                    glm_rbnd_1='glm_mats_ace', glm_rbnd_2='glm_mats_aee',
                    color_1=(.7, .7, .7), color_2=(0, 0, 0),
                    after_cols_1=afterce_cols, after_cols_2=afteree_cols,
                    ax_kernels=ax_rnns, **{'fntsz': font})
    ax_rnns.invert_xaxis()
    pf.sv_fig(f=f_krnls, name='fig_6_from_python', sv_folder=SV_FOLDER)


def create_axes(ncols=10, nrows=4, figsize=(8, 4)):
    """
    Create axes for Fig. 5 (transition matrices).

    Parameters
    ----------
    ncols : int, optional
        number of cols (10)
    nrows : int, optional
        number of row (4)
    figsize : tuple, optional
        size of figure ((8, 4))

    Returns
    -------
    TYPE
        fig, main_axes, ax_circles, label_axes

    """
    def setup_axes(fig, rect, axisScale=[1, 5], axisLimits=(-0, 7, 0, 1),
                   rotation=-45):
        tr = Affine2D().scale(axisScale[0], axisScale[1]).rotate_deg(rotation)
        grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=axisLimits)
        ax = floating_axes.FloatingSubplot(fig, grid_helper=grid_helper, *rect)
        fig.add_subplot(ax)
        aux_ax = ax.get_aux_axes(tr)
        grid_helper.grid_finder.grid_locator1._nbins = 1
        grid_helper.grid_finder.grid_locator2._nbins = 1
        return ax, aux_ax
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    # remove top axes to add rotated histograms
    for a in axs[0, :]:
        a.remove()
    # remove odd axes
    for a in axs[1, range(1, ncols, 2)]:
        a.remove()
    # create axis for circles
    pos = axs[3, 0].get_position()
    ax_circles = plt.axes([pos.x0-.7*pos.width, pos.y0-pos.height/4,
                           1.32*ncols*pos.width, 2.64*pos.height])
    ax_circles.set_frame_on(False)
    ax_circles.set_xticks([])
    ax_circles.set_yticks([])
    for a in axs[2:, :].flatten():
        a.remove()
    axes = []
    label_axes = []
    for i_ax in range(2, ncols+1, 2):
        ax, aux_ax = setup_axes(fig, rect=[nrows, ncols, i_ax])
        axes.append(aux_ax)
        label_axes.append(ax)
        # ax.axis["left"].label.set_text('Freq.')
        ax.axis['bottom'].toggle(ticklabels=False)
        ax.axis['left'].toggle(ticklabels=False)
        for axisLoc in ['top', 'right']:
            ax.axis[axisLoc].set_visible(False)
        pos = axs[1, i_ax-2].get_position()
        axs[1, i_ax-2].set_position([pos.x0-pos.width/4, pos.y0-pos.height/4,
                                     1.5*pos.width, 1.5*pos.height])
        pos = axs[1, i_ax-2].get_position()
        ax.set_position([pos.x0+pos.width-0.082, pos.y0+pos.height-.074, .2, .2])
    main_axes = np.array([axes, axs[1, range(0, ncols, 2)]])
    return fig, main_axes, ax_circles, label_axes


def highlight_squares(ax, n=6, margin=.45, lw=.4):
    """
    Highligth transition specific to rep, cw and acw contexts.

    Parameters
    ----------
    ax : axis
        where to plot.
    n : int, optional
        size of transition matrix (6)
    margin : float, optional
        margin to left between squares (.45)
    lw : float, optional
        line width (.4)

    Returns
    -------
    None.

    """
    for i_sq in range(n):
        ax.add_patch(Rectangle((i_sq-margin, ((i_sq+1) % n)-margin), 2*margin,
                               2*margin, facecolor='none', edgecolor=hf.rosa,
                               lw=lw))
        ax.add_patch(Rectangle((i_sq-margin, ((i_sq-1) % n)-margin), 2*margin,
                               2*margin, facecolor='none', edgecolor=hf.rojo_2,
                               lw=lw))
        ax.add_patch(Rectangle((i_sq-margin, i_sq-margin), 2*margin, 2*margin,
                               facecolor='none', edgecolor=hf.azul, lw=lw))


def plot_circles_and_arrows(ax, num_circles=5, ch_list=['1', '2', '3', '4', '5'],
                            clr_list=['k', 'k', 'k', 'k', 'k'], circ_sz=0.25):
    """
    Plot circles and arrows in figure 5.

    Parameters
    ----------
    ax : axis
        where to plot.
    num_circles : int, optional
        number of circles (5)
    ch_list : list, optional
        letter to add in each circle (['1', '2', '3', '4', '5'])
    clr_list : list, optional
        color of each circle (['k', 'k', 'k', 'k', 'k'])
    circ_sz : float, optional
        circle size (0.25)

    Returns
    -------
    None.

    """
    def demo_con_style(ax, strt_pnt, end_pnt, color, lnstl, connectionstyle):
        ax.annotate("", xy=strt_pnt, xycoords='data',
                    xytext=end_pnt, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color,
                                    shrinkA=5, shrinkB=5, lw=1,
                                    patchA=None, patchB=None, linestyle=lnstl,
                                    connectionstyle=connectionstyle))
    for i_c in range(num_circles):
        ls = '--' if i_c == num_circles-1 else '-'
        pos = [max(2*i_c-1, -0.05), 2*(i_c+1)-1]
        demo_con_style(ax=ax, strt_pnt=(pos[1], 1), end_pnt=(pos[0], 1),
                       color=clr_list[i_c], lnstl=ls,
                       connectionstyle="angle3,angleA=45,angleB=-45")
        circle1 = plt.Circle((pos[0], 1-circ_sz), circ_sz, edgecolor='k',
                             facecolor='none')
        ax.add_patch(circle1)
        ax.annotate(ch_list[i_c], xy=(pos[0], 1-circ_sz-.025), fontsize=20,
                    horizontalalignment='center', verticalalignment='center')
    ax.set_ylim(-1., 2)
    ax.set_xlim(-1., 10)


def fig_trans_mats(file, perf_th=0.6, sel_vals=['16'], n_ch=6, remove_first=True):
    """
    Plot transition matrices after specific sequences.

    Parameters
    ----------
    file : str
        file to load.
    perf_th : float, optional
        performance threshold to filter experiments (0.6)
    sel_vals : list, optional
        list of strings specifying max-n-ch values to use (['16'])
    n_ch : int, optional
        number of choices used to obtained the behav. data  (6)

    Returns
    -------
    None.

    """
    data = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    # transform vals to float and then back to str
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    tr_mats_cond = filter_mat(mat=data['trans_mats'], mean_prf=mean_prf,
                              perf_th=perf_th, val_mat=data['val_mat'],
                              sel_vals=sel_vals)
    contexts = ['rbnd', 'to_cw', 'to_ccw', 'to_rep', 'neutr_trans']
    for blk_ctx, seq, arr_clrs in zip(['ccw_', 'cw_', 'rep_'],
                                      [['6', '5', '4'], ['2', '3', '4'],
                                       ['4', '4', '4']],
                                      [2*[hf.rojo_2], 2*[hf.rosa], 2*[hf.azul]]):
        arrows_colors = [[arr_clrs[0]]+[(0, 0, 0)], 2*[hf.rosa], 2*[hf.rojo_2],
                         2*[hf.azul], [(.7, .7, .7), arr_clrs[0]]]
        arrows_colors = [x+[(0, 0, 0)] for x in arrows_colors]

        if blk_ctx == 'ccw_':
            seqs = [['3', '5'], ['5', '6'], ['3', '2'], ['4', '4'], ['2', '1']]
        elif blk_ctx == 'cw_':
            seqs = [['5', '3'], ['5', '6'], ['3', '2'], ['4', '4'], ['2', '3']]
        elif blk_ctx == 'rep_':
            seqs = [['4', '3'], ['5', '6'], ['3', '2'], ['4', '4'], ['2', '2']]

        seq_tmp = [seq+x for x in seqs]
        arr_clrs_tmp = [arr_clrs+x for x in arrows_colors]
        for ctxt, sq, ac in zip(contexts, seq_tmp, arr_clrs_tmp):
            f, axs_trans_mats, ax_circles, lbl_axes = create_axes()
            plot_circles_and_arrows(ax=ax_circles, ch_list=sq,
                                    clr_list=ac)
            f_temp = plt.figure()
            ax_perf = plt.axes([0.1, 0.8, 0.75, 0.1])
            axs = [axs_trans_mats, ax_perf]
            tr_mats = []
            al_tr_mats = []
            num_s_mat = []
            perf_mat = []
            for i_tr, tr_m in enumerate(tr_mats_cond):
                d = tr_m[blk_ctx+ctxt]
                tr_mats.append(np.array(d['tr_mats']))
                al_tr_mats.append(np.array(d['al_tr_mats']))
                num_s_mat.append(np.array(d['num_s_mat']))
                perf_mat.append(np.array(d['perf_mat']))
                # if blk_ctx == 'ccw_' and ctxt == 'rbnd':
                # print(d['al_tr_mats'][4])
                plot_opts = {'color': (.7, .7, .7)}
                pf.plot_trans_prob_mats_after_error(trans_mats=d['tr_mats'],
                                                    al_trans_mats=d['al_tr_mats'],
                                                    num_samples_mat=d['num_s_mat'],
                                                    perf_mat=d['perf_mat'],
                                                    n_ch=n_ch, sv_folder=SV_FOLDER,
                                                    name=blk_ctx+ctxt, axs=axs,
                                                    plot_mats=False, **plot_opts)

            sem_ = np.sqrt(len(perf_mat))
            tr_mats = np.mean(np.array(tr_mats), axis=0)
            al_tr_mats = np.mean(np.array(al_tr_mats), axis=0)
            num_s_mat = np.mean(np.array(num_s_mat), axis=0)
            perf_mat = np.mean(np.array(perf_mat), axis=0)
            std_vals = {'perf_mat': np.std(np.array(perf_mat), axis=0)/sem_,
                        'al_trans_mats': np.std(np.array(al_tr_mats), axis=0)/sem_}
            plot_opts = {'color': 'k'}
            im = pf.plot_trans_prob_mats_after_error(trans_mats=tr_mats,
                                                     al_trans_mats=al_tr_mats,
                                                     num_samples_mat=num_s_mat,
                                                     perf_mat=perf_mat, n_ch=n_ch,
                                                     sv_folder=SV_FOLDER,
                                                     name=blk_ctx+ctxt,
                                                     axs=axs, std_vals=std_vals,
                                                     plot_mats=True, **plot_opts)
            for a in axs_trans_mats[1]:
                a.set_frame_on(False)
                highlight_squares(a)
            cbar_ax = f.add_axes([0.91, 0.5, 0.01, 0.2])
            f.colorbar(im, cax=cbar_ax)
            plt.close(f_temp)
            if remove_first:
                axs_trans_mats[0][0].remove()
                axs_trans_mats[1][0].remove()
                lbl_axes[0].remove()
            pf.sv_fig(f=f, name=blk_ctx+ctxt+'_bis', sv_folder=SV_FOLDER)


def zt(tau=10, window=200, resp_lat=100):
    """
    Plot cartoon example of zT.

    Parameters
    ----------
    tau : flaot, optional
        tau for exponential decay (10)
    window : int, optional
        duration of entire trace (200)
    resp_lat : int, optional
        response latency (100)

    Returns
    -------
    firing_rate : TYPE
        DESCRIPTION.

    """
    resp_dur = window-resp_lat
    response = np.exp(-np.arange(resp_dur)/tau)
    firing_rate = np.zeros((window,))
    firing_rate[resp_lat:] += response

    return firing_rate


def fig_theoretical_trans_matrices(num_ch=6, probs=0.8):
    """
    Plot example matrices with transition probabilities.

    Returns
    -------
    None.

    """
    indx = np.arange(num_ch)
    tr_mat = np.eye(num_ch)*probs
    tr_mat1 = tr_mat[indx, :]
    tr_mat1[tr_mat1 == 0] = (1-probs)/(num_ch-1)
    # clockwise context
    indx = np.append(np.arange(1, num_ch), 0)
    tr_mat = np.eye(num_ch)*probs
    tr_mat2 = tr_mat[indx, :]
    tr_mat2[tr_mat2 == 0] = (1-probs)/(num_ch-1)
    # repeating context
    indx = np.insert(np.arange(0, num_ch-1), 0, num_ch-1)
    tr_mat = np.eye(num_ch)*probs
    tr_mat3 = tr_mat[indx, :]
    tr_mat3[tr_mat3 == 0] = (1-probs)/(num_ch-1)

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(3, 1))
    ax[0].imshow(tr_mat1, vmin=0, vmax=1, aspect='auto', origin='lower',
                 cmap='Greys_r')
    ax[0].set_title('Repeating', color=hf.azul)
    ax[1].imshow(tr_mat3, vmin=0, vmax=1, aspect='auto', origin='lower',
                 cmap='Greys_r')
    ax[1].set_title('Clockwise', color=hf.rosa)
    im = ax[2].imshow(tr_mat2, vmin=0, vmax=1, aspect='auto', origin='lower',
                      cmap='Greys_r')
    ax[2].set_title('Anticlockwise', color=hf.rojo_2)
    cbar_ax = f.add_axes([0.91, 0.11, 0.02, 0.77])
    f.colorbar(im, cax=cbar_ax)
    ax[0].set_xlabel('Choice trial t')
    ax[0].set_ylabel('Choice trial t+1')
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    pf.sv_fig(f=f, name='fig_3__trMats_from_python', sv_folder=SV_FOLDER)


def plot_all_trans_regr_exp(main_folder='/home/molano/priors/rats', font=8,
                            ax_kernel_exp=None):
    """
    Plot reset index for experiments and ideal observer.

    Parameters
    ----------
    main_folder : str, optional
        where to find the data ('/home/molano/priors/rats')
    font : float, optional
        fontsize of labels (8)

    Returns
    -------
    None.

    """
    if ax_kernel_exp is None:
        f_kernel_exp, ax_kernel_exp = plt.subplots(nrows=2, ncols=2,
                                                   figsize=(4, 4))
    else:
        f_kernel_exp = None
    ax_kernel_exp = ax_kernel_exp.flatten()
    f_temp, ax_temp = plt.subplots(nrows=1, ncols=1)
    for ax in ax_kernel_exp:
        ax.plot([1, 6], [0, 0], '--k', lw=0.5)
    # axs_glm_krnls = [[ax_temp]]
    folder = main_folder+'/80_20/'
    prd.results_frm_matlab_glm(main_folder=folder, ax_inset=ax_temp,
                               ax_tr=ax_kernel_exp, name='80-20',
                               plt_ind_trcs=False)
    # PLOT RESET INDEX FOR IDEAL OBSERVER
    regressors = ['T++', 'T-+', 'T+-', 'T--']
    for ax, r in zip(ax_kernel_exp, regressors):
        ax.set_title(r, fontsize=font)
        ax.set_ylim([-0.9, 0.9])
        ax.invert_xaxis()
        pf.rm_top_right_lines(ax=ax)
        ax.set_xticks(np.arange(6)+1)
        ax.set_xticklabels(['1', '2', '3', '4', '5', '6-10'])
        ax.set_xlim([6.2, .5])
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=font)

    ax_kernel_exp[0].set_ylabel('GLM weight', fontsize=font)
    ax_kernel_exp[2].set_ylabel('GLM weight', fontsize=font)
    ax_kernel_exp[2].set_xlabel('Trial lag', fontsize=font)
    ax_kernel_exp[3].set_xlabel('Trial lag', fontsize=font)
    ax_kernel_exp[1].set_yticks([])
    ax_kernel_exp[3].set_yticks([])
    if f_kernel_exp is not None:
        pf.sv_fig(f=f_kernel_exp, name='supp_all_trans_regr_exps_from_python',
                  sv_folder=SV_FOLDER)


def plot_all_krnls(sel_vals=['2'], ylim=[-2.3, 3], ax=None,
                   regressors=['T++', 'T-+', 'T+-', 'T--'], **plt_kwargs):
    """
    Plot transition and lateral kernels for specific simulations.

    Parameters
    ----------
    main_folder : str
        where to get the data.
    sel_vals : list, optional
        list of strings specifying the values (e.g. num. of chs) to select (['2'])
    ylim : list, optional
        limits for y-axis ([-2.3, 3])
    **plt_kwargs : dir
        plotting options.

    Returns
    -------
    None.

    """
    # this is created and stored in plot_ri_across_training
    files = np.load(SV_FOLDER+'ri_max_'+''.join(sel_vals)+'.npz',
                    allow_pickle=1)
    save_fig = False
    if ax is None:
        f, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
        ax = ax.flatten()
        save_fig = True
    # regressors = ['T++', 'T-+', 'L+', 'T+-', 'T--', 'L-']
    regressors = ['T++', 'T-+', 'T+-', 'T--']
    for i_a, (a, r) in enumerate(zip(ax, regressors)):
        a.plot([1, 6], [0, 0], '--k', lw=0.5)
        # for each instance, we load and plot the data corresponding to max. RI
        for i_k, k in enumerate(files.keys()):
            file = files[k].item()['f']
            plot_kernels(file=file,  sel_vals=sel_vals, ax_kernels=a,
                         perf_th=0., regressors=[r], sel_files=[k],
                         plot_mean=False, **plt_kwargs)
        a.set_ylim(ylim)
        a.set_title(r, fontsize=8)
        if i_a in [1, 2, 4, 5]:
            a.set_ylabel('')
            # a.set_yticks([])
        if i_a in [0, 1, 2]:
            a.set_xlabel('')
        a.invert_xaxis()
        # a.set_xticks([])
    if save_fig:
        pf.sv_fig(f=f, name='supp_all_regressors_N_'+sel_vals[0]+'_from_python',
                  sv_folder=SV_FOLDER)


def plot_reset_index_diff_vals(main_folder, perf_th=0.6, ax_RI_diff_nch=None,
                               sel_vals=['.01', '.05', '.1', '.5', '.75', '1.0'],
                               name='', file=None, eq_dist_xs=True, bxp_ops={},
                               offset_xs=0, connect_values=False,
                               plt_ind_vals=True, **plt_kwargs):
    """
    Plot reset index (RI) for RNNs trained with different values (e.g. n_ch).

    Parameters
    ----------
    main_folder : str
        where to find the data.
    sel_vals : list, optional
        values to consider (e.g. prob_12=['0.01', '0.05', '0.1',
                                          '0.5', '0.75', '1.0'])
    prf_panel : list, optional
        position of perf. panel for reference ([0.37, 0.44])
    perf_th : float, optional
        threshold to filter networks by performance
        (note that perf. is w.r.t perfect integrator) (0.)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    bxplt_ops = {'lw': 0.2, 'fliersize': 3, 'widths': 0.5, 'color': 'k'}
    bxplt_ops.update(bxp_ops)
    widths = bxplt_ops['widths']
    plt_opts = {'lw': 1., 'alpha': 1., 'font': 8, 'marker': '.'}
    plt_opts.update(plt_kwargs)
    font = plt_opts['font']
    plt_opts['color'] = bxplt_ops['color']
    del plt_opts['font']
    if ax_RI_diff_nch is None:
        f, ax_RI_diff_nch = plt.subplots(figsize=(3, 2))
    else:
        ax_RI_diff_nch = plt.axes(ax_RI_diff_nch)
    if file is None:
        file = main_folder + '/data_ACER_test_2AFC_.npz'
    data = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    # transform vals to float and then back to str
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    vals = np.array([v for v, p in zip(data['val_mat'], mean_prf)
                     if p >= perf_th and v in sel_vals])
    glm_ac = np.array([b for b, p, v in zip(data['glm_mats_ac'], mean_prf,
                                            data['val_mat'])
                       if p >= perf_th and v in sel_vals])
    glm_ae = np.array([b for b, p, v in zip(data['glm_mats_ae'], mean_prf,
                                            data['val_mat'])
                       if p >= perf_th and v in sel_vals])
    files = np.array([b for b, p, v in zip(data['files'], mean_prf,
                                           data['val_mat'])
                      if p >= perf_th and v in sel_vals])

    unq_vals = np.unique(vals)
    n_stps_ws = 1
    if connect_values:
        medians = []
        indexes = []
    # filter experiments by performance
    for i_v, val_str in enumerate(unq_vals):
        glm_ac_cond = glm_ac[vals == val_str]
        glm_ae_cond = glm_ae[vals == val_str]
        f_cond = files[vals == val_str]
        if eq_dist_xs:
            indx = np.where(np.array(sel_vals) == val_str)[0]+offset_xs
        else:
            indx = np.array([float(val_str)])+offset_xs
        plt_opts['alpha'] = 0.2
        plt_opts['edgecolor'] = 'none'
        reset_mat = []
        for ind_glm, glm_ac_tmp in enumerate(glm_ac_cond):
            if len(glm_ac_tmp) != 0:
                glm_ae_tmp = glm_ae_cond[ind_glm]
                ws_ac = np.nanmean(glm_ac_tmp[-n_stps_ws:, :, :], axis=0)
                ws_ac = np.expand_dims(ws_ac, 0)
                ws_ae = np.nanmean(glm_ae_tmp[-n_stps_ws:, :, :], axis=0)
                ws_ae = np.expand_dims(ws_ae, 0)
                xtcks = ['T++'+x for x in ['2', '3', '4', '5', '6-10']]
                reset, krnl_ac, krnl_ae =\
                    pf.compute_reset_index(ws_ac, ws_ae, xtcks=xtcks,
                                           full_reset_index=False)
                contr_ac = np.abs(np.mean(krnl_ac))
                contr_ae = np.abs(np.mean(krnl_ae))
                fctr = indx if not eq_dist_xs else 1
                if plt_ind_vals:
                    ax_RI_diff_nch.scatter(indx+np.random.randn()*0.001*fctr,
                                           reset, **plt_opts)
                reset_mat.append(reset)
                if val_str == '16':
                    print(f_cond[ind_glm])
                    print(reset)
                    print(contr_ac)
                    print(contr_ae)
                    print('------')
        plt_opts['alpha'] = 1
        del plt_opts['edgecolor']
        if not eq_dist_xs:
            bxplt_ops['widths'] = widths*indx
        pf.box_plot(data=reset_mat, ax=ax_RI_diff_nch, x=indx[0], **bxplt_ops)
        if connect_values:
            indexes.append(indx[0])
            medians.append(np.median(reset_mat))
    # ax_RI_diff_nch.plot([2.9, 3], [.67, .77], 'k', lw=0.5)
    if connect_values:
        ax_RI_diff_nch.plot(indexes, medians, '-', lw='1',
                            color=plt_opts['color'])
    ax_RI_diff_nch.set_yticks([0, 1])
    pf.rm_top_right_lines(ax=ax_RI_diff_nch)
    ax_RI_diff_nch.set_ylabel('Reset Index', fontsize=font)
    if "f" in locals():
        ax_RI_diff_nch.set_xticks(np.arange(i_v+1))
        ax_RI_diff_nch.set_xticklabels(sel_vals)
        pf.sv_fig(f=f, name=name+'_from_python', sv_folder=SV_FOLDER)
    return ax_RI_diff_nch


def plot_perf_diff_vals(main_folder, perf_th=0.6, ax_prf_diff_nch=None,
                        sel_vals=['.01', '.05', '.1', '.5', '.75', '1.0'],
                        name='', file=None, eq_dist_xs=True, bxp_ops={},
                        m_or_bx='bx', separate_rep_alt=True,
                        connect_values=False, plt_ind_vals=True, **plt_kwargs):
    """
    Plot reset index (RI) for RNNs trained with different values (e.g. n_ch).

    Parameters
    ----------
    main_folder : str
        where to find the data.
    sel_vals : list, optional
        values to consider (e.g. prob_12=['0.01', '0.05', '0.1',
                                          '0.5', '0.75', '1.0'])
    prf_panel : list, optional
        position of perf. panel for reference ([0.37, 0.44])
    perf_th : float, optional
        threshold to filter networks by performance
        (note that perf. is w.r.t perfect integrator) (0.)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    connect_values = connect_values and not separate_rep_alt
    bxplt_ops = {'lw': 0.2, 'fliersize': 3, 'widths': 0.5, 'color': 'c'}
    bxplt_ops.update(bxp_ops)
    base_color = bxplt_ops['color']
    widths = bxplt_ops['widths']
    opts = {'alpha': 1., 'font': 8}
    opts.update(plt_kwargs)
    opts['color'] = bxplt_ops['color']
    font = opts['font']
    del opts['font']
    if ax_prf_diff_nch is None:
        f, ax_prf_diff_nch = plt.subplots(figsize=(3, 2))
    else:
        ax_prf_diff_nch = plt.axes(ax_prf_diff_nch)
    ax_prf_diff_nch.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    if file is None:
        file = main_folder + '/data_ACER_test_2AFC_.npz'
    data = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    # transform vals to float and then back to str
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    vals = np.array([v for v, p in zip(data['val_mat'], mean_prf)
                     if p >= perf_th and v in sel_vals])
    all_perf_mat = np.array([b for b, p, v in zip(data['perfs_cond'], mean_prf,
                                                  data['val_mat'])
                             if p >= perf_th and v in sel_vals])
    unq_vals = np.unique(vals)
    n_stps_ws = 1
    opts['alpha'] = 0.5
    opts['edgecolor'] = 'none'
    if connect_values:
        medians = []
        indexes = []
    # filter experiments by performance
    for i_v, val_str in enumerate(unq_vals):
        perf_mat_cond = all_perf_mat[vals == val_str]
        if eq_dist_xs:
            indx = np.where(np.array(sel_vals) == val_str)[0]
        else:
            indx = np.array([float(val_str)])
        perf_rep = []
        if separate_rep_alt:
            perf_alt = []
        for ind_prf, prf_tmp in enumerate(perf_mat_cond):
            if len(prf_tmp) != 0:
                fctr = indx if not eq_dist_xs else 1
                mean_rep = np.mean(prf_tmp['2-1']['m_perf'])+1/2
                mean_alt = np.mean(prf_tmp['2-2']['m_perf'])+1/2
                mean_rep = mean_rep if separate_rep_alt else (mean_rep+mean_alt)/2
                perf_rep.append(mean_rep)
                if separate_rep_alt:
                    perf_alt.append(mean_alt)
                if m_or_bx == 'bx' and plt_ind_vals:
                    opts['color'] = base_color
                    ax_prf_diff_nch.scatter(indx+np.random.randn()*0.001*fctr,
                                            mean_rep, **opts)
                    if separate_rep_alt:
                        opts['color'] = hf.rojo
                        ax_prf_diff_nch.scatter(indx+np.random.randn()*0.001*fctr,
                                                mean_alt, **opts)
        m_r = np.median(perf_rep)
        if connect_values:
            medians.append(m_r)
            indexes.append(indx[0])
        if m_or_bx == 'bx':
            if not eq_dist_xs:
                bxplt_ops['widths'] = widths*indx
            bxplt_ops['color'] = base_color
            pf.box_plot(data=perf_rep, ax=ax_prf_diff_nch, x=indx[0],
                        **bxplt_ops)
            if separate_rep_alt:
                bxplt_ops['color'] = hf.rojo
                pf.box_plot(data=perf_alt, ax=ax_prf_diff_nch, x=indx[0],
                            **bxplt_ops)
        else:
            s_r = np.std(perf_rep)  # /len(perf_rep)
            ax_prf_diff_nch.errorbar(indx[0], m_r, s_r, color=base_color,
                                     marker='.')
            if separate_rep_alt:
                m_a = np.mean(perf_alt)
                s_a = np.std(perf_alt)/len(perf_alt)
                ax_prf_diff_nch.errorbar(indx[0], m_a, s_a, color=hf.rojo,
                                         marker='.')

    if connect_values:
        ax_prf_diff_nch.plot(indexes, medians, '-', color=base_color, lw=1)
    ax_prf_diff_nch.set_ylabel('Performance', fontsize=font)
    pf.rm_top_right_lines(ax=ax_prf_diff_nch)
    if "f" in locals():
        ax_prf_diff_nch.set_xticks(np.arange(i_v+1))
        ax_prf_diff_nch.set_xticklabels(sel_vals)
        pf.sv_fig(f=f, name=name+'_from_python', sv_folder=SV_FOLDER)
    return ax_prf_diff_nch


def compute_probs(N, lamb, rounding=5, verbose=False, eps=0.001):
    """
    Compute probs of success just following transition history for a given n-ch=N.

    Parameters
    ----------
    N : int
        n-ch.
    lamb : float
        transition probability to most likely choice.
    rounding : int, optional
        number of decimals (5)
    verbose : bool, optional
        print or not probs (False)
    eps : float, optional
        small value for probs checking (0.001)

    Returns
    -------
    data : dict
        data with all probs.

    """
    after_corr_congr_ch_corr = lamb
    after_corr_congr_ch_err = np.round((1-lamb), rounding)
    assert np.abs(after_corr_congr_ch_corr+after_corr_congr_ch_err-1) < eps

    after_corr_incongr_ch_corr = np.round((1-lamb)/(N-1), rounding)
    after_corr_incongr_ch_err = 1-after_corr_incongr_ch_corr
    assert np.abs(after_corr_incongr_ch_corr+after_corr_incongr_ch_err-1) < eps

    after_err_congr_ch_corr = after_corr_incongr_ch_corr
    after_err_congr_ch_err = after_corr_incongr_ch_err
    assert np.abs(after_err_congr_ch_corr+after_err_congr_ch_err-1) < eps

    # (λ+(N-2)*(1-λ))/(N-1)
    after_err_incongr_ch_corr =\
        np.round((after_corr_congr_ch_corr +
                  (N-2)*after_corr_incongr_ch_corr)/(N-1), rounding)

    # ((1-λ)+(N-2)*λ)/(N-1)

    after_err_incongr_ch_err =\
        np.round((after_corr_congr_ch_err +
                  (N-2)*(after_corr_congr_ch_corr +
                         (N-2)*after_corr_incongr_ch_corr))/(N-1), rounding)

    assert np.abs(after_err_incongr_ch_corr+after_err_incongr_ch_err-1) < eps,\
        str(after_err_incongr_ch_corr+after_err_incongr_ch_err)
    if verbose:
        print('\n\nxxxxxxxxxxxxxxxxx')
        print('N = ', N)
        print('AFTER CORRECT TRIAL')
        print('Probability of congruent correct choice')
        print(after_corr_congr_ch_corr)
        print('Probability of congruent error choice')
        print(after_corr_congr_ch_err)
        print('Probability of incongruent correct choice')
        print(after_corr_incongr_ch_corr)
        print('Probability of incongruent error choice')
        print(after_corr_incongr_ch_err)

        print('\nAFTER ERROR TRIAL')
        print('Probability of congruent correct choice')
        print(after_err_congr_ch_corr)
        print('Probability of congruent error choice')
        print(after_err_congr_ch_err)
        print('Probability of incongruent correct choice')
        print(after_err_incongr_ch_corr)
        print('Probability of incongruent error choice')
        print(after_err_incongr_ch_err)
    data = {'after_corr_congr_ch_corr': after_corr_congr_ch_corr,
            'after_corr_congr_ch_err': after_corr_congr_ch_err,
            'after_corr_incongr_ch_corr': after_corr_incongr_ch_corr,
            'after_corr_incongr_ch_err': after_corr_incongr_ch_err,
            'after_err_congr_ch_corr': after_err_congr_ch_corr,
            'after_err_congr_ch_err': after_err_congr_ch_err,
            'after_err_incongr_ch_corr': after_err_incongr_ch_corr,
            'after_err_incongr_ch_err': after_err_incongr_ch_err}
    return data


def fig_theoretical_accuracies_NAFC():
    """
    Accuracies when using trans. hist. after error/correct for diff. num. of chs.

    Returns
    -------
    None.

    """
    inset = False
    data = {'after_corr_congr_ch_corr': [],
            'after_corr_congr_ch_err': [],
            'after_corr_incongr_ch_corr': [],
            'after_corr_incongr_ch_err': [],
            'after_err_congr_ch_corr': [],
            'after_err_congr_ch_err': [],
            'after_err_incongr_ch_corr': [],
            'after_err_incongr_ch_err': []}
    ns = range(2, 200)
    for n in ns:
        dat = compute_probs(N=n, lamb=0.8)
        for key in data.keys():
            data[key].append(dat[key])
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5))
    if inset:
        ax_inset = plt.axes((0.7, 0.62, 0.2, 0.2))
    ax.plot(ns, data['after_corr_congr_ch_corr'], 'k', label='After correct')
    ax.plot(ns, data['after_err_incongr_ch_corr'], 'k', alpha=.5,
            label='After error (Reverse)')

    ax.legend()
    ax.set_xlabel('Number of choices')
    ax.set_ylabel('Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if inset:
        ax_inset.plot(ns, np.array(data['after_err_congr_ch_corr']) /
                      np.array(data['after_err_incongr_ch_corr']),
                      label='after error (green-solid/green-dashed)', color='k')
        ax_inset.plot(ns, np.array(data['after_corr_incongr_ch_corr']) /
                      np.array(data['after_corr_congr_ch_corr']),
                      label='after correct (green-dash/green-solid)',
                      color=(.7, .7, .7))
        ax_inset.set_xlabel('Number of choices')
        ax_inset.set_ylabel('Ratio')
    pf.sv_fig(f=f, name='accuracies_NAFC', sv_folder=SV_FOLDER)


def win_stay_lose_swtich(file, ax, exp='sims_21', sel_vals=['16'], perf_th=0.6,
                         regressors=['L+', 'L-'], sel_files=None, x=0,
                         **plt_kwargs):
    """
    Plot reset index (RI) for networks.

    Parameters
    ----------
    main_folder : str
        where to find the data.
    sel_vals : list, optional
        max. num. of choices for which to plot the RI (['2', '4', '8', '16'])
    kernel_panel : list, optional
        position of perf. panel for reference ([0.37, 0.44])
    perf_th : float, optional
        threshold to filter networks by performance
        (note that perf. is w.r.t perfect integrator) (0.)
    **plt_kwargs : dict
        plot properties.

    Returns
    -------
    None.

    """
    plt_opts = {'lw': 1., 'alpha': 1., 'font': 8}
    plt_opts.update(plt_kwargs)
    # font = plt_opts['font']
    del plt_opts['font']
    f_tmp, ax_tmp = plt.subplots()
    data = np.load(file, allow_pickle=1)
    mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
    # transform vals to float and then back to str
    perf_mat = data['perf_mats']
    # filter experiments by performance
    mean_prf = [np.mean(p[-10:]) for p in perf_mat]
    glm_ac = filter_mat(mat=data['glm_mats_ac'], mean_prf=mean_prf,
                        perf_th=perf_th, val_mat=data['val_mat'],
                        sel_vals=sel_vals)
    glm_ae = filter_mat(mat=data['glm_mats_ae'], mean_prf=mean_prf,
                        perf_th=perf_th, val_mat=data['val_mat'],
                        sel_vals=sel_vals)
    # files = filter_mat(mat=data['files'], mean_prf=mean_prf, perf_th=perf_th,
    #                    val_mat=data['val_mat'], sel_vals=sel_vals)
    # sel_fs = files if sel_files is None else sel_files
    plt_opts_insts = deepc(plt_opts)
    plt_opts_insts['marker'] = ''
    ac_mat = []
    ae_mat = []
    for ind_glm, glm_ac_tmp in enumerate(glm_ac):
        if len(glm_ac_tmp) != 0:
            glm_ae_tmp = glm_ae[ind_glm]
            _, kernel_ac, _, _, _ = pf.plot_kernels(glm_ac_tmp, glm_ae_tmp,
                                                    ax=[ax_tmp], regressors=['L+'],
                                                    **plt_opts_insts)
            ac_mat.append(kernel_ac[0])
            _, _, kernel_ae, _, _ = pf.plot_kernels(glm_ac_tmp, glm_ae_tmp,
                                                    ax=[ax_tmp], regressors=['L-'],
                                                    **plt_opts_insts)
            ae_mat.append(kernel_ae[0])
    ac_mean = np.median(ac_mat)
    ac_mat = np.array(ac_mat)/ac_mean
    ae_mat = np.array(ae_mat)/ac_mean
    ax[1].scatter(x+0.01*np.random.rand(len(ac_mat)), ac_mat, color=hf.naranja)
    ax[1].scatter(x+0.01*np.random.rand(len(ac_mat)), ae_mat, color='k')
    pf.box_plot(data=ac_mat, ax=ax[1], x=x, lw=.5, fliersize=4, color=hf.naranja)
    pf.box_plot(data=ae_mat, ax=ax[1], x=x, lw=.5, fliersize=4, color='k')
    folder = '/home/molano/priors/rats/data_Ainhoa/'  # Ainhoa's data
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    prd.glm_krnls(main_folder=folder, tag='mat', x=0, ax_inset=ax_tmp,
                  axs_glm_krnls=[[ax_tmp, ax_tmp]], color=None, ax_wsls=ax[0],
                  name='', tags_mat=[['L+', 'L-']], plt_ind_trcs=False)
    plt.close(f_tmp)


def plot_biases_diff_seqs_evs(bias_seqs_evs, seqs_mat, ax, tau=2):
    t = np.array([0, 1, 2])
    exp = np.flip(np.exp(-t / tau))
    exp = exp/np.sum(exp)
    for biases_mat, seqs in zip(bias_seqs_evs, seqs_mat):
        for b, s in zip(biases_mat, seqs):
            reps_in_seq = np.diff(s) == 0
            rep_bias = np.sum(reps_in_seq*exp)
            if rep_bias > 0.5:
                color = hf.azul
                alpha = rep_bias**2
            else:
                color = hf.rojo
                alpha = (1-rep_bias)**2
            ax.plot(b[0], b[1], marker='.', color=color, alpha=alpha)
        ax.set_xlim([-2., 2])
        ax.set_ylim([-2., 2])
        ax.axhline(y=0, linestyle='--', color='k', lw=0.5)
        ax.axvline(x=0, linestyle='--', color='k', lw=0.5)
        ax.plot([-2, 2], [-2, 2], linestyle='--', color='k', lw=0.5)
        pf.rm_top_right_lines(ax)
        ax.set_xlabel('Bias after weak evidence')
        ax.set_ylabel('Bias after strong evidence')


# --- MAIN


if __name__ == '__main__':
    # f, ax = plt.subplots(nrows=1, ncols=2)
    # exp = 'sims_21'
    # main_folder = MAIN_FOLDER+'/'+exp
    # file = main_folder + '/data_ACER_test_2AFC_.npz'
    # win_stay_lose_swtich(file=file, ax=ax, sel_vals=['16'], x=1)
    # win_stay_lose_swtich(file=file, ax=ax, sel_vals=['2'], x=0)
    # pf.sv_fig(f=f, name='wsls', sv_folder=SV_FOLDER)
    # asd
    plt.close('all')
    fig1_krnls_RI = True  # rats kernels and RI
    fig1_psycho = True  # rats psycho-curves
    supp_fig_1_1 = True  # all experiments and all transitions kernels
    # supp_fig_1_2 = True  # all exps kernels
    fig2 = True
    supp_fig_2_1 = True  # all kernels for 2AFC RNN training
    supp_fig_2_2 = True  # psycho-curves and kernels for 2AFC task with no-blocks
    # all kernels for NAFC RNN training for diff n-ch
    fig3 = False
    supp_fig_3_1 = False  # theoretical accuracies
    fig4 = True
    supp_fig_4_0 = True  # all kernels for 16AFC RNN training
    supp_fig_4_2 = True  # pass-gt and diff-prob12
    # plot bias cond. on ev ~ 0 VS bias cond. on ev ~ 1
    # kernels for RNNs trained with longer trials
    supp_fig_4_4 = True  # 2AFC VS NAFC performance
    supp_fig_4_5 = True  # plot stim-transition interaction kernels
    supp_fig_4_8 = True  # invalid trials during training
    fig5 = True
    plot_rats_fig_5 = True
    supp_fig_5_1 = True  # plot transition mats for 6AFC test
    tests = False

    if tests:
        # PLOT EXP AND RNNS KERNELS TOGETHER
        ytckslbls = ['-1', '0', '-1']
        # RATS
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(7.5, 2.5))
        f_temp, ax_temp = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
        exp_sel = {'exps': ['/80_20/'], 'names': ['', '.8'], 'exp_for_krnls': '.8'}
        plot_exp_reset_index(main_folder='/home/molano/priors/rats',
                             ax_kernel_exp=ax[0], ax_RI_exp=ax_temp, **exp_sel)
        ylims = ax[0].get_ylim()
        ylims = [-ylims[1], ylims[1]]
        ax[0].set_ylim(ylims)
        yticks = [ylims[0], 0, ylims[1]]
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels(ytckslbls)
        ax[0].set_ylabel('Transition weight')
        plt.close(f_temp)

        # RNNS 2AFC
        exp = 'sims_21_biasCorr_extended'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        file = main_folder + '/data_ACER_test_2AFC_.npz'
        ax_kernel =\
            plot_kernels(file=file, sel_vals=['2'], ax_kernels=ax[1], perf_th=0.6,
                         **{'lw': 1.5})
        ax[1].axhline(y=0, linestyle='--', color='k', lw=0.5)
        ylims = ax[1].get_ylim()
        ylims = [-ylims[1], ylims[1]]
        ax[1].set_ylim(ylims)
        ax[1].set_ylabel('')
        yticks = [ylims[0], 0, ylims[1]]
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(ytckslbls)

        # RNNS 16-AFC
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        file = main_folder + '/data_ACER_test_2AFC_.npz'
        ax_kernel =\
            plot_kernels(file=file, sel_vals=['16'], ax_kernels=ax[2], perf_th=0.6,
                         **{'lw': 1.5})
        ax[2].axhline(y=0, linestyle='--', color='k', lw=0.5)
        ylims = ax[2].get_ylim()
        ylims = [-ylims[1], ylims[1]]
        ax[2].set_ylim(ylims)
        ax[2].set_ylabel('')
        yticks = [ylims[0], 0, ylims[1]]
        ax[2].set_yticks(yticks)
        ax[2].set_yticklabels(ytckslbls)

        pf.sv_fig(f=f, name='exp_rnns_krnls', sv_folder=SV_FOLDER)

    #################################
    # FIGURE 1
    #################################
    if fig1_krnls_RI:
        # PLOT EXPERIMENTAL RESULTS
        fig_exps()
    if fig1_psycho:
        plot_exp_psychoCurves(sv_folder=SV_FOLDER)
    if supp_fig_1_1:  # all kernels for rats
        font_krnls = 12
        margin_x = 0.07
        margin_y = 0.02
        fact = 0.7  # to reduce the size of panels
        f_kernel_exp, ax_kernel_exp = plt.subplots(nrows=4, ncols=3,
                                                   figsize=(10, 12))
        ax_kernel_exp[2, 2].set_axis_off()
        ax_kernel_exp[3, 2].set_axis_off()
        # PLOT ALL KERNELS FOR EXPERIMENTS
        ax_tmp = ax_kernel_exp[2:4, :2].flatten()
        letters = 'ghij'
        y_offset = [0.025, 0.025, 0, 0]
        for i_ax, ax in enumerate(ax_tmp):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0-y_offset[i_ax], pos.width*fact,
                             pos.height*fact])
            pf.add_letters(fig=f_kernel_exp, ax=ax, letter=letters[i_ax],
                           size=16, margin_x=margin_x, margin_y=margin_y)
        plot_all_trans_regr_exp(main_folder='/home/molano/priors/rats',
                                font=font_krnls, ax_kernel_exp=ax_tmp)
        # all exps kernels
        exps = ['/80_20/', '/95_5/', '/uncorrelated/',
                '/silent_80_20/', '/silent_95_05/', '']
        names = ['ILD-0.8', 'ILD-0.95', 'ILD-0.5', 'S0.8', 'S0.95', 'Freq.']
        f_temp, ax_temp = plt.subplots()
        # f_krnls, ax_krnls = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        ax_krnls = ax_kernel_exp[:2, :].flatten()
        letters = 'abcdef'
        for i_ax, (e, n) in enumerate(zip(exps, names)):
            plt.figure(f_kernel_exp.number)
            print(n)
            a = ax_krnls[i_ax]
            pos = a.get_position()
            a.set_position([pos.x0, pos.y0, pos.width*fact, pos.height*fact])
            # this is a hack I have to do because I treat differently ainhoa's
            # data in plot_exp_reset_index
            e_list = [e] if e != '' else []
            exps_selected = {'exps': e_list, 'names': ['', n],
                             'exp_for_krnls': n}
            plot_exp_reset_index(main_folder='/home/molano/priors/rats',
                                 font=font_krnls, ax_RI_exp=ax_temp,
                                 ax_kernel_exp=a, plt_io=False,
                                 ax_2d_plot=None, **exps_selected)
            a.set_title(n, fontsize=font_krnls)
            if i_ax not in [0, 3]:
                a.set_ylabel('')
            if i_ax < 3:
                a.set_ylim([-.5, 1])
                a.set_yticks([-.4, 0, .5, 1])
                a.set_yticklabels(['-0.4', '0.0', '0.5', '1.0'],
                                  fontsize=font_krnls)
                a.set_xlabel('')
            elif i_ax < 5:
                a.set_ylim([-.3, 2.2])
                a.set_yticks([0, 1, 2])
                a.set_yticklabels(['0.0', '1.0', '2.0'], fontsize=font_krnls)
            else:
                a.set_ylim([-.2, 1.2])
                a.set_yticks([0, 1])
                a.set_yticklabels(['0.0', '1.0'], fontsize=font_krnls)
            a.set_xticklabels(a.get_xticklabels(), fontsize=font_krnls)
            pf.add_letters(fig=f_kernel_exp, ax=a, letter=letters[i_ax],
                           size=16, margin_x=margin_x, margin_y=margin_y)

        f_kernel_exp.savefig(SV_FOLDER+'/merge_supps_fg_1_2_all_krnls.svg',
                             dpi=400, bbox_inches='tight')
        f_kernel_exp.savefig(SV_FOLDER+'/merge_supps_fg_1_2_all_krnls.png',
                             dpi=400, bbox_inches='tight')

    #################################
    # FIGURE 2
    #################################
    if fig2:
        # PLOT RESULTS FOR N = 2
        exp = 'sims_21_biasCorr_extended'  # sims_21_biasCorr
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        align = True
        name = 'fig_2_'+exp+'_RI_pp'
        print('N=2')
        # '200000000'])
        fig_N2(main_folder=main_folder, sel_vals=['2'], align=align,
               folder_examples='alg_ACER_seed_0_n_ch_2', y_lim=[0.7, 0.85],
               name=name, fldrs_psychCurves=['8000000', '16000000', '32000000',
                                             '200000000'])
    if supp_fig_2_1:  # all kernels for 2AFC RNN training
        num_ws = 3
        xtcks_ws = ['T++1', 'T+-1'] + ['T++'+str(x) for x in range(2, num_ws+1)]
        sel_val = '2'  # '16'
        font_lttrs = 12
        font = 10
        margin_x = 0.035
        margin_y = 0.01
        # PLOT RESET INDEX AND WEIGHTS ACROS TRAINING
        if sel_val == '2':
            exp = 'sims_21_biasCorr_extended/'
        else:
            exp = 'sims_21'  # 'sims_21_biasCorr_extended/'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        figsize = (10, 5)
        # PLOT NETWORKS PERFORMANCE ON 2AFC AND PSYCHO-CURVES
        f, ax = plt.subplots(ncols=4, nrows=2, figsize=figsize)
        margin = 0.15
        pos = ax[0, 0].get_position()
        ax_ri = plt.axes([pos.x0+margin, pos.y0, 2*pos.width-margin, pos.height])
        pf.add_letters(fig=f, ax=ax_ri, letter='a', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        pos = ax[1, 0].get_position()
        ax_ws = plt.axes([pos.x0+margin, pos.y0, 2*pos.width-margin, pos.height])
        pf.add_letters(fig=f, ax=ax_ws, letter='b', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        for a in ax[:, :2].flatten():
            a.remove()
        plt_opts = {'lw': 1., 'alpha': 1., 'color': 'k'}
        plot_ri_across_training(main_folder, ax=ax_ri, align=True, margin=100000,
                                ax_wgts_acr_tr=ax_ws, evs=None, num_ws=num_ws,
                                sel_val=sel_val, plot_ind_dots=True,
                                xtcks_ws=xtcks_ws, **plt_opts)
        # plot other weights
        # f_temp, ax_temp = plt.subplots()
        # xtcks_ws = ['T-+1', 'T--1'] + ['T++'+str(x) for x in range(2, num_ws+1)]
        # plot_ri_across_training(main_folder, ax=ax_temp,align=True,margin=100000,
        #                         ax_wgts_acr_tr=ax_ws, evs=None, num_ws=num_ws,
        #                         sel_val=sel_val, plot_ind_dots=True,
        #                         xtcks_ws=xtcks_ws, **plt_opts)
        if sel_val == '2':
            ax_ri.add_patch(Rectangle((0, -0.05), 2e5, 1.1, facecolor='none',
                                      edgecolor=(.5, 1, 1)))
            ax_ws.add_patch(Rectangle((0, -1.), 2e5, 3.5, facecolor='none',
                                      edgecolor=(.5, 1, 1)))
        ax_ri.set_xlabel('')
        ax_ri.set_ylabel('Reset Index', fontsize=font)
        ax_ws.set_xlim(ax_ri.get_xlim())
        ax_ws.set_xticks([0, 1.5e6])
        ax_ws.set_xticklabels(['0', '1.5'])
        pf.rm_top_right_lines(ax=ax_ws)
        ax_ws.set_ylabel('Weights', fontsize=font)
        ax_ws.set_xlabel('Number of trials (x $10^6$)', fontsize=font)
        # ax_ws.legend()
        # PLOT ALL KERNELS FOR N = 2 (AT MAX R.I.)
        plot_all_krnls(sel_vals=[sel_val], ylim=[-6, 6], ax=ax[:, 2:].flatten())
        letters = 'cdef'
        for i_a, a in enumerate(ax[:, 2:].flatten()):
            pf.add_letters(fig=f, ax=a, letter=letters[i_a], size=font_lttrs,
                           margin_x=margin_x, margin_y=margin_y)
            if i_a % 2 == 0:
                a.set_ylabel('Transition weight', fontsize=font)
            else:
                a.set_ylabel('')
            if i_a < 2:
                a.set_xlabel('')
            else:
                a.set_xlabel('Trial lag', fontsize=font)
        pf.sv_fig(f=f, name='supp_ws_traj_all_regrs_N_'+sel_val+'_from_python',
                  sv_folder=SV_FOLDER)

    if supp_fig_2_2:  # psycho-curves and kernels for 2AFC task with no-blocks
        font = 12
        main_folder = MAIN_FOLDER + 'sims_21_2AFC_noblocks/'
        f, ax_main = plt.subplots(figsize=(8, 5), ncols=3, nrows=2)
        ax_main = ax_main.flatten()
        # plot performance from main experiment
        # plot_nets_perf(main_folder=MAIN_FOLDER + 'sims_21/', sel_vals=['2'],
        #                perf_th=0.6, plt_zoom=False, ax_perfs=ax[0],
        #                plot_mean=False, **{'colors': np.zeros((20, 3))})
        # # plot performances for no-blocks RNNs
        # plot_nets_perf(main_folder=main_folder, sel_vals=['2'], perf_th=0.6,
        #                plt_zoom=False, ax_perfs=ax[0], plot_pi=False,
        #                plot_mean=False,
        #                **{'colors': np.zeros((20, 3))+np.array((0, 1, 1))})
        # ax[0].set_xlim([0, 55e4])
        pnl_fctr = 0.8
        margin = 0.02
        pos = ax_main[0].get_position()
        ax_main[0].set_axis_off()
        psych_panel = [pos.x0+margin, pos.y0+margin, pos.width*pnl_fctr,
                       pos.height*pnl_fctr]
        # here we analyze the behavioral data directly from the testing,
        # but we gave it the address:
        # alg_ACER_seed_0_n_ch_2/test_2AFC_all/_model_40000000_steps/
        # because it is what plot_psycho_curves expects
        axs_psy = plot_psycho_curves(main_folder=main_folder,
                                     psych_panel=psych_panel,
                                     folder_examples='alg_ACER_seed_0_n_ch_2',
                                     font_inset=8, fldrs_psychCurves=['40000000'])
        axs_psy[0].set_ylabel('Repeating probability', fontsize=8)
        pf.add_letters(fig=f, ax=axs_psy[0], letter='a', size=font, margin_x=0.06,
                       margin_y=0.02)

        # plot kernels
        file = main_folder + '/data_ACER*n_ch_2*_test_2AFC_.npz'
        sel_vals = ['2']
        ax_kernel = plot_kernels(file=file, sel_vals=sel_vals,
                                 ax_kernels=ax_main[3])
        ax_kernel.set_yticks([-1, 0, 1, 2])
        pos = ax_kernel.get_position()
        ax_kernel.set_position([pos.x0+margin, pos.y0+margin, pos.width*pnl_fctr,
                                pos.height*pnl_fctr])

        pf.add_letters(fig=f, ax=ax_kernel, letter='b', size=font, margin_x=0.06,
                       margin_y=0.02)

        # pf.sv_fig(f=f, name='psychoC_2AFC_nets_noblocks', sv_folder=SV_FOLDER)
        # PLOT ALL KERNELS FOR N = 2, 4, 8, 16 (AT END OF TRAINING)
        ax = np.concatenate((ax_main[1:3], ax_main[4:6]))
        ylim = [-2.2, 2.5]
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        file = main_folder+'data_ACER_test_2AFC_.npz'
        f_temp, ax_temp = plt.subplots()
        pos = ax_temp.get_position()
        sel_vals_mat = ['2', '4', '8', '12', '16']
        regressors = ['T++', 'T-+', 'T+-', 'T--']
        alpha_mat = np.linspace(0.15, 1, len(sel_vals_mat))
        lw_mat = np.linspace(0.5, 1., len(sel_vals_mat))
        letters = 'cdef'
        pnl_fctr = 0.97
        sep = 0.01
        for i_sv, s_v in enumerate(sel_vals_mat):
            plt_kwargs = {'alpha': alpha_mat[i_sv], 'lw': lw_mat[i_sv]}
            for i_a, (a, r) in enumerate(zip(ax, regressors)):
                pos = a.get_position()
                a.set_position([pos.x0+sep, pos.y0, pos.width*pnl_fctr,
                                pos.height*pnl_fctr])
                a.plot([1, 6], [0, 0], '--k', lw=0.5)
                plot_kernels(file=file,  sel_vals=[s_v], ax_kernels=a,
                             perf_th=0.6, regressors=[r], ax_krnls_ind_tr=pos,
                             **plt_kwargs)
                a.set_ylim(ylim)
                if i_a in [1, 3]:
                    a.set_ylabel('')
                    a.set_yticks([])
                if i_a in [0, 1]:
                    a.set_xlabel('')
                if i_sv == 3:
                    pf.add_letters(fig=f, ax=a, letter=letters[i_a], size=font,
                                   margin_x=0.05-(0.03*(i_a % 2)), margin_y=0)
                    plt.figure(f_temp.number)
                    a.set_title(r)
        # plt.close(f_temp)
        plt.figure(f.number)
        pos = ax[3].get_position()
        ax_clbr = plt.axes([pos.x0+pos.width+.01, pos.y0, 0.02, pos.height/2])
        pf.add_grad_colorbar(ax=ax_clbr, yticks=[2, 16])
        # ax_clbr.set_title('Nmax', fontsize=6)
        pf.sv_fig(f=f, name='supp_noblocks_all_kernels_all_nch',
                  sv_folder=SV_FOLDER)

    #################################
    # FIGURE 3
    #################################
    if fig3:
        fig_theoretical_trans_matrices()
    if supp_fig_3_1:  # theoretical accuracies
        fig_theoretical_accuracies_NAFC()
    #################################
    # FIGURE 4
    #################################
    if fig4:
        # PLOT RESULTS FOR N = 16
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        name = 'fig_4_'+exp+'_RI_pp'
        fig_N16(main_folder=main_folder, sel_vals=['16'], y_lim=[0.5, 0.85],
                folder_examples='alg_ACER_seed_1_n_ch_16', name=name,
                binning=300000, fldrs_psychCurves=['4000000', '20000000',
                                                   '164000000'])
    if supp_fig_4_0:  # all kernels for 16AFC RNN training
        # PLOT ALL KERNELS FOR N = 16 (AT END OF TRAINING)
        ylim = [-2.2, 3.5]
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        file = main_folder+'data_ACER_test_2AFC_.npz'
        sel_vals_mat = ['16']
        regressors = ['T++', 'T-+', 'T+-', 'T--']
        letters = 'abcd'
        font = 10
        for i_sv, s_v in enumerate(sel_vals_mat):
            f, ax = plt.subplots(nrows=2, ncols=2, figsize=(4, 4))
            ax = ax.flatten()
            for i_a, (a, r) in enumerate(zip(ax, regressors)):
                a.plot([1, 6], [0, 0], '--k', lw=0.5)
                plot_kernels(file=file,  sel_vals=[s_v], ax_kernels=a,
                             perf_th=0.6, regressors=[r])
                a.set_ylim(ylim)
                if i_a in [1, 3]:
                    a.set_ylabel('')
                    a.set_yticks([])
                if i_a in [0, 1]:
                    a.set_xlabel('')
                if i_sv == 0:
                    pf.add_letters(fig=f, ax=a, letter=letters[i_a], size=font,
                                   margin_x=0.07, margin_y=0)
            pf.sv_fig(f=f, name='supp_all_kernels_nch_'+s_v, sv_folder=SV_FOLDER)

    if supp_fig_4_2:
        # pass-gt and diff-prob12
        font_lttrs = 18
        margin_x = 0.07
        margin_y = 0.02
        perf_th = 0.6
        pnl_wdth = 0.25
        pnl_hgt = 0.3
        fig_size = (6, 9)
        bottom_margin = 0.03
        panel_w_steps = [.1, .6, .1, .6]
        panel_h_steps = [.65, .31, .585]
        f = plt.figure(figsize=fig_size)
        ax_pass_gt = plt.axes([panel_w_steps[0], panel_h_steps[0], .3, .3])
        pf.add_letters(fig=f, ax=ax_pass_gt, letter='a', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        ax_perf = plt.axes([panel_w_steps[1], panel_h_steps[0]-.05, .35, .13])
        pf.add_letters(fig=f, ax=ax_perf, letter='c', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        ax_RI = plt.axes([panel_w_steps[1], panel_h_steps[0]+.17, .35, .13])
        pf.add_letters(fig=f, ax=ax_RI, letter='b', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        pnl_wdth = 0.27
        pnl_hgt = pnl_wdth*fig_size[0]/fig_size[1]
        ax_seqs_1 = plt.axes([panel_w_steps[2], panel_h_steps[1], pnl_wdth,
                              pnl_hgt])
        ax_seqs_2 = plt.axes([panel_w_steps[3], panel_h_steps[1], pnl_wdth,
                              pnl_hgt])
        pf.add_letters(fig=f, ax=ax_seqs_1, letter='d', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        pf.add_letters(fig=f, ax=ax_seqs_2, letter='e', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        pnl_wdth = 0.35
        pnl_hgt = 0.2
        ax_long_tr_1 = plt.axes([panel_w_steps[2], bottom_margin, pnl_wdth,
                                 pnl_hgt])
        ax_long_tr_clbr = plt.axes([panel_w_steps[2]+pnl_wdth, bottom_margin,
                                    pnl_wdth/10, pnl_hgt/2])
        ax_long_tr_2 = plt.axes([panel_w_steps[3], bottom_margin, pnl_wdth,
                                 pnl_hgt])
        pf.add_letters(fig=f, ax=ax_long_tr_1, letter='f', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        pf.add_letters(fig=f, ax=ax_long_tr_2, letter='g', size=font_lttrs,
                       margin_x=margin_x, margin_y=margin_y)
        # PLOT RESET INDEX FOR DIFFERENT PROB-12
        bxp_ops = {'widths': 0.25, 'lw': 1}
        sel_vals = ['2']
        exp = 'sims_21_pass_gt'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        plot_reset_index_diff_vals(main_folder=main_folder, sel_vals=sel_vals,
                                   perf_th=0.6, ax_RI_diff_nch=ax_pass_gt,
                                   bxp_ops=bxp_ops, plt_ind_vals=False)
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        sel_vals = ['16']
        plot_reset_index_diff_vals(main_folder=main_folder, sel_vals=sel_vals,
                                   perf_th=0.6, ax_RI_diff_nch=ax_pass_gt,
                                   offset_xs=1, bxp_ops=bxp_ops,
                                   plt_ind_vals=False)
        sel_vals = ['16']
        exp = 'sims_21_pass_gt'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        plot_reset_index_diff_vals(main_folder=main_folder, sel_vals=sel_vals,
                                   perf_th=0.6, ax_RI_diff_nch=ax_pass_gt,
                                   offset_xs=2, bxp_ops=bxp_ops,
                                   plt_ind_vals=False)

        ax_pass_gt.set_xlabel('Training protocol')
        ax_pass_gt.set_xticks(np.arange(3))
        ax_pass_gt.set_yticks(np.array([0, 0.5, 1]))
        ax_pass_gt.set_yticklabels(['0', '0.5', '1'])
        ax_pass_gt.set_xticklabels(['2AFC\ntrained', 'NAFC\npre-trained',
                                    'NAFC\nwith info'])

        # PLOT REFERENCE PERFORMANCE FROM NETS TRAINED DIRECTLY ON 2AFC TASK
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        file = main_folder + '/data_ACER*n_ch_2*__bin_20K.npz'
        data = np.load(file, allow_pickle=1)
        ax_tw = ax_perf.twiny()
        # filter experiments by performance and max number of choices
        plot_nets_N_cond_perf(main_folder, sel_vals=['2'], binning=20000,
                              ax_perfs=ax_tw, n_mat=[2],  # perc_tr=1/275e4,
                              plot_chance_ref=False, plt_clrbar=False,
                              perf_th=perf_th, file=file, alpha=0.5)
        color_axis = sns.color_palette("mako", n_colors=1)
        ax_perf.spines['top'].set_color(color_axis[0])
        ax_tw.set_xlim([1.5e3, 3.1e6])
        # ax_tw.spines['bottom'].set_color(color_axis[0])
        # asd
        ax_tw.set_xscale('log')
        colors = np.array([[127, 201, 127], [190, 174, 212], [253, 192, 134]])/255
        separate_rep_alt = False
        bxp_ops = {'widths': 0.25, 'color': colors[0], 'lw': 0.8}
        # # PLOT RESET INDEX FOR DIFFERENT PROB-12
        sel_vals = ['0.001', '0.0025', '0.01', '0.025', '0.05', '0.1', '0.25',
                    '0.5', '1.0']
        exp = 'sims_21_diff_prob12'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        file = main_folder+'data_ACER_test_2AFC_sims_21_n_ch_16.npz'
        plot_reset_index_diff_vals(main_folder=main_folder, sel_vals=sel_vals,
                                   perf_th=perf_th, ax_RI_diff_nch=ax_RI,
                                   file=file, eq_dist_xs=False, bxp_ops=bxp_ops,
                                   connect_values=True, plt_ind_vals=False)
        ax_RI.set_xscale('log')
        ax_RI.set_xticks([])
        bxp_ops['color'] = colors[0]
        plot_perf_diff_vals(main_folder=main_folder, sel_vals=sel_vals,
                            perf_th=perf_th, ax_prf_diff_nch=ax_perf, m_or_bx='bx',
                            eq_dist_xs=False, file=file, bxp_ops=bxp_ops,
                            separate_rep_alt=separate_rep_alt,
                            connect_values=True, plt_ind_vals=False)
        exp = 'sims_21_rand_pretr'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        sel_vals = ['0.01', '0.05', '0.1', '0.25', '0.5']
        bxp_ops['color'] = colors[1]
        plot_perf_diff_vals(main_folder=main_folder, sel_vals=sel_vals,
                            perf_th=perf_th, ax_prf_diff_nch=ax_perf, m_or_bx='bx',
                            eq_dist_xs=False, bxp_ops=bxp_ops,
                            separate_rep_alt=separate_rep_alt,
                            connect_values=True, plt_ind_vals=False)
        ax_perf.set_xscale('log')
        # sel_vals = ['0.25', '0.5']
        # bxp_ops['color'] = colors[1]
        # plot_reset_index_diff_vals(main_folder=main_folder, sel_vals=sel_vals,
        #                            perf_th=perf_th, ax_RI_diff_nch=ax[0],
        #                            eq_dist_xs=False, bxp_ops=bxp_ops,
        #                            connect_values=True)
        # ax[0].set_xscale('log')
        for a in [ax_perf, ax_RI]:
            a.set_xlim([0.0009, 1.2])
            a.set_xticks(np.array([0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.]))
            a.set_xticklabels(np.array(['0.1', '1', '5', '10', '25', '50',
                                        '100']))
        ax_perf.set_xlabel('Percentage of 2AFC trials\nduring pre-training')
        ax_perf.set_ylim([0.4, .75])
        ax_RI.set_ylim([-.05, 1.05])
        ax_RI.set_yticks(np.array([0, 0.5, 1]))
        ax_RI.set_yticklabels(['0', '0.5', '1'])
        ax_perf.set_yticks(np.array([0.5, 0.6, 0.7]))  # +[exp_x])
        ax_perf.set_yticklabels(['0.5', '0.6', '0.7'])
        # pf.sv_fig(f=f, name='supp_RI_diff_prob_12', sv_folder=SV_FOLDER)
        # plot bias cond. on ev ~ 0 VS bias cond. on ev ~ 1
        main_folder = MAIN_FOLDER + 'sims_21/'
        file = main_folder+'data_ACER*n_ch_16*_test_2AFC_150222_bias_seqs_evs.npz'
        sel_vals = ['16']
        data = np.load(file, allow_pickle=1)
        # f, ax = plt.subplots(figsize=(4, 4), ncols=2, nrows=2)
        # ax = ax.flatten()
        mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
        bias_seqs_evs = filter_mat(mat=data['bias_seqs_evs'], mean_prf=mean_prf,
                                   val_mat=data['val_mat'], sel_vals=sel_vals,
                                   perf_th=0.6)
        seqs_mat = filter_mat(mat=data['seqs'], mean_prf=mean_prf,
                              val_mat=data['val_mat'], sel_vals=sel_vals,
                              perf_th=0.6)
        plot_biases_diff_seqs_evs(bias_seqs_evs=bias_seqs_evs,
                                  seqs_mat=seqs_mat, ax=ax_seqs_1)
        # biases_mat = np.array(biases_mat)
        # colors = np.array(colors)
        # plot rats biases
        bias_seqs_evs, seqs = prd.bias_diff_seqs_evs(verbose=True)
        assert (seqs_mat[0] == seqs).all()
        plot_biases_diff_seqs_evs(bias_seqs_evs=bias_seqs_evs,
                                  seqs_mat=seqs_mat, ax=ax_seqs_2)
        ax_seqs_1.set_ylabel('')
        # pf.sv_fig(f=f, name='bias_seqs_evs', sv_folder=SV_FOLDER)

        # kernels for RNNs trained with longer trials
        main_folder = MAIN_FOLDER + 'sims_21_longer/'
        # f, ax = plt.subplots(figsize=(3.5, 1.5), ncols=2)
        # plot_nets_perf(main_folder=main_folder, sel_vals=['16'], perf_th=0.6,
        #                plt_zoom=False, ax_perfs=ax[0], y_lim=[0.2, 0.85])
        plot_nets_N_cond_perf(file=main_folder+'data_ACER__.npz',
                              main_folder=main_folder, sel_vals=['16'],
                              ax_perfs=ax_long_tr_1, ax_clbr=ax_long_tr_clbr)
        # plot kernels
        file = main_folder + '/data_ACER_test_2AFC_.npz'
        sel_vals = ['16']
        ax_kernel = plot_kernels(file=file, sel_vals=sel_vals,
                                 ax_kernels=ax_long_tr_2)
        ax_kernel.set_yticks([-1, 0, 1, 2])
        ax_kernel.axhline(y=0, color='k', lw=0.5, linestyle='--')
        pf.sv_fig(f=f, name='controls', sv_folder=SV_FOLDER)

    if supp_fig_4_4:  # 2AFC VS NAFC performance
        binning = 200000
        spacing = 10000
        perc_tr = 0.01
        plt_opts = {'lw': 1., 'alpha': .5}

        # PLOT PERFORMANCES AT TEST TIME FOR 2AFC VS NAFC NETWORKS
        exp = 'sims_21_biasCorr_extended'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
        ax_perfs = plot_nets_perf(main_folder=main_folder, sel_vals='2',
                                  plt_zoom=False, ax_perfs=ax, binning=10000,
                                  **plt_opts)
        # performance at end of pre-training
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        # get pre-training durations
        file = main_folder + '/data_ACER__.npz'
        data = np.load(file, allow_pickle=1)
        mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
        perfs = filter_mat(mat=data['perf_mats'], mean_prf=mean_prf,
                           perf_th=-100, val_mat=data['val_mat'],
                           sel_vals='16')
        tr_durs = [perc_tr*spacing*len(x) for x in perfs]
        mean_training = np.mean(tr_durs)
        std_training = np.std(tr_durs)  # /np.sqrt(len(tr_durs))

        # get performances at testing time
        file = main_folder + '/data_ACER_test_2AFC_.npz'
        data = np.load(file, allow_pickle=1)
        # filter experiments by performance
        # mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
        perfs = filter_mat(mat=data['perf_mats'], mean_prf=mean_prf,
                           perf_th=-100, val_mat=data['val_mat'],
                           sel_vals='16')
        mean_prfs = [np.mean(p[-10:]) for p in perfs]
        mean_perf = np.mean(mean_prfs)
        std_perf = np.std(mean_prfs)  # /np.sqrt(len(mean_prfs))
        ax_perfs.errorbar(x=mean_training, y=mean_perf, xerr=std_training,
                          yerr=std_perf, marker='.', markersize=6, color='k')
        ax_perfs.set_xlim([-1e4, 1e6])
        ax_perfs.set_xticks([0, 25e4, 5e5, 75e4, 1e6])
        ax_perfs.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        pf.sv_fig(f=f, name='supp_perf_comparison', sv_folder=SV_FOLDER)

    if supp_fig_4_5:  # plot stim-transition interaction kernels
        ylim = [-2.2, 3.5]
        # PLOT STIM-TRANSITION INTERACTION KERNELS
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        sel_vals_mat = ['2', '16']
        regressors = ['ev-T++', 'T++']
        titles = ['Evidence-Transition\ninteraction', 'T++ weights']
        font = 10
        letters = 'ab'
        f, ax_all = plt.subplots(nrows=2, ncols=3, figsize=(5, 4))
        for i_sv, s_v in enumerate(sel_vals_mat):
            ax = ax_all[i_sv, :]
            file = main_folder+'data_ACER*n_ch_'+s_v +\
                '*_test_2AFC_150222_ev_tr_int.npz'
            for i_a, (a, r) in enumerate(zip(ax, regressors)):
                a.plot([1, 6], [0, 0], '--k', lw=0.5)
                plot_kernels(file=file,  sel_vals=[s_v], ax_kernels=a,
                             perf_th=0.6, regressors=[r])
                a.set_ylim(ylim)
                a.set_title(titles[i_a]+' (n-ch = '+s_v+')')
                if i_a in [1, 3]:
                    a.set_ylabel('')
                if i_sv == 0:
                    pf.add_letters(fig=f, ax=a, letter=letters[i_a], size=font,
                                   margin_x=0.07, margin_y=0)
        ax_krnl = ax_all[0, 2]
        f_tmp, ax_tmp = plt.subplots()
        folder = '/home/molano/priors/rats/data_Ainhoa/'  # Ainhoa's data
        prd.glm_krnls(main_folder=folder, tag='mat', x=0, ax_inset=ax_tmp,
                      axs_glm_krnls=[[ax_krnl]], color=None, name='Freq.',
                      tags_mat=[['ev-T++']], plt_ind_trcs=True)
        plt.close(f_tmp)
        pf.sv_fig(f=f, name='supp_stim_tr_int_kernels', sv_folder=SV_FOLDER)

    if supp_fig_4_8:  # invalid trials during training
        file = MAIN_FOLDER+'/sims_21/data_ACER*n_ch_16__invalids.npz'
        data = np.load(file, allow_pickle=1)
        mean_prf = [np.mean(p[-10:]) for p in data['perf_mats']]
        perfs_cond = filter_mat(mat=data['perfs_cond'], mean_prf=mean_prf,
                                perf_th=-100, val_mat=data['val_mat'],
                                sel_vals=['16'])
        files = filter_mat(mat=data['files'], mean_prf=mean_prf,
                           perf_th=-100, val_mat=data['val_mat'],
                           sel_vals=['16'])

        colors = sns.color_palette("mako", n_colors=15)
        f, ax = plt.subplots(ncols=5, nrows=3)
        ax = ax.flatten()
        mat_invs = []
        for n in range(2, 16):
            print(n)
            mat = []
            for pc, f in zip(perfs_cond, files):
                ax[n-2].plot(pc['inv_'+str(n)], color='k', alpha=0.1, lw=0.5)
                mat.append(pc['inv_'+str(n)])
                if np.max(pc['inv_'+str(n)]) > 0.9:
                    print('--------')
                    print(np.mean(pc['inv_'+str(n)]))
                    print(f)
            max_l = np.max([len(x) for x in mat])
            mat = [list(x)+[np.nan]*(max_l-len(x)) for x in mat]
            mat = np.array(mat)
            mean_tr = np.nanmean(mat, axis=0)
            ax[n-2].plot(mean_tr, color='k')
            if n == 2:
                lims = [int(1e4), int(3e4)]
            else:
                lims = [int(5e4), int(15e4)]
            mean_inv = np.mean(mean_tr[lims[0]:lims[1]])
            ax[n-2].set_title(str(n)+' mean = '+str(np.round(mean_inv, 3)))
            mat_invs.append(mean_inv)
        f, ax = plt.subplots(figsize=(3, 2))
        ax.plot(np.arange(2, 16), mat_invs)
        ax.plot(np.arange(2, 16), (16-np.arange(2, 16))/16, color=(.5, .5, .5))
        ax.set_xticks(np.arange(2, 16))
        ax.set_xlabel('N')
        ax.set_ylabel('Proportion of invalid (choice > N) trials')
        pf.rm_top_right_lines(ax)
        f.savefig(SV_FOLDER+'/prop_invalids_from_python.svg', dpi=400,
                  bbox_inches='tight')
        f.savefig(SV_FOLDER+'/prop_invalids_from_python.png', dpi=400,
                  bbox_inches='tight')

    #################################
    # FIGURE 5
    #################################
    if fig5:
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        if plot_rats_fig_5:
            fig_rebound(folder_nets=main_folder)
        aux = zt(tau=200, window=500, resp_lat=100)
        aux1 = zt(tau=200, window=500, resp_lat=200)
        zt = aux+aux1
        f, ax = plt.subplots(figsize=(10, 1))
        plt.plot(zt, color='k')
        pf.rm_top_right_lines(ax=ax)
        pf.sv_fig(f=f, name='zt', sv_folder=SV_FOLDER)
    if supp_fig_5_1:  # plot transition mats for 6AFC test
        # PLOT TRANSITION BIAS MATS
        exp = 'sims_21'
        main_folder = MAIN_FOLDER+'/'+exp+'/'
        file = main_folder+'/data_ACER*n_ch_16_test_nch_6_.npz'
        n_ch = 6
        fig_trans_mats(file=file, n_ch=n_ch)
