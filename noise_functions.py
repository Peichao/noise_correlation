import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.markers import TICKDOWN


def get_chans(vis_info):
    vis_index = pd.DataFrame(columns=['ind1', 'ind2', 'channel', 'cluster'])
    counter = 0
    for ind1, ch in enumerate(vis_info):
        try:
            chan = vis_info[ind1].ch_label
        except AttributeError:
            chan = vis_info[ind1][0].ch_label
        inds = []
        try:
            for ind2, unit in enumerate(ch):
                try:
                    clust = vis_info[ind1][ind2].cluster
                    inds = np.append(inds, clust)
                    vis_index.loc[counter] = [ind1, ind2, chan, clust]
                    counter += 1
                except AttributeError:
                    pass
        except TypeError:
            clust = vis_info[ind1].cluster
            inds = np.append(inds, clust)
            vis_index.loc[counter] = [ind1, 0, chan, clust]
            counter += 1
    return vis_index


def main_plots(data_path, vstim, all_stat_coeff, all_run_coeff, pv_list, pyr_list):
    plt.ioff()

    plt.boxplot([all_stat_coeff['corr_coefficient'], all_run_coeff['corr_coefficient']])
    plt.xticks([1, 2], ['Stationary', 'Running'])
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('All Pairs')
    plt.savefig(data_path + 'figures/' + 'movement_boxplot_all.pdf', format='pdf')
    plt.close()

    pv_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(pv_list)) & (all_stat_coeff['Column'].isin(pv_list))]
    pv_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(pv_list)) & (all_run_coeff['Column'].isin(pv_list))]
    plt.boxplot([pv_stat_coeff['corr_coefficient'], pv_run_coeff['corr_coefficient']])
    plt.xticks([1, 2], ['Stationary', 'Running'])
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('All PV/PV Pairs')
    plt.savefig(data_path + 'figures/' + 'movement_boxplot_pv.pdf', format='pdf')
    plt.close()

    pyr_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(pyr_list)) & (all_stat_coeff['Column'].isin(pyr_list))]
    pyr_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(pyr_list)) & (all_run_coeff['Column'].isin(pyr_list))]
    plt.boxplot([pyr_stat_coeff['corr_coefficient'], pyr_run_coeff['corr_coefficient']])
    plt.xticks([1, 2], ['Stationary', 'Running'])
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('All non-PV/non-PV Pairs')
    plt.savefig(data_path + 'figures/' + 'movement_boxplot_pyr.pdf', format='pdf')
    plt.close()

    mix_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(pyr_list)) & (all_stat_coeff['Column'].isin(pv_list)) |
                                    (all_stat_coeff['Row'].isin(pv_list)) & (all_stat_coeff['Column'].isin(pyr_list))]
    mix_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(pyr_list)) & (all_run_coeff['Column'].isin(pv_list)) |
                                  (all_run_coeff['Row'].isin(pv_list)) & (all_run_coeff['Column'].isin(pyr_list))]
    plt.boxplot([mix_stat_coeff['corr_coefficient'], mix_run_coeff['corr_coefficient']])
    plt.xticks([1, 2], ['Stationary', 'Running'])
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('All PV/non-PV Pairs')
    plt.savefig(data_path + 'figures/' + 'movement_boxplot_mix.pdf', format='pdf')
    plt.close()

    p_vals = np.array([sp.stats.wilcoxon(all_stat_coeff['corr_coefficient'], all_run_coeff['corr_coefficient']),
                       sp.stats.wilcoxon(pv_stat_coeff['corr_coefficient'], pv_run_coeff['corr_coefficient']),
                       sp.stats.wilcoxon(pyr_stat_coeff['corr_coefficient'], pyr_run_coeff['corr_coefficient']),
                       sp.stats.wilcoxon(mix_stat_coeff['corr_coefficient'], mix_run_coeff['corr_coefficient'])])

    all_coeff = pd.concat([all_stat_coeff['corr_coefficient'],
                           all_run_coeff['corr_coefficient']], axis=1)
    all_coeff.columns = ['Stationary', 'Running']
    all_pv_coeff = pd.concat([pv_stat_coeff['corr_coefficient'],
                              pv_run_coeff['corr_coefficient']], axis=1)
    all_pv_coeff.columns = ['Stationary', 'Running']
    all_pyr_coeff = pd.concat([pyr_stat_coeff['corr_coefficient'],
                               pyr_run_coeff['corr_coefficient']], axis=1)
    all_pyr_coeff.columns = ['Stationary', 'Running']
    all_mix_coeff = pd.concat([mix_stat_coeff['corr_coefficient'],
                               mix_run_coeff['corr_coefficient']], axis=1)
    all_mix_coeff.columns = ['Stationary', 'Running']

    run_means = np.array([all_coeff['Running'].mean(),
                          all_pv_coeff['Running'].mean(),
                          all_pyr_coeff['Running'].mean(),
                          all_mix_coeff['Running'].mean()])

    run_std = np.array([all_coeff['Running'].std() / np.sqrt(len(all_run_coeff)),
                        all_pv_coeff['Running'].std() / np.sqrt(len(pv_run_coeff)),
                        all_pyr_coeff['Running'].std() / np.sqrt(len(pyr_run_coeff)),
                        all_mix_coeff['Running'].std() / np.sqrt(len(mix_run_coeff))])

    stat_means = np.array([all_coeff['Stationary'].mean(),
                           all_pv_coeff['Stationary'].mean(),
                           all_pyr_coeff['Stationary'].mean(),
                           all_mix_coeff['Stationary'].mean()])

    stat_std = np.array([all_coeff['Stationary'].std() / np.sqrt(len(all_stat_coeff)),
                         all_pv_coeff['Stationary'].std() / np.sqrt(len(pv_stat_coeff)),
                         all_pyr_coeff['Stationary'].std() / np.sqrt(len(pyr_stat_coeff)),
                         all_mix_coeff['Stationary'].std() / np.sqrt(len(pyr_stat_coeff))])

    ind = np.arange(len(run_means))
    width = 0.25

    fig, ax = plt.subplots()
    stat_bar = ax.bar(ind, stat_means, width, color='#348ABD', yerr=stat_std, ecolor='k')
    run_bar = ax.bar(ind + width, run_means, width, color='#E24A33', yerr=run_std, ecolor='k')

    y_lim = ax.get_ylim()
    offset = (y_lim[1] - y_lim[0]) / 5

    for i, p in enumerate(p_vals):
        if p[1] >= 0.05:
            display_string = r'n.s.'
        elif p[1] < 0.001:
            display_string = r'***'
        elif p[1] < 0.01:
            display_string = r'**'
        else:
            display_string = r'*'

        height = offset + stat_means.max()
        bar_centers = ind[i] + np.array([0.5, 1.5]) * width
        significance_bar(bar_centers[0], bar_centers[1], height, display_string)

    ax.legend((stat_bar[0], run_bar[0]), ('Stationary', 'Running'), loc=3)

    if vstim == 'y':
        stim_str = 'Visual Stimulus'
    else:
        stim_str = 'No Stimulus'

    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('All Units', 'PV/PV', 'non-PV/non-PV', 'non-PV/PV'))

    plt.savefig(data_path + 'figures/movement_bar.pdf', format='pdf')
    plt.close()

    f1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    all_coeff.plot.hist(alpha=0.5, ax=ax1)
    ax1.set_title('All Units')
    all_pv_coeff.plot.hist(alpha=0.5, ax=ax2)
    ax2.set_title('PV/PV')
    all_pyr_coeff.plot.hist(alpha=0.5, ax=ax3)
    ax3.set_title('non-PV/non-PV')
    all_mix_coeff.plot.hist(alpha=0.5, ax=ax3)
    ax3.set_title('non-PV/PV')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(data_path + 'figures/movement_hist.pdf', format='pdf')

    plt.close()
    plt.ion()

    return all_coeff, all_pv_coeff, all_pyr_coeff, all_mix_coeff


def layer_plots(data_path, vstim, all_stat_coeff, all_run_coeff, supra_list, gran_list, infra_list):
    # layer plots
    plt.ioff()
    supra_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(supra_list)) &
                                      (all_stat_coeff['Column'].isin(supra_list))]
    supra_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(supra_list)) &
                                    (all_run_coeff['Column'].isin(supra_list))]

    gran_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(gran_list)) &
                                     (all_stat_coeff['Column'].isin(gran_list))]
    gran_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(gran_list)) &
                                   (all_run_coeff['Column'].isin(gran_list))]

    infra_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(infra_list)) &
                                      (all_stat_coeff['Column'].isin(infra_list))]
    infra_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(infra_list)) &
                                    (all_run_coeff['Column'].isin(infra_list))]

    all_supra_coeff = pd.concat([supra_stat_coeff['corr_coefficient'],
                                 supra_run_coeff['corr_coefficient']], axis=1)
    all_supra_coeff.columns = ['Stationary', 'Running']
    all_gran_coeff = pd.concat([gran_stat_coeff['corr_coefficient'],
                                gran_run_coeff['corr_coefficient']], axis=1)
    all_gran_coeff.columns = ['Stationary', 'Running']
    all_infra_coeff = pd.concat([infra_stat_coeff['corr_coefficient'],
                                 infra_run_coeff['corr_coefficient']], axis=1)
    all_infra_coeff.columns = ['Stationary', 'Running']

    layer_run_means = np.array([all_supra_coeff['Running'].mean(),
                                all_gran_coeff['Running'].mean(),
                                all_infra_coeff['Running'].mean()])

    layer_run_std = np.array([all_supra_coeff['Running'].std() / np.sqrt(len(supra_stat_coeff)),
                              all_gran_coeff['Running'].std() / np.sqrt(len(gran_stat_coeff)),
                              all_infra_coeff['Running'].std() / np.sqrt(len(infra_stat_coeff))])

    layer_stat_means = np.array([all_supra_coeff['Stationary'].mean(),
                                 all_gran_coeff['Stationary'].mean(),
                                 all_infra_coeff['Stationary'].mean()])

    layer_stat_std = np.array([all_supra_coeff['Stationary'].std() / np.sqrt(len(supra_stat_coeff)),
                               all_gran_coeff['Stationary'].std() / np.sqrt(len(gran_stat_coeff)),
                               all_infra_coeff['Stationary'].std() / np.sqrt(len(infra_stat_coeff))])

    p_vals = np.array([sp.stats.wilcoxon(all_supra_coeff['Stationary'], all_supra_coeff['Running']),
                       sp.stats.wilcoxon(all_gran_coeff['Stationary'], all_gran_coeff['Running']),
                       sp.stats.wilcoxon(all_infra_coeff['Stationary'], all_infra_coeff['Running'])])

    ind = np.arange(len(layer_run_means))
    width = 0.25

    fig2, ax = plt.subplots()
    stat_bar = ax.bar(ind, layer_stat_means, width, color='#348ABD', yerr=layer_stat_std, ecolor='k')
    run_bar = ax.bar(ind + width, layer_run_means, width, color='#E24A33', yerr=layer_run_std, ecolor='k')

    y_lim = ax.get_ylim()
    offset = (y_lim[1] - y_lim[0]) / 5

    for i, p in enumerate(p_vals):
        if p[1] >= 0.05:
            displaystring = r'n.s.'
        elif p[1] < 0.001:
            displaystring = r'***'
        elif p[1] < 0.01:
            displaystring = r'**'
        else:
            displaystring = r'*'

        height = offset + layer_stat_means.max()
        bar_centers = ind[i] + np.array([0.5, 1.5]) * width
        significance_bar(bar_centers[0], bar_centers[1], height, displaystring)

    ax.legend((stat_bar[0], run_bar[0]), ('Stationary', 'Running'), loc=3)

    if vstim == 'y':
        stim_str = 'Visual Stimulus'
    else:
        stim_str = 'No Stimulus'

    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Supragranular', 'Granular', 'Infragranular'))

    plt.savefig(data_path + 'figures/movement_bar_layer.pdf', format='pdf')
    plt.close()

    f2, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    all_supra_coeff.plot.hist(alpha=0.5, ax=ax1)
    ax1.set_title('Supragranular')
    all_gran_coeff.plot.hist(alpha=0.5, ax=ax2)
    ax2.set_title('Granular')
    all_infra_coeff.plot.hist(alpha=0.5, ax=ax3)
    ax3.set_title('Infragranular')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(data_path + 'figures/movement_hist_layer.pdf', format='pdf')

    plt.close()
    plt.ion()

    return p_vals


def layer_type_plots(data_path, vstim, all_stat_coeff, all_run_coeff, supra_list,
                     gran_list, infra_list, pv_list, pyr_list):
    # layer plots by cell type
    plt.ioff()
    supra_pv_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(supra_list)) &
                                         (all_stat_coeff['Row'].isin(pv_list)) &
                                         (all_stat_coeff['Column'].isin(supra_list)) &
                                         (all_stat_coeff['Column'].isin(pv_list))]
    supra_pv_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(supra_list)) &
                                       (all_run_coeff['Row'].isin(pv_list)) &
                                       (all_run_coeff['Column'].isin(supra_list)) &
                                       (all_run_coeff['Column'].isin(pv_list))]
    supra_pyr_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(supra_list)) &
                                          (all_stat_coeff['Row'].isin(pyr_list)) &
                                          (all_stat_coeff['Column'].isin(supra_list)) &
                                          (all_stat_coeff['Column'].isin(pyr_list))]
    supra_pyr_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(supra_list)) &
                                        (all_run_coeff['Row'].isin(pyr_list)) &
                                        (all_run_coeff['Column'].isin(supra_list)) &
                                        (all_run_coeff['Column'].isin(pyr_list))]

    gran_pv_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(gran_list)) &
                                        (all_stat_coeff['Row'].isin(pv_list)) &
                                        (all_stat_coeff['Column'].isin(gran_list)) &
                                        (all_stat_coeff['Column'].isin(pv_list))]
    gran_pv_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(gran_list)) &
                                      (all_run_coeff['Row'].isin(pv_list)) &
                                      (all_run_coeff['Column'].isin(gran_list)) &
                                      (all_run_coeff['Column'].isin(pv_list))]
    gran_pyr_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(gran_list)) &
                                         (all_stat_coeff['Row'].isin(pyr_list)) &
                                         (all_stat_coeff['Column'].isin(gran_list)) &
                                         (all_stat_coeff['Column'].isin(pyr_list))]
    gran_pyr_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(gran_list)) &
                                       (all_run_coeff['Row'].isin(pyr_list)) &
                                       (all_run_coeff['Column'].isin(gran_list)) &
                                       (all_run_coeff['Column'].isin(pyr_list))]

    infra_pv_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(infra_list)) &
                                         (all_stat_coeff['Row'].isin(pv_list)) &
                                         (all_stat_coeff['Column'].isin(infra_list)) &
                                         (all_stat_coeff['Column'].isin(pv_list))]
    infra_pv_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(infra_list)) &
                                       (all_run_coeff['Row'].isin(pv_list)) &
                                       (all_run_coeff['Column'].isin(infra_list)) &
                                       (all_run_coeff['Column'].isin(pv_list))]
    infra_pyr_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(infra_list)) &
                                          (all_stat_coeff['Row'].isin(pyr_list)) &
                                          (all_stat_coeff['Column'].isin(infra_list)) &
                                          (all_stat_coeff['Column'].isin(pyr_list))]
    infra_pyr_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(infra_list)) &
                                        (all_run_coeff['Row'].isin(pyr_list)) &
                                        (all_run_coeff['Column'].isin(infra_list)) &
                                        (all_run_coeff['Column'].isin(pyr_list))]

    all_pv_supra_coeff = pd.concat([supra_pv_stat_coeff['corr_coefficient'],
                                    supra_pv_run_coeff['corr_coefficient']], axis=1)
    all_pyr_supra_coeff = pd.concat([supra_pyr_stat_coeff['corr_coefficient'],
                                     supra_pyr_run_coeff['corr_coefficient']], axis=1)
    all_pv_supra_coeff.columns = ['Stationary', 'Running']
    all_pyr_supra_coeff.columns = ['Stationary', 'Running']

    all_pv_gran_coeff = pd.concat([gran_pv_stat_coeff['corr_coefficient'],
                                   gran_pv_run_coeff['corr_coefficient']], axis=1)
    all_pyr_gran_coeff = pd.concat([gran_pyr_stat_coeff['corr_coefficient'],
                                    gran_pyr_run_coeff['corr_coefficient']], axis=1)
    all_pv_gran_coeff.columns = ['Stationary', 'Running']
    all_pyr_gran_coeff.columns = ['Stationary', 'Running']

    all_pv_infra_coeff = pd.concat([infra_pv_stat_coeff['corr_coefficient'],
                                    infra_pv_run_coeff['corr_coefficient']], axis=1)
    all_pyr_infra_coeff = pd.concat([infra_pyr_stat_coeff['corr_coefficient'],
                                     infra_pyr_run_coeff['corr_coefficient']], axis=1)
    all_pv_infra_coeff.columns = ['Stationary', 'Running']
    all_pyr_infra_coeff.columns = ['Stationary', 'Running']

    layer_type_run_means = np.array([all_pv_supra_coeff['Running'].mean(),
                                     all_pyr_supra_coeff['Running'].mean(),
                                     all_pv_gran_coeff['Running'].mean(),
                                     all_pyr_gran_coeff['Running'].mean(),
                                     all_pv_infra_coeff['Running'].mean(),
                                     all_pyr_infra_coeff['Running'].mean()])

    layer_type_run_std = np.array([all_pv_supra_coeff['Running'].std() / np.sqrt(len(supra_pv_run_coeff)),
                                   all_pyr_supra_coeff['Running'].std() / np.sqrt(len(supra_pyr_run_coeff)),
                                   all_pv_gran_coeff['Running'].std() / np.sqrt(len(gran_pv_run_coeff)),
                                   all_pyr_gran_coeff['Running'].std() / np.sqrt(len(gran_pyr_run_coeff)),
                                   all_pv_infra_coeff['Running'].std() / np.sqrt(len(infra_pv_run_coeff)),
                                   all_pyr_infra_coeff['Running'].std() / np.sqrt(len(infra_pyr_run_coeff))])

    layer_type_stat_means = np.array([all_pv_supra_coeff['Stationary'].mean(),
                                      all_pyr_supra_coeff['Stationary'].mean(),
                                      all_pv_gran_coeff['Stationary'].mean(),
                                      all_pyr_gran_coeff['Stationary'].mean(),
                                      all_pv_infra_coeff['Stationary'].mean(),
                                      all_pyr_infra_coeff['Stationary'].mean()])

    layer_type_stat_std = np.array([all_pv_supra_coeff['Stationary'].std() / np.sqrt(len(supra_pv_stat_coeff)),
                                    all_pyr_supra_coeff['Stationary'].std() / np.sqrt(len(supra_pyr_stat_coeff)),
                                    all_pv_gran_coeff['Stationary'].std() / np.sqrt(len(gran_pv_stat_coeff)),
                                    all_pyr_gran_coeff['Stationary'].std() / np.sqrt(len(gran_pyr_stat_coeff)),
                                    all_pv_infra_coeff['Stationary'].std() / np.sqrt(len(infra_pv_stat_coeff)),
                                    all_pyr_infra_coeff['Stationary'].std() / np.sqrt(len(infra_pyr_stat_coeff))])

    ind = np.arange(len(layer_type_stat_means))
    width = 0.25

    fig3, ax = plt.subplots()
    stat_bar = ax.bar(ind, layer_type_stat_means, width, color='#348ABD', yerr=layer_type_stat_std, ecolor='k')
    run_bar = ax.bar(ind + width, layer_type_run_means, width, color='#E24A33', yerr=layer_type_run_std, ecolor='k')

    if vstim == 'y':
        stim_str = 'Visual Stimulus'
    else:
        stim_str = 'No Stimulus'

    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Supra\n(PV)', 'Supra\n(Pyr)', 'Gran\n(PV)', 'Gran\n(Pyr)',
                        'Infra\n(PV)', 'Infra\n(Pyr)'))
    ax.legend((stat_bar[0], run_bar[0]), ('Stationary', 'Running'))

    plt.savefig(data_path + 'figures/movement_bar_layer_type.pdf', format='pdf')
    plt.close()

    f3, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True)
    all_pv_supra_coeff.plot.hist(alpha=0.5, ax=ax1)
    ax1.set_title('Supragranular (PV)')
    all_pyr_supra_coeff.plot.hist(alpha=0.5, ax=ax2)
    ax2.set_title('Supragranular (Pyr)')
    all_pv_gran_coeff.plot.hist(alpha=0.5, ax=ax3)
    ax3.set_title('Granular (PV)')
    all_pyr_gran_coeff.plot.hist(alpha=0.5, ax=ax4)
    ax4.set_title('Granular (Pyr)')
    all_pv_infra_coeff.plot.hist(alpha=0.5, ax=ax5)
    ax5.set_title('Infragranular (PV)')
    all_pyr_infra_coeff.plot.hist(alpha=0.5, ax=ax6)
    ax6.set_title('Infragranular (Pyr)')
    plt.subplots_adjust(hspace=0.7)
    plt.savefig(data_path + 'figures/movement_hist_layer_type.pdf', format='pdf')

    plt.close()
    plt.ion()


def fano_plot(data_path, fano):
    plt.ioff()
    unity = np.arange(0, 5, 0.01)
    plt.plot(unity, unity, '--')

    pv_map = {'n': '#8EBA42',
              'y': '#988ED5'}

    classes = ['non-PV', 'PV']
    class_colors = ['#8EBA42', '#988ED5']
    recs = []

    for i in range(0, len(class_colors)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colors[i]))

    plt.scatter(fano.run, fano.stat, marker='o', color=fano['PV'].map(pv_map))
    plt.xlabel('Running')
    plt.ylabel('Stationary')
    plt.axis([0, 5, 0, 5])
    plt.legend(recs, classes, loc=4)
    plt.savefig(data_path + 'figures/fano.pdf', format='pdf')
    plt.close()
    plt.ion()


def corr_coeff_df(binned_counts):
    counts_corr = binned_counts.corr().where(
        np.triu(np.ones(binned_counts.corr().shape)).astype(np.bool) == False)
    counts_corr = counts_corr.stack().reset_index()
    counts_corr.columns = ['Row', 'Column', 'corr_coefficient']
    return counts_corr


def significance_bar(start, end, height, displaystring, linewidth=1.2,
                     markersize=8, boxpad=0.3, fontsize=15, color='k'):
    # draw a line with downticks at the ends
    plt.plot([start, end], [height] * 2, '-', color=color, lw=linewidth, marker=TICKDOWN, markeredgewidth=linewidth,
             markersize=markersize)
    # draw the text with a bounding box covering up the line
    t = plt.text(0.5*(start+end), height + 0.005, displaystring, ha='center', va='center',
             bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad='+str(boxpad)), size=fontsize)
    t.set_bbox(dict(color='white', alpha=0.0, edgecolor='white'))


def calc_MI(x, y, bins):

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = sklearn.metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi


def bar_plot(data_run, data_stat, err_run, err_stat, p_vals):
    fig, ax = plt.subplots()
    N = len(data_run)
    ind = np.arange(N)
    width = 0.25

    rects1 = ax.bar(ind + width, data_stat, width, color='#348ABD', alpha=0.9, yerr=err_stat, ecolor='k')
    rects2 = ax.bar(ind, data_run, width, color='#E24A33', alpha=0.9, yerr=err_run, ecolor='k')

    y_lim = ax.get_ylim()
    offset = (y_lim[1] - y_lim[0]) / 2
    for i, p in enumerate(p_vals):
        if p[1] >= 0.05:
            display_string = r'n.s.'
        elif p[1] < 0.001:
            display_string = r'***'
        elif p[1] < 0.01:
            display_string = r'**'
        else:
            display_string = r'*'

        height = offset + np.max(data_run)
        bar_centers = ind[i] + np.array([0.5, 1.5]) * width
        significance_bar(bar_centers[0], bar_centers[1], height, display_string)

    return fig, ax
