import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def main_plots(data_path, all_stat_coeff, all_run_coeff, pv_list, pyr_list):
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
    plt.title('All PV-PV Pairs')
    plt.savefig(data_path + 'figures/' + 'movement_boxplot_pv.pdf', format='pdf')
    plt.close()

    pyr_stat_coeff = all_stat_coeff[(all_stat_coeff['Row'].isin(pyr_list)) & (all_stat_coeff['Column'].isin(pyr_list))]
    pyr_run_coeff = all_run_coeff[(all_run_coeff['Row'].isin(pyr_list)) & (all_run_coeff['Column'].isin(pyr_list))]
    plt.boxplot([pyr_stat_coeff['corr_coefficient'], pyr_run_coeff['corr_coefficient']])
    plt.xticks([1, 2], ['Stationary', 'Running'])
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('All Pyr-Pyr Pairs')
    plt.savefig(data_path + 'figures/' + 'movement_boxplot_pyr.pdf', format='pdf')
    plt.close()

    all_coeff = pd.concat([all_stat_coeff['corr_coefficient'],
                           all_run_coeff['corr_coefficient']], axis=1)
    all_coeff.columns = ['Stationary', 'Running']
    all_pv_coeff = pd.concat([pv_stat_coeff['corr_coefficient'],
                              pv_run_coeff['corr_coefficient']], axis=1)
    all_pv_coeff.columns = ['Stationary', 'Running']
    all_pyr_coeff = pd.concat([pyr_stat_coeff['corr_coefficient'],
                               pyr_run_coeff['corr_coefficient']], axis=1)
    all_pyr_coeff.columns = ['Stationary', 'Running']

    run_means = np.array([all_coeff['Running'].mean(),
                          all_pv_coeff['Running'].mean(),
                          all_pyr_coeff['Running'].mean()])

    run_std = np.array([all_coeff['Running'].std() / np.sqrt(len(all_run_coeff)),
                        all_pv_coeff['Running'].std() / np.sqrt(len(pv_run_coeff)),
                        all_pyr_coeff['Running'].std() / np.sqrt(len(pyr_run_coeff))])

    stat_means = np.array([all_coeff['Stationary'].mean(),
                           all_pv_coeff['Stationary'].mean(),
                           all_pyr_coeff['Stationary'].mean()])

    stat_std = np.array([all_coeff['Stationary'].std() / np.sqrt(len(all_stat_coeff)),
                         all_pv_coeff['Stationary'].std() / np.sqrt(len(pv_stat_coeff)),
                         all_pyr_coeff['Stationary'].std() / np.sqrt(len(pyr_stat_coeff))])

    ind = np.arange(len(run_means))
    width = 0.25

    fig, ax = plt.subplots()
    stat_bar = ax.bar(ind, stat_means, width, color='#E24A33', yerr=stat_std, ecolor='k')
    run_bar = ax.bar(ind + width, run_means, width, color='#348ABD', yerr=run_std, ecolor='k')

    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_title('Correlation Coefficients, No Stimulus')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('All Units', 'PV-PV', 'Pyr-Pyr'))
    ax.legend((stat_bar[0], run_bar[0]), ('Stationary', 'Running'))

    plt.savefig(data_path + 'figures/movement_bar.pdf', format='pdf')
    plt.close()

    f1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    all_coeff.plot.hist(alpha=0.5, ax=ax1)
    ax1.set_title('All Units')
    all_pv_coeff.plot.hist(alpha=0.5, ax=ax2)
    ax2.set_title('PV-PV')
    all_pyr_coeff.plot.hist(alpha=0.5, ax=ax3)
    ax3.set_title('Pyr-Pyr')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(data_path + 'figures/movement_hist.pdf', format='pdf')

    plt.close()
    plt.ion()


def layer_plots(data_path, all_stat_coeff, all_run_coeff, supra_list, gran_list, infra_list):
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

    ind = np.arange(len(layer_run_means))
    width = 0.25

    fig2, ax = plt.subplots()
    stat_bar = ax.bar(ind, layer_stat_means, width, color='#E24A33', yerr=layer_stat_std, ecolor='k')
    run_bar = ax.bar(ind + width, layer_run_means, width, color='#348ABD', yerr=layer_run_std, ecolor='k')

    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_title('Correlation Coefficients, No Stimulus')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Supragranular', 'Granular', 'Infragranular'))
    ax.legend((stat_bar[0], run_bar[0]), ('Stationary', 'Running'))

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


def layer_type_plots(data_path, all_stat_coeff, all_run_coeff, supra_list, gran_list, infra_list, pv_list, pyr_list):
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
    stat_bar = ax.bar(ind, layer_type_stat_means, width, color='#E24A33', yerr=layer_type_stat_std, ecolor='k')
    run_bar = ax.bar(ind + width, layer_type_run_means, width, color='#348ABD', yerr=layer_type_run_std, ecolor='k')

    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_title('Correlation Coefficients, No Stimulus')
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
