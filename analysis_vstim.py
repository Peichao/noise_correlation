import glob
from datetime import datetime
import ntpath
import os
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import noise_functions
import multiprocessing
from joblib import Parallel, delayed

plt.switch_backend('agg')

# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('fivethirtyeight')

num_cores = multiprocessing.cpu_count()

# get list of mat files and unit info from CSV in data folder
bin_size = 0.1  # bin size for correlograms
plt_fano = 'n'  # plot fano factor graphs?
plt_units = 'n'  # plot individual correlelograms?
plt_summary = 'y'  # plot summary population analyses?
data_path = '/Volumes/anupam/Amanda Data/noise_correlation/resort/'  # path to all data
mat_files = list(glob.iglob(data_path + 'vstim/*.mat'))  # list of all mat files in data folder
unit_data = pd.read_csv(data_path + 'unit_info.csv')  # unit data from CSV in data folder
layers = {1: 'supragranular', 2: 'granular', 3: 'infragranular'}  # replace layer numbers with layer names
unit_data = unit_data.replace({"layer": layers})


def analysis_vstim(bin_size, plt_fano, plt_units, plt_summary, data_path, mat_files, unit_data):
    # create folder for figures
    if not os.path.exists(data_path + 'vstim/figures'):
        os.makedirs(data_path + 'vstim/figures')

    # initialize dataframes and lists
    all_stat_coeff_pref, all_run_coeff_pref, all_stat_coeff_nonpref, all_run_coeff_nonpref = (pd.DataFrame() for
                                                                                              i in range(4))
    unit_counter = 0
    vis_info = {}
    pv_list, pyr_list, supra_list, gran_list, infra_list = ([] for i2 in range(5))

    for file in mat_files:
        date_stat_counts, date_run_counts = (pd.DataFrame() for i3 in range(2))
        stat_bins_pref, mov_bins_pref, stat_bins_nonpref, mov_bins_nonpref = (pd.DataFrame() for i4 in range(4))

        # get date from filename (first 8 characters)
        date = int(ntpath.basename(file)[:8])
        # print message of date to the console
        print(str(datetime.now()) + " Currently processing: " + str(date))

        # load mat file for date
        vis_info = sio.loadmat(file, squeeze_me=True, struct_as_record=False)['vis_info']

        # filter units that belong to date
        date_units = unit_data[(unit_data.date == date) & (unit_data.v_stim == 'y')]

        # iterate through each unit (fix to remove pandas iteration?)
        for unit in date_units.iterrows():
            unit = unit[1]
            layer = unit.layer  # get unit layer as string
            channel = unit['channel']  # get unit channel number
            unit_number = unit['unit_number']  # get unit number
            unit_str = '%s_%s_%s' % (date, channel, unit_number)

            channel_ind = channel - 1
            unit_ind = unit_number - 1

            try:
                vresp_sig = vis_info[channel_ind, unit_ind].vresp_sig
                osi_sig = vis_info[channel_ind, unit_ind].osi_sig
            except IndexError:
                vresp_sig = vis_info[channel_ind].vresp_sig
                osi_sig = vis_info[channel_ind].osi_sig

            if vresp_sig < 0.05:

                if unit.layer == 'infragranular':
                    infra_list.append(unit_str)
                elif unit.layer == 'granular':
                    gran_list.append(unit_str)
                elif unit.layer == 'supragranular':
                    supra_list.append(unit_str)

                if unit.PV == 'y':
                    pv_list.append(unit_str)
                else:
                    pyr_list.append(unit_str)

                try:
                    trial_num = pd.DataFrame(vis_info[channel_ind, unit_ind].trials_num)
                    spk_times = pd.DataFrame(vis_info[channel_ind, unit_ind].spk_trial)
                    stat_trials = vis_info[channel_ind, unit_ind].trials_ori_Loff_stat
                    mov_trials = vis_info[channel_ind, unit_ind].trials_ori_Loff_run
                    ori = vis_info[channel_ind, unit_ind].deg
                    stim_time = vis_info[channel_ind, unit_ind].stim_time
                    fsample = vis_info[channel_ind, unit_ind].fsample
                    max_ori_ind = vis_info[channel_ind, unit_ind].max_min_deg_osi[0] - 1
                except IndexError:
                    trial_num = pd.DataFrame(vis_info[channel_ind].trials_num)
                    spk_times = pd.DataFrame(vis_info[channel_ind].spk_trial)
                    stat_trials = vis_info[channel_ind].trials_ori_Loff_stat
                    mov_trials = vis_info[channel_ind].trials_ori_Loff_run
                    ori = vis_info[channel_ind].deg
                    stim_time = vis_info[channel_ind].stim_time
                    fsample = vis_info[channel_ind].fsample
                    max_ori_ind = vis_info[channel_ind].max_min_deg_osi[0] - 1

                if (unit.PV == 'y') or ((unit.PV == 'n') & (osi_sig < 0.05)):
                    stat_trials_pref = stat_trials[max_ori_ind] - 1
                    mov_trials_pref = mov_trials[max_ori_ind] - 1

                    stat_trials_nonpref = np.ma.array(stat_trials, mask=False)
                    stat_trials_nonpref.mask[max_ori_ind] = True
                    stat_trials_nonpref = stat_trials_nonpref.compressed()
                    for i, element in enumerate(stat_trials_nonpref):
                        if type(element) == int:
                            stat_trials_nonpref[i] = np.reshape(np.array(element), 1, )

                    stat_trials_nonpref = np.concatenate(stat_trials_nonpref)
                    stat_trials_nonpref -= 1

                    mov_trials_nonpref = np.ma.array(mov_trials, mask=False)
                    mov_trials_nonpref.mask[max_ori_ind] = True
                    mov_trials_nonpref = mov_trials_nonpref.compressed()
                    for i, element in enumerate(mov_trials_nonpref):
                        if type(element) == int:
                            mov_trials_nonpref[i] = np.reshape(np.array(element), 1, )
                    mov_trials_nonpref = np.concatenate(mov_trials_nonpref)
                    mov_trials_nonpref -= 1

                    # compute fano factor analysis
                    if (plt_fano == 'y') & (type(mov_trials_pref) != int):
                        bin_samples_fano = np.round(bin_size * fsample)
                        spk_times_fano = spk_times.groupby(spk_times.index // bin_samples_fano).sum()
                        # spk_times_fano_corrected = spk_times_fano.ix[:(spk_times_fano.index[-1])-1]

                        fano_stat = spk_times_fano[stat_trials_pref].std(axis=1) ** 2 \
                            / spk_times_fano[stat_trials_pref].mean(axis=1)
                        fano_mov = spk_times_fano[mov_trials_pref].std(axis=1) ** 2 \
                            / spk_times_fano[mov_trials_pref].mean(axis=1)

                        plt.ioff()
                        fig, ax = plt.subplots()
                        plt.plot(fano_mov, label='Running', linewidth=2)
                        plt.hold(True)
                        plt.plot(fano_stat, label='Stationary', linewidth=2)
                        plt.legend()
                        plt.title('Fano Factor for Unit %s' % unit_str)
                        plt.savefig(data_path + 'vstim/figures/' + unit_str + '_fano.pdf', format='pdf')
                        plt.close()
                        plt.ion()

                    stim_start_samp = int(np.round((stim_time[0] + 0.1) * fsample))
                    stim_end_samp = int(stim_start_samp + np.round((stim_time[1] - 0.1) * fsample))
                    stim_spk_times = spk_times[stim_start_samp:stim_end_samp]

                    # get number of samples per bin based on frequency of samples in dataset
                    bin_samples = np.round(bin_size * fsample)
                    stim_spk_times_bin = stim_spk_times.groupby(stim_spk_times.index // bin_samples).sum()
                    # delete last row due to unequal bin size resulting from integer division
                    stim_spk_times_bin_corrected = stim_spk_times_bin.ix[:(stim_spk_times_bin.index[-1])-1]

                    spk_bins_stat_pref = stim_spk_times_bin_corrected[stat_trials_pref]
                    spk_bins_mov_pref = stim_spk_times_bin_corrected[mov_trials_pref]

                    spk_bins_stat_nonpref = stim_spk_times_bin_corrected[stat_trials_nonpref]
                    spk_bins_mov_nonpref = stim_spk_times_bin_corrected[mov_trials_nonpref]

                    stat_bins_pref[unit_str] = pd.DataFrame(spk_bins_stat_pref.as_matrix().flatten(), columns=[unit_str])
                    mov_bins_pref[unit_str] = pd.DataFrame(spk_bins_mov_pref.as_matrix().flatten(), columns=[unit_str])

                    stat_bins_nonpref[unit_str] = pd.DataFrame(spk_bins_stat_nonpref.as_matrix().flatten(),
                                                               columns=[unit_str])
                    mov_bins_nonpref[unit_str] = pd.DataFrame(spk_bins_mov_nonpref.as_matrix().flatten(),
                                                              columns=[unit_str])

        stat_counts_corr_pref = noise_functions.corr_coeff_df(stat_bins_pref)
        run_counts_corr_pref = noise_functions.corr_coeff_df(mov_bins_pref)
        all_stat_coeff_pref = pd.concat([all_stat_coeff_pref, stat_counts_corr_pref], axis=0, ignore_index=True)
        all_run_coeff_pref = pd.concat([all_run_coeff_pref, run_counts_corr_pref], axis=0, ignore_index=True)

        stat_counts_corr_nonpref = noise_functions.corr_coeff_df(stat_bins_nonpref)
        run_counts_corr_nonpref = noise_functions.corr_coeff_df(mov_bins_nonpref)
        all_stat_coeff_nonpref = pd.concat([all_stat_coeff_nonpref, stat_counts_corr_nonpref], axis=0, ignore_index=True)
        all_run_coeff_nonpref = pd.concat([all_run_coeff_nonpref, run_counts_corr_nonpref], axis=0, ignore_index=True)

        # plot individual correlogram for units if plt_units = y
        if plt_units == 'y':
            plt.ioff()
            plt.matshow(stat_bins_pref.corr())
            plt.savefig(data_path + 'vstim/figures/' + str(date) + '_stat.pdf', format='pdf')
            plt.close()

            plt.matshow(mov_bins_pref.corr())
            plt.savefig(data_path + 'vstim/figures/' + str(date) + '_run.pdf', format='pdf')
            plt.close()
            plt.ion()

    # plot population data in histograms and bar charts
    if plt_summary == 'y':
        all_coeff, all_pv_coeff, all_pyr_coeff, all_mix_coeff = noise_functions.main_plots(
            data_path + 'vstim/', 'y', all_stat_coeff_pref, all_run_coeff_pref, pv_list, pyr_list)

        noise_functions.layer_plots(data_path + 'vstim/', 'y', all_stat_coeff_pref, all_run_coeff_pref,
                                    supra_list, gran_list, infra_list)

        noise_functions.layer_type_plots(data_path + 'vstim/', 'y', all_stat_coeff_pref, all_run_coeff_pref,
                                         supra_list, gran_list, infra_list, pv_list, pyr_list)

    return all_coeff, all_pv_coeff, all_pyr_coeff, all_mix_coeff

bin_size_mult = [0.1]
coeffs_run, coeffs_stat, coeffs_run_pv, coeffs_stat_pv, coeffs_run_pyr, coeffs_stat_pyr = [pd.DataFrame()
                                                                                           for i in range(6)]

#
# def par_analysis(bin, plt_fano, plt_units, plt_summary, data_path, mat_files, unit_data):
#     all_coeff, all_pv_coeff, all_pyr_coeff, all_mix_coeff = analysis_vstim(bin, plt_fano, plt_units, plt_summary,
#                                                                            data_path, mat_files, unit_data)
#     return all_coeff
#
# results = []
# if __name__ == '__main__':
#     results = Parallel(n_jobs=num_cores)(delayed(par_analysis)(i2, plt_fano, plt_units, plt_summary,
#                                                              data_path, mat_files, unit_data) for i2 in bin_size_mult)
#
#     plt.switch_backend('Qt4Agg')
#
#     for idx, bin_idx in enumerate(results):
#         bin = bin_size_mult[idx]
#         coeffs_run[bin] = bin_idx['Running']
#         coeffs_stat[bin] = bin_idx['Stationary']
#
#     coeffs = np.stack((coeffs_stat.as_matrix(), coeffs_run.as_matrix()), axis=2)
#     colors = {'Stationary': '#348ABD',
#               'Running': '#E24A33'}
#     fig, ax = plt.subplots()
#     sns.tsplot(data=coeffs, time=bin_size_mult, ax=ax, condition=['Stationary', 'Running'], color=colors)
#     ax.set_title('Comparison of Correlation Coefficient and Counting Window, Visual Stimulus')
#     ax.set_xlabel('Counting Window (seconds)')
#     ax.set_ylabel('Pearson Correlation Coefficient')
#     plt.show()


for bin in bin_size_mult:
    all_coeff, all_pv_coeff, all_pyr_coeff, all_mix_coeff = analysis_vstim(bin, plt_fano, plt_units, plt_summary,
                                                                           data_path, mat_files, unit_data)
    coeffs_run[bin] = all_coeff['Running']
    coeffs_stat[bin] = all_coeff['Stationary']

    coeffs_run_pv[bin] = all_pv_coeff['Running']
    coeffs_stat_pv[bin] = all_pv_coeff['Stationary']

    coeffs_run_pyr[bin] = all_pyr_coeff['Running']
    coeffs_stat_pyr[bin] = all_pyr_coeff['Stationary']

coeffs = np.stack((coeffs_stat.as_matrix(), coeffs_run.as_matrix()), axis=2)

fig, ax = plt.subplots()

ax.plot(coeffs_run.mean(), color='#E24A33', label='All Units (Running)')
ax.fill_between(bin_size_mult, coeffs_run.mean() - coeffs_run.sem(), coeffs_run.mean() + coeffs_run.sem(),
                color='#E24A33', alpha=0.5)

ax.plot(coeffs_stat.mean(), color='#348ABD', label='All Units (Stationary)')
ax.fill_between(bin_size_mult, coeffs_stat.mean() - coeffs_stat.sem(), coeffs_stat.mean() + coeffs_stat.sem(),
                color='#348ABD', alpha=0.5)

ax.plot(coeffs_run_pv.mean(), color='#E24A33', linestyle='dashed', label='PV Units (Running)')
ax.fill_between(bin_size_mult, coeffs_run_pv.mean() - coeffs_run_pv.sem(), coeffs_run_pv.mean() + coeffs_run_pv.sem(),
                color='#E24A33', alpha=0.5)

ax.plot(coeffs_stat_pv.mean(), color='#348ABD', linestyle='dashed', label='PV Units (Stationary)')
ax.fill_between(bin_size_mult, coeffs_stat_pv.mean() - coeffs_stat_pv.sem(),
                coeffs_stat_pv.mean() + coeffs_stat_pv.sem(),
                color='#348ABD', alpha=0.5)

ax.set_title('Comparison of Correlation Coefficient and Counting Window, Visual Stimulus')
ax.set_xlabel('Counting Window (seconds)')
ax.set_ylabel('Pearson Correlation Coefficient')
ax.legend()
plt.savefig(data_path + 'vstim/figures/corr_time.pdf', format='pdf')
