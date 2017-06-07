import glob
from datetime import datetime
import ntpath
import os
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import noise_functions
import multiprocessing

plt.switch_backend('agg')

# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial']

num_cores = multiprocessing.cpu_count()

# get list of mat files and unit info from CSV in data folder
bin_sizes = np.linspace(0.01, 0.3, 30)  # bin size for correlograms
# bin_sizes = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
plt_fano = 'n'  # plot fano factor graphs?
plt_units = 'n'  # plot individual correlelograms?
plt_summary = 'y'  # plot summary population analyses?
data_path = '/Volumes/anupam/Amanda Data/noise_correlation/resort/'  # path to all data
mat_files = list(glob.iglob(data_path + 'vstim/*.mat'))  # list of all mat files in data folder
unit_data = pd.read_csv(data_path + 'unit_info.csv')  # unit data from CSV in data folder
layers = {1: 'supragranular', 2: 'granular', 3: 'infragranular'}  # replace layer numbers with layer names
unit_data = unit_data.replace({"layer": layers})


# create folder for figures
if not os.path.exists(data_path + 'vstim/figures'):
    os.makedirs(data_path + 'vstim/figures')

# initialize dataframes and lists
all_stat_coeff, all_run_coeff = ({} for i in range(2))
unit_counter = 0
vis_info = {}
pv_list, pyr_list, supra_list, gran_list, infra_list = ([] for i2 in range(5))
v_resp, ori_sel, pv_list_all, pyr_list_all = ([] for prob in range(4))
ori_pref = {}

coeffs_run, coeffs_stat, coeffs_run_pv, coeffs_stat_pv, coeffs_run_pyr, coeffs_stat_pyr = [pd.DataFrame()
                                                                                           for i3 in range(6)]

for bin_size in bin_sizes:
    print('Analyzing bin size: %.2f' % bin_size)
    for file in mat_files:
        date_stat_counts, date_run_counts = (pd.DataFrame() for i4 in range(2))
        stat_bins, mov_bins = (pd.DataFrame() for i5 in range(2))

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

            if unit.PV == 'y':
                pv_list_all.append(unit_str)
            else:
                pyr_list_all.append(unit_str)

            try:
                vresp_sig = vis_info[channel_ind, unit_ind].vresp_sig
                osi_sig = vis_info[channel_ind, unit_ind].osi_sig
            except IndexError:
                vresp_sig = vis_info[channel_ind].vresp_sig
                osi_sig = vis_info[channel_ind].osi_sig

            if vresp_sig < 0.05:
                v_resp.append(unit_str)
                if osi_sig < 0.05:
                    ori_sel.append(unit_str)

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

                ori_pref[unit_str] = ori

                if (unit.PV == 'y') or ((unit.PV == 'n') & (osi_sig < 0.05)):
                    for i, item in enumerate(stat_trials):
                        if not type(item) == np.ndarray:
                            stat_trials[i] = np.array([item])
                    stat_trials = np.sort(np.concatenate(stat_trials.astype(list))) - 1

                    for i, item in enumerate(mov_trials):
                        if not type(item) == np.ndarray:
                            mov_trials[i] = np.array([item])
                    mov_trials = np.sort(np.concatenate(mov_trials.astype(list))) - 1

                    # compute fano factor analysis
                    if (plt_fano == 'y') & (type(mov_trials) != int):
                        bin_samples_fano = np.round(bin_size * fsample)
                        spk_times_fano = spk_times.groupby(spk_times.index // bin_samples_fano).sum()
                        # spk_times_fano_corrected = spk_times_fano.ix[:(spk_times_fano.index[-1])-1]

                        fano_stat = spk_times_fano[stat_trials].std(axis=1) ** 2 \
                            / spk_times_fano[stat_trials].mean(axis=1)
                        fano_mov = spk_times_fano[mov_trials].std(axis=1) ** 2 \
                            / spk_times_fano[mov_trials].mean(axis=1)

                        plt.ioff()
                        fig_1, ax_1 = plt.subplots()
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

                    spk_bins_stat = stim_spk_times_bin_corrected[stat_trials]
                    spk_bins_mov = stim_spk_times_bin_corrected[mov_trials]

                    stat_bins[unit_str] = pd.DataFrame(spk_bins_stat.as_matrix().flatten(), columns=[unit_str])
                    mov_bins[unit_str] = pd.DataFrame(spk_bins_mov.as_matrix().flatten(), columns=[unit_str])

        stat_counts_corr = noise_functions.corr_coeff_df(stat_bins)
        run_counts_corr = noise_functions.corr_coeff_df(mov_bins)

        all_stat_coeff[bin_size] = pd.DataFrame()
        all_run_coeff[bin_size] = pd.DataFrame()
        all_stat_coeff[bin_size] = pd.concat([all_stat_coeff[bin_size], stat_counts_corr], axis=0, ignore_index=True)
        all_run_coeff[bin_size] = pd.concat([all_run_coeff[bin_size], run_counts_corr], axis=0, ignore_index=True)

        # plot individual correlogram for units if plt_units = y
        if plt_units == 'y':
            plt.ioff()
            plt.matshow(stat_bins.corr())
            plt.savefig(data_path + 'vstim/figures/' + str(date) + '_stat.pdf', format='pdf')
            plt.close()

            plt.matshow(mov_bins.corr())
            plt.savefig(data_path + 'vstim/figures/' + str(date) + '_run.pdf', format='pdf')
            plt.close()
            plt.ion()

# plot population data in histograms and bar charts
if plt_summary == 'y':
    summary_bin = bin_sizes[9]
    all_coeff, all_pv_coeff, all_pyr_coeff, all_mix_coeff = noise_functions.main_plots(
        data_path + 'vstim/', 'y', all_stat_coeff[summary_bin], all_run_coeff[summary_bin], pv_list, pyr_list)

    p_vals = noise_functions.layer_plots(data_path + 'vstim/', 'y', all_stat_coeff[summary_bin],
                                         all_run_coeff[summary_bin], supra_list, gran_list, infra_list)

    noise_functions.layer_type_plots(data_path + 'vstim/', 'y', all_stat_coeff[summary_bin], all_run_coeff[summary_bin],
                                     supra_list, gran_list, infra_list, pv_list, pyr_list)

stat_coeff = all_stat_coeff[bin_sizes[0]]
run_coeff = all_run_coeff[bin_sizes[0]]

infra_stat_coeff = all_stat_coeff[bin_sizes[0]][(all_stat_coeff[bin_sizes[0]]['Row'].isin(infra_list)) &
                                                (all_stat_coeff[bin_sizes[0]]['Column'].isin(infra_list))]
gran_stat_coeff = all_stat_coeff[bin_sizes[0]][(all_stat_coeff[bin_sizes[0]]['Row'].isin(gran_list)) &
                                               (all_stat_coeff[bin_sizes[0]]['Column'].isin(gran_list))]
supra_stat_coeff = all_stat_coeff[bin_sizes[0]][(all_stat_coeff[bin_sizes[0]]['Row'].isin(supra_list)) &
                                                (all_stat_coeff[bin_sizes[0]]['Column'].isin(supra_list))]

infra_run_coeff = all_run_coeff[bin_sizes[0]][(all_run_coeff[bin_sizes[0]]['Row'].isin(infra_list)) &
                                              (all_run_coeff[bin_sizes[0]]['Column'].isin(infra_list))]
gran_run_coeff = all_run_coeff[bin_sizes[0]][(all_run_coeff[bin_sizes[0]]['Row'].isin(gran_list)) &
                                             (all_run_coeff[bin_sizes[0]]['Column'].isin(gran_list))]
supra_run_coeff = all_run_coeff[bin_sizes[0]][(all_run_coeff[bin_sizes[0]]['Row'].isin(supra_list)) &
                                              (all_run_coeff[bin_sizes[0]]['Column'].isin(supra_list))]

sgm_stat_coeff = all_stat_coeff[bin_sizes[0]][(all_stat_coeff[bin_sizes[0]]['Row'].isin(supra_list)) &
                                              (all_stat_coeff[bin_sizes[0]]['Column'].isin(gran_list)) |
                                              (all_stat_coeff[bin_sizes[0]]['Row'].isin(gran_list)) &
                                              (all_stat_coeff[bin_sizes[0]]['Column'].isin(supra_list))]
sim_stat_coeff = all_stat_coeff[bin_sizes[0]][(all_stat_coeff[bin_sizes[0]]['Row'].isin(supra_list)) &
                                              (all_stat_coeff[bin_sizes[0]]['Column'].isin(infra_list)) |
                                              (all_stat_coeff[bin_sizes[0]]['Row'].isin(infra_list)) &
                                              (all_stat_coeff[bin_sizes[0]]['Column'].isin(supra_list))]
gim_stat_coeff = all_stat_coeff[bin_sizes[0]][(all_stat_coeff[bin_sizes[0]]['Row'].isin(gran_list)) &
                                              (all_stat_coeff[bin_sizes[0]]['Column'].isin(infra_list)) |
                                              (all_stat_coeff[bin_sizes[0]]['Row'].isin(infra_list)) &
                                              (all_stat_coeff[bin_sizes[0]]['Column'].isin(gran_list))]

pv_stat_coeff = all_stat_coeff[bin_sizes[0]][(all_stat_coeff[bin_sizes[0]]['Row'].isin(pv_list)) &
                                             (all_stat_coeff[bin_sizes[0]]['Column'].isin(pv_list))]
pv_run_coeff = all_run_coeff[bin_sizes[0]][(all_run_coeff[bin_sizes[0]]['Row'].isin(pv_list)) &
                                           (all_run_coeff[bin_sizes[0]]['Column'].isin(pv_list))]

for i, bin_size in enumerate(bin_sizes[1:]):
    stat_coeff = pd.concat([stat_coeff, all_stat_coeff[bin_size]['corr_coefficient']], axis=1)
    run_coeff = pd.concat([run_coeff, all_run_coeff[bin_size]['corr_coefficient']], axis=1)

    pv_stat_coeff = pd.concat([pv_stat_coeff,
                               all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(pv_list)) &
                                                        (all_stat_coeff[bin_size]['Column'].isin(pv_list))]],
                              axis=1)
    pv_run_coeff = pd.concat([pv_run_coeff,
                               all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(pv_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(pv_list))]],
                             axis=1)

    infra_stat_coeff = pd.concat([infra_stat_coeff,
                                 all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(infra_list)) &
                                                          (all_stat_coeff[bin_size]['Column'].isin(infra_list))]
                                 ['corr_coefficient']], axis=1)
    gran_stat_coeff = pd.concat([gran_stat_coeff,
                                all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(gran_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(gran_list))]
                                ['corr_coefficient']], axis=1)
    supra_stat_coeff = pd.concat([supra_stat_coeff,
                                 all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(supra_list)) &
                                                          (all_stat_coeff[bin_size]['Column'].isin(supra_list))]
                                 ['corr_coefficient']], axis=1)

    infra_run_coeff = pd.concat([infra_run_coeff,
                                 all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(infra_list)) &
                                                         (all_run_coeff[bin_size]['Column'].isin(infra_list))]
                                 ['corr_coefficient']], axis=1)
    gran_run_coeff = pd.concat([gran_run_coeff,
                                all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(gran_list)) &
                                                        (all_run_coeff[bin_size]['Column'].isin(gran_list))]
                                ['corr_coefficient']], axis=1)
    supra_run_coeff = pd.concat([supra_run_coeff,
                                 all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(supra_list)) &
                                                         (all_run_coeff[bin_size]['Column'].isin(supra_list))]
                                 ['corr_coefficient']], axis=1)

    sgm_stat_coeff = pd.concat([sgm_stat_coeff,
                                all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(supra_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(gran_list)) |
                                                         (all_stat_coeff[bin_size]['Row'].isin(gran_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(supra_list))]
                               ['corr_coefficient']], axis=1)
    sim_stat_coeff = pd.concat([sim_stat_coeff,
                                all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(supra_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(infra_list)) |
                                                         (all_stat_coeff[bin_size]['Row'].isin(infra_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(supra_list))]
                               ['corr_coefficient']], axis=1)
    gim_stat_coeff = pd.concat([gim_stat_coeff,
                                all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(gran_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(infra_list)) |
                                                         (all_stat_coeff[bin_size]['Row'].isin(infra_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(gran_list))]
                               ['corr_coefficient']], axis=1)

fig, ax = plt.subplots()
noise_functions.simpleaxis(ax)
plot_stat = stat_coeff['corr_coefficient']
plot_run = run_coeff['corr_coefficient']

plot_stat_pv = pv_stat_coeff['corr_coefficient']
plot_run_pv = pv_run_coeff['corr_coefficient']

ax.plot(bin_sizes, plot_run.mean(), color='#ED1F24', label='Running')
ax.fill_between(bin_sizes, plot_run.mean() - plot_run.sem(), plot_run.mean() + plot_run.sem(),
                color='#ED1F24', alpha=0.5)

ax.plot(bin_sizes, plot_stat.mean(), color='#3852A4', label='Stationary')
ax.fill_between(bin_sizes, plot_stat.mean() - plot_stat.sem(), plot_stat.mean() + plot_stat.sem(),
                color='#3852A4', alpha=0.5)

# ax.plot(bin_sizes, plot_run_pv.mean(), color='#ED1F24', label='Running (PV)')
# ax.fill_between(bin_sizes, plot_run_pv.mean() - plot_run_pv.sem(), plot_run_pv.mean() + plot_run_pv.sem(),
#                 color='#ED1F24', alpha=0.5, linestyle='dashed')
#
# ax.plot(bin_sizes, plot_stat_pv.mean(), color='#3852A4', label='Stationary (PV)')
# ax.fill_between(bin_sizes, plot_stat_pv.mean() - plot_stat_pv.sem(), plot_stat_pv.mean() + plot_stat_pv.sem(),
#                 color='#3852A4', alpha=0.5, linestyle='dashed')

ax.set_xlabel('Counting Window (seconds)')
ax.set_ylabel('Pearson Correlation Coefficient')
ax.legend()
ax.set_ylim([0, 0.25])
plt.savefig(data_path + 'vstim/figures/corr_time.pdf', format='pdf')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
noise_functions.simpleaxis(ax1)
noise_functions.simpleaxis(ax2)
plot_stat_infra = infra_stat_coeff['corr_coefficient']
plot_stat_gran = gran_stat_coeff['corr_coefficient']
plot_stat_supra = supra_stat_coeff['corr_coefficient']

plot_run_infra = infra_run_coeff['corr_coefficient']
plot_run_gran = gran_run_coeff['corr_coefficient']
plot_run_supra = supra_run_coeff['corr_coefficient']

ax1.plot(bin_sizes, plot_stat_supra.mean(), color='#F05A22', label='Supragranular')
ax1.fill_between(bin_sizes, plot_stat_supra.mean() - plot_stat_supra.sem(),
                 plot_stat_supra.mean() + plot_stat_supra.sem(),
                 color='#F05A22', alpha=0.5)

ax1.plot(bin_sizes, plot_stat_gran.mean(), color='#20B473', label='Granular')
ax1.fill_between(bin_sizes, plot_stat_gran.mean() - plot_stat_gran.sem(),
                 plot_stat_gran.mean() + plot_stat_gran.sem(),
                 color='#20B473', alpha=0.5)

ax1.plot(bin_sizes, plot_stat_infra.mean(), color='#662F8F', label='Infragranular')
ax1.fill_between(bin_sizes, plot_stat_infra.mean() - plot_stat_infra.sem(),
                 plot_stat_infra.mean() + plot_stat_infra.sem(),
                 color='#662F8F', alpha=0.5)


ax2.plot(bin_sizes, plot_run_supra.mean(), color='#F05A22', label='Supragranular')
ax2.fill_between(bin_sizes, plot_run_supra.mean() - plot_run_supra.sem(),
                 plot_run_supra.mean() + plot_run_supra.sem(),
                 color='#F05A22', alpha=0.5)

ax2.plot(bin_sizes, plot_run_gran.mean(), color='#20B473', label='Granular')
ax2.fill_between(bin_sizes, plot_run_gran.mean() - plot_run_gran.sem(),
                 plot_run_gran.mean() + plot_run_gran.sem(),
                 color='#20B473', alpha=0.5)

ax2.plot(bin_sizes, plot_run_infra.mean(), color='#662F8F', label='Infragranular')
ax2.fill_between(bin_sizes, plot_run_infra.mean() - plot_run_infra.sem(),
                 plot_run_infra.mean() + plot_run_infra.sem(),
                 color='#662F8F', alpha=0.5)

ax1.set_xlabel('Counting Window (seconds)')
ax1.set_ylabel('Correlation Coefficient')
ax2.legend(loc=3)
plt.savefig(data_path + 'vstim/figures/corr_time_layer.pdf', format='pdf')
plt.close()

fig, ax1 = plt.subplots()
noise_functions.simpleaxis(ax1)
plot_stat_sgm = sgm_stat_coeff['corr_coefficient']
plot_stat_sim = sim_stat_coeff['corr_coefficient']
plot_stat_gsm = gim_stat_coeff['corr_coefficient']

ax1.plot(bin_sizes, plot_stat_sgm.mean(), color='#F05A22', label='Supragranular/Granular')
ax1.fill_between(bin_sizes, plot_stat_sgm.mean() - plot_stat_sgm.sem(),
                 plot_stat_sgm.mean() + plot_stat_sgm.sem(),
                 color='#F05A22', alpha=0.5)

ax1.plot(bin_sizes, plot_stat_sim.mean(), color='#20B473', label='Supragranular/Infragranular')
ax1.fill_between(bin_sizes, plot_stat_sim.mean() - plot_stat_sim.sem(),
                 plot_stat_sim.mean() + plot_stat_sim.sem(),
                 color='#20B473', alpha=0.5)

ax1.plot(bin_sizes, plot_stat_gsm.mean(), color='#662F8F', label='Granular/Infragranular')
ax1.fill_between(bin_sizes, plot_stat_gsm.mean() - plot_stat_gsm.sem(),
                 plot_stat_gsm.mean() + plot_stat_gsm.sem(),
                 color='#662F8F', alpha=0.5)

ax1.set_xlabel('Counting Window (seconds)')
ax1.set_ylabel('Correlation Coefficient')
ax1.legend(loc=3)
plt.savefig(data_path + 'vstim/figures/corr_time_mix.pdf', format='pdf')
plt.close()

v_resp_prop = np.array([np.size(list(set(v_resp).intersection(pv_list_all))) / np.size(pv_list_all),
                        np.size(list(set(v_resp).intersection(pyr_list_all))) / np.size(pyr_list_all)]) * 100
orisig_prop = np.array([np.size(list(set(ori_sel).intersection(pv_list_all))) / np.size(pv_list_all),
                        np.size(list(set(ori_sel).intersection(pyr_list_all))) / np.size(pyr_list_all)]) * 100
fig2, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
pos = np.arange(2)
vr_bar = ax1.bar(pos, v_resp_prop, align='center', alpha=0.9, color=['#988ED5', '#8EBA42'])
ax1.set_xticks(pos)
ax1.set_xticklabels(('PV', 'non-PV'))
ax1.set_xlabel('Visually Responsive')

os_bar = ax2.bar(pos, orisig_prop, align='center', alpha=0.9, color=['#988ED5', '#8EBA42'])
ax2.set_xticks(pos)
ax2.set_xticklabels(('PV', 'non-PV'))
ax2.set_xlabel('Orientation Selective')

ax1.set_ylabel('Selective Cells (Percentage)')

for bar in vr_bar:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., 1.02 * height,
             '%.2f%%' % height,
             ha='center', va='bottom')

for bar in os_bar:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., 1.02 * height,
             '%.2f%%' % height,
             ha='center', va='bottom')

plt.savefig(data_path + 'vstim/figures/props.pdf', format='pdf')
plt.close()
