import glob
from datetime import datetime
import ntpath
import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import noise_functions

# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# get list of mat files and unit info from CSV in data folder
bin_sizes = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
plot_bin_size = 0.1
plt_units = 'n'  # plot individual correlelograms?
plt_summary = 'y'  # plot summary population analyses?
data_path = '/Volumes/anupam/Amanda Data/noise_correlation/resort/'  # path to all data
mat_files = list(glob.iglob(data_path + 'nostim/*.mat'))  # list of all mat files in data folder
unit_data = pd.read_csv(data_path + 'unit_info.csv')  # unit data from CSV in data folder
layers = {1: 'supragranular', 2: 'granular', 3: 'infragranular'}  # replace layer numbers with layer names
unit_data = unit_data.replace({"layer": layers})

# create folder for figures
if not os.path.exists(data_path + 'nostim/figures'):
    os.makedirs(data_path + 'nostim/figures')

# initialize dataframes and lists
all_stat_coeff, all_run_coeff = ({} for i in range(2))
vis_info = {}
pv_list, pyr_list, supra_list, gran_list, infra_list = ([] for i in range(5))
fano = pd.DataFrame(columns=['unit', 'PV', 'layer', 'stat', 'run'])

for bin_size in bin_sizes:
    unit_counter = 0
    spk_rates = pd.DataFrame(columns=['unit', 'run', 'stat'])
    print('Analyzing bin size: %.2f' % bin_size)
    all_stat_coeff[bin_size] = pd.DataFrame()
    all_run_coeff[bin_size] = pd.DataFrame()
    for file in mat_files:
        date_stat_counts, date_run_counts = (pd.DataFrame() for i in range(2))

        # get date from filename (first 8 characters)
        date = int(ntpath.basename(file)[:8])
        # print message of date to the console
        print(str(datetime.now()) + " Currently processing: " + str(date))

        # load mat file for date
        vis_info = sio.loadmat(file, squeeze_me=True, struct_as_record=False)['info']

        # filter units that belong to date
        date_units = unit_data[(unit_data.date == date) & (unit_data.v_stim == 'n')]
        vis_index = noise_functions.get_chans(vis_info)  # get channel numbers by iterating through mat file

        # iterate through each unit (fix to remove pandas iteration?)
        for unit in date_units.iterrows():
            unit = unit[1]
            layer = unit.layer  # get unit layer as string
            channel = unit['channel']  # get unit channel number
            unit_number = unit['unit_number']  # get unit number

            channel_ind = int(vis_index[(vis_index.channel == channel) & (vis_index.cluster == unit_number)]['ind1'].values)
            unit_ind = int(vis_index[(vis_index.channel == channel) & (vis_index.cluster == unit_number)]['ind2'].values)

            # pull timing information, not all mat files have >1 column causing TypeError
            try:
                spk_times = pd.DataFrame(vis_info[channel_ind][unit_ind].Spk_times_s, columns=['spike times'])
                stat_times = vis_info[channel_ind][unit_ind].Stat_times
                mov_times = vis_info[channel_ind][unit_ind].Mov_times
                exp_dur = vis_info[channel_ind][unit_ind].Exp_dur
                stat_rate = vis_info[channel_ind][unit_ind].stat_spk_rate_tot
                mov_rate = vis_info[channel_ind][unit_ind].mov_spk_rate_tot
            except TypeError:
                spk_times = pd.DataFrame(vis_info[channel_ind].Spk_times_s, columns=['spike times'])
                stat_times = vis_info[channel_ind].Stat_times
                mov_times = vis_info[channel_ind].Mov_times
                exp_dur = vis_info[channel_ind].Exp_dur
                stat_rate = vis_info[channel_ind].stat_spk_rate_tot
                mov_rate = vis_info[channel_ind].mov_spk_rate_tot

            # get bins for pd.cut function by flattening stat and making sure that stat is first or adjusting
            bins = stat_times.flatten('F')
            first = ['stat', 'run']
            if np.max(bins) != exp_dur:
                bins = np.append(bins, exp_dur)
            if bins[0] != 0:
                bins = np.insert(bins, 0, np.zeros(1), axis=0)
                first = ['run', 'stat']

            spk_times['movement'] = np.where(pd.cut(spk_times['spike times'], bins, labels=False) % 2 == 1,
                                             first[1], first[0])

            run_counts, stat_counts = (pd.DataFrame() for i in range(2))
            for i, bin in enumerate(bins[:-1]):
                new_bins = np.arange(bins[i], bins[i+1], bin_size)
                spk_cut = pd.cut(spk_times['spike times'], bins=new_bins)
                spk_counts = pd.value_counts(spk_cut).reindex(spk_cut.cat.categories)

                # in case beginning of trial is not stat
                if first[0] == 'stat':
                    if i % 2 == 0:
                        stat_counts = pd.concat([stat_counts, spk_counts], axis=0)
                    else:
                        run_counts = pd.concat([run_counts, spk_counts], axis=0)
                else:
                    if i % 2 == 0:
                        run_counts = pd.concat([run_counts, spk_counts], axis=0)
                    else:
                        stat_counts = pd.concat([stat_counts, spk_counts], axis=0)

            unit_str = '%s_%s_%s' % (date, channel, unit_number)

            spk_rates.loc[unit_counter] = [unit_str, mov_rate, stat_rate]

            stat_fano = stat_counts[0].var() / stat_counts[0].mean()
            run_fano = run_counts[0].var() / run_counts[0].mean()
            fano_unit = [unit_str, unit.PV, unit.layer, stat_fano, run_fano]
            fano.loc[unit_counter] = fano_unit
            # create running list of units by layer and PV
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

            stat_counts.columns = [unit_str]
            run_counts.columns = [unit_str]

            date_stat_counts = pd.concat([date_stat_counts, stat_counts], axis=1)
            date_run_counts = pd.concat([date_run_counts, run_counts], axis=1)
            unit_counter += 1

        stat_counts_corr = date_stat_counts.corr().where(
            np.triu(np.ones(date_stat_counts.corr().shape)).astype(np.bool) == False)
        stat_counts_corr = stat_counts_corr.stack().reset_index()
        stat_counts_corr.columns = ['Row', 'Column', 'corr_coefficient']

        run_counts_corr = date_run_counts.corr().where(
            np.triu(np.ones(date_run_counts.corr().shape)).astype(np.bool) == False)
        run_counts_corr = run_counts_corr.stack().reset_index()
        run_counts_corr.columns = ['Row', 'Column', 'corr_coefficient']

        all_stat_coeff[bin_size] = pd.concat([all_stat_coeff[bin_size], stat_counts_corr], axis=0)
        all_run_coeff[bin_size] = pd.concat([all_run_coeff[bin_size], run_counts_corr], axis=0)

        # plot individual correlogram for units if plt_units = y
        if plt_units == 'y':
            plt.ioff()
            plt.matshow(date_stat_counts.corr())
            plt.savefig(data_path + 'figures/' + str(date) + '_stat.pdf', format='pdf')
            plt.close()

            plt.matshow(date_run_counts.corr())
            plt.savefig(data_path + 'figures/' + str(date) + '_run.pdf', format='pdf')
            plt.close()
            plt.ion()

# plot population data in histograms and bar charts
if plt_summary == 'y':
    noise_functions.main_plots(data_path + 'nostim/', 'n',
                               all_stat_coeff[plot_bin_size], all_run_coeff[plot_bin_size], pv_list, pyr_list)

    p_vals = noise_functions.layer_plots(data_path + 'nostim/', 'n',
                                all_stat_coeff[plot_bin_size], all_run_coeff[plot_bin_size],
                                supra_list, gran_list, infra_list)

    noise_functions.layer_type_plots(data_path + 'nostim/', 'n',
                                     all_stat_coeff[plot_bin_size], all_run_coeff[plot_bin_size],
                                     supra_list, gran_list, infra_list, pv_list, pyr_list)

    noise_functions.fano_plot(data_path + 'nostim/', fano)

# Plot correlation coefficients for multiple bin sizes

means_stat = np.zeros(np.size(bin_sizes))
means_run = np.zeros(np.size(bin_sizes))
means_stat_pv = np.zeros(np.size(bin_sizes))
means_run_pv = np.zeros(np.size(bin_sizes))

std_stat = np.zeros(np.size(bin_sizes))
std_run = np.zeros(np.size(bin_sizes))
std_stat_pv = np.zeros(np.size(bin_sizes))
std_run_pv = np.zeros(np.size(bin_sizes))

pv_stat_coeff = {}
pv_run_coeff = {}
pyr_stat_coeff = {}
pyr_run_coeff = {}

infra_stat_coeff = {}
gran_stat_coeff = {}
supra_stat_coeff = {}
means_stat_infra = np.zeros(np.size(bin_sizes))
std_stat_infra = np.zeros(np.size(bin_sizes))
means_stat_gran = np.zeros(np.size(bin_sizes))
std_stat_gran = np.zeros(np.size(bin_sizes))
means_stat_supra = np.zeros(np.size(bin_sizes))
std_stat_supra = np.zeros(np.size(bin_sizes))

infra_run_coeff = {}
gran_run_coeff = {}
supra_run_coeff = {}
means_run_infra = np.zeros(np.size(bin_sizes))
std_run_infra = np.zeros(np.size(bin_sizes))
means_run_gran = np.zeros(np.size(bin_sizes))
std_run_gran = np.zeros(np.size(bin_sizes))
means_run_supra = np.zeros(np.size(bin_sizes))
std_run_supra = np.zeros(np.size(bin_sizes))

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

sgm_run_coeff = all_run_coeff[bin_sizes[0]][(all_run_coeff[bin_sizes[0]]['Row'].isin(supra_list)) &
                                            (all_run_coeff[bin_sizes[0]]['Column'].isin(gran_list)) |
                                            (all_run_coeff[bin_sizes[0]]['Row'].isin(gran_list)) &
                                            (all_run_coeff[bin_sizes[0]]['Column'].isin(supra_list))]
sim_run_coeff = all_run_coeff[bin_sizes[0]][(all_run_coeff[bin_sizes[0]]['Row'].isin(supra_list)) &
                                            (all_run_coeff[bin_sizes[0]]['Column'].isin(infra_list)) |
                                            (all_run_coeff[bin_sizes[0]]['Row'].isin(infra_list)) &
                                            (all_run_coeff[bin_sizes[0]]['Column'].isin(supra_list))]
gim_run_coeff = all_run_coeff[bin_sizes[0]][(all_run_coeff[bin_sizes[0]]['Row'].isin(gran_list)) &
                                            (all_run_coeff[bin_sizes[0]]['Column'].isin(infra_list)) |
                                            (all_run_coeff[bin_sizes[0]]['Row'].isin(infra_list)) &
                                            (all_run_coeff[bin_sizes[0]]['Column'].isin(gran_list))]

for i, bin_size in enumerate(bin_sizes[:-1]):

    pv_stat_coeff[bin_size] = all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(pv_list)) &
                                                       (all_stat_coeff[bin_size]['Column'].isin(pv_list))]
    pv_run_coeff[bin_size] = all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(pv_list)) &
                                                     (all_run_coeff[bin_size]['Column'].isin(pv_list))]

    pyr_stat_coeff[bin_size] = all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(pyr_list)) &
                                                        (all_stat_coeff[bin_size]['Column'].isin(pyr_list))]
    pyr_run_coeff[bin_size] = all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(pyr_list)) &
                                                      (all_run_coeff[bin_size]['Column'].isin(pyr_list))]

    infra_stat_coeff[bin_size] = all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(infra_list)) &
                                                          (all_stat_coeff[bin_size]['Column'].isin(infra_list))]
    gran_stat_coeff[bin_size] = all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(gran_list)) &
                                                         (all_stat_coeff[bin_size]['Column'].isin(gran_list))]
    supra_stat_coeff[bin_size] = all_stat_coeff[bin_size][(all_stat_coeff[bin_size]['Row'].isin(supra_list)) &
                                                          (all_stat_coeff[bin_size]['Column'].isin(supra_list))]

    infra_run_coeff[bin_size] = all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(infra_list)) &
                                                        (all_run_coeff[bin_size]['Column'].isin(infra_list))]
    gran_run_coeff[bin_size] = all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(gran_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(gran_list))]
    supra_run_coeff[bin_size] = all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(supra_list)) &
                                                        (all_run_coeff[bin_size]['Column'].isin(supra_list))]

    means_stat[i] = all_stat_coeff[bin_size].corr_coefficient.mean()
    means_run[i] = all_run_coeff[bin_size].corr_coefficient.mean()
    std_stat[i] = all_stat_coeff[bin_size].corr_coefficient.mean() /\
        np.sqrt(np.size(all_stat_coeff[bin_size].corr_coefficient))
    std_run[i] = all_run_coeff[bin_size].corr_coefficient.mean() /\
        np.sqrt(np.size(all_run_coeff[bin_size].corr_coefficient))

    means_stat_pv[i] = pv_stat_coeff[bin_size].corr_coefficient.mean()
    means_run_pv[i] = pv_run_coeff[bin_size].corr_coefficient.mean()
    std_stat_pv[i] = pv_stat_coeff[bin_size].corr_coefficient.mean() /\
        np.sqrt(np.size(pv_stat_coeff[bin_size].corr_coefficient))
    std_run_pv[i] = pv_run_coeff[bin_size].corr_coefficient.mean() /\
        np.sqrt(np.size(pv_run_coeff[bin_size].corr_coefficient))

    means_stat_infra[i] = infra_stat_coeff[bin_size].corr_coefficient.mean()
    std_stat_infra[i] = sp.stats.sem(infra_stat_coeff[bin_size].corr_coefficient)
    means_stat_gran[i] = gran_stat_coeff[bin_size].corr_coefficient.mean()
    std_stat_gran[i] = sp.stats.sem(gran_stat_coeff[bin_size].corr_coefficient)
    means_stat_supra[i] = supra_stat_coeff[bin_size].corr_coefficient.mean()
    std_stat_supra[i] = sp.stats.sem(supra_stat_coeff[bin_size].corr_coefficient)

    means_run_infra[i] = infra_run_coeff[bin_size].corr_coefficient.mean()
    std_run_infra[i] = sp.stats.sem(infra_run_coeff[bin_size].corr_coefficient)
    means_run_gran[i] = gran_run_coeff[bin_size].corr_coefficient.mean()
    std_run_gran[i] = sp.stats.sem(gran_run_coeff[bin_size].corr_coefficient)
    means_run_supra[i] = supra_run_coeff[bin_size].corr_coefficient.mean()
    std_run_supra[i] = sp.stats.sem(supra_run_coeff[bin_size].corr_coefficient)

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

    sgm_run_coeff = pd.concat([sgm_run_coeff,
                               all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(supra_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(gran_list)) |
                                                       (all_run_coeff[bin_size]['Row'].isin(gran_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(supra_list))]
                               ['corr_coefficient']], axis=1)
    sim_run_coeff = pd.concat([sim_run_coeff,
                               all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(supra_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(infra_list)) |
                                                       (all_run_coeff[bin_size]['Row'].isin(infra_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(supra_list))]
                               ['corr_coefficient']], axis=1)
    gim_run_coeff = pd.concat([gim_run_coeff,
                               all_run_coeff[bin_size][(all_run_coeff[bin_size]['Row'].isin(gran_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(infra_list)) |
                                                       (all_run_coeff[bin_size]['Row'].isin(infra_list)) &
                                                       (all_run_coeff[bin_size]['Column'].isin(gran_list))]
                               ['corr_coefficient']], axis=1)

plt.ioff()
fig, ax = plt.subplots()
noise_functions.simpleaxis(ax)
ax.plot(bin_sizes, means_run, linewidth=2, color='#ED1F24', label='Running')
ax.fill_between(bin_sizes, means_run - std_run, means_run + std_run, color='#ED1F24', alpha=0.5)
ax.plot(bin_sizes, means_stat, linewidth=2, color='#3852A4', label='Stationary')
ax.fill_between(bin_sizes, means_stat - std_stat, means_stat + std_stat, color='#3852A4', alpha=0.5)

ax.plot(bin_sizes, means_run_pv, linewidth=2, color='#ED1F24', linestyle='dashed', label='Running (PV)')
ax.fill_between(bin_sizes, means_run_pv - std_run_pv, means_run_pv + std_run_pv, color='#ED1F24', alpha=0.5)
ax.plot(bin_sizes, means_stat_pv, linewidth=2, color='#3852A4', linestyle='dashed', label='Stationary (PV)')
ax.fill_between(bin_sizes, means_stat_pv - std_stat_pv, means_stat_pv + std_stat_pv, color='#3852A4', alpha=0.5)

ax.legend(loc=2)
ax.set_xlabel('Bin Size (seconds)')
ax.set_ylabel('Pairwise Correlation Coefficient')
plt.savefig(data_path + 'nostim/figures/timescale.pdf', format='pdf')
plt.close()

plt.ioff()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
noise_functions.simpleaxis(ax1)
noise_functions.simpleaxis(ax2)
ax1.plot(bin_sizes, means_stat_supra, linewidth=2, color='#F05A22', label='Supragranular')
ax1.fill_between(bin_sizes, means_stat_supra - std_stat_supra, means_stat_supra + std_stat_supra,
                color='#F05A22', alpha=0.25)
ax1.plot(bin_sizes, means_stat_gran, linewidth=2, color='#20B473', label='Granular')
ax1.fill_between(bin_sizes, means_stat_gran - std_stat_gran, means_stat_gran + std_stat_gran,
                color='#20B473', alpha=0.25)
ax1.plot(bin_sizes, means_stat_infra, linewidth=2, color='#662F8F', label='Infragranular')
ax1.fill_between(bin_sizes, means_stat_infra - std_stat_infra, means_stat_infra + std_stat_infra,
                color='#662F8F', alpha=0.25)

ax2.plot(bin_sizes, means_run_supra, linewidth=2, color='#F05A22', label='Supragranular')
ax2.fill_between(bin_sizes, means_run_supra - std_run_supra, means_run_supra + std_run_supra,
                color='#F05A22', alpha=0.25)
ax2.plot(bin_sizes, means_run_gran, linewidth=2, color='#20B473', label='Granular')
ax2.fill_between(bin_sizes, means_run_gran - std_run_gran, means_run_gran + std_run_gran,
                color='#20B473', alpha=0.25)
ax2.plot(bin_sizes, means_run_infra, linewidth=2, color='#662F8F', label='Infragranular')
ax2.fill_between(bin_sizes, means_run_infra - std_run_infra, means_run_infra + std_run_infra,
                color='#662F8F', alpha=0.25)

ax2.legend(loc=4)
ax1.set_xlabel('Bin Size (seconds)')
ax1.set_ylabel('Correlation Coefficient')
ax1.set_ylim([-0.05, 0.30])
plt.savefig(data_path + 'nostim/figures/timescale_layer.pdf', format='pdf')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
noise_functions.simpleaxis(ax1)
noise_functions.simpleaxis(ax2)
plot_stat_sgm = sgm_stat_coeff['corr_coefficient']
plot_stat_sim = sim_stat_coeff['corr_coefficient']
plot_stat_gsm = gim_stat_coeff['corr_coefficient']

plot_run_sgm = sgm_run_coeff['corr_coefficient']
plot_run_sim = sim_run_coeff['corr_coefficient']
plot_run_gsm = gim_run_coeff['corr_coefficient']

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

ax2.plot(bin_sizes, plot_run_sgm.mean(), color='#F05A22', label='Supragranular/Granular')
ax2.fill_between(bin_sizes, plot_run_sgm.mean() - plot_run_sgm.sem(),
                 plot_run_sgm.mean() + plot_run_sgm.sem(),
                 color='#F05A22', alpha=0.5)

ax2.plot(bin_sizes, plot_run_sim.mean(), color='#20B473', label='Supragranular/Infragranular')
ax2.fill_between(bin_sizes, plot_run_sim.mean() - plot_run_sim.sem(),
                 plot_run_sim.mean() + plot_run_sim.sem(),
                 color='#20B473', alpha=0.5)

ax2.plot(bin_sizes, plot_run_gsm.mean(), color='#662F8F', label='Granular/Infragranular')
ax2.fill_between(bin_sizes, plot_run_gsm.mean() - plot_run_gsm.sem(),
                 plot_run_gsm.mean() + plot_run_gsm.sem(),
                 color='#662F8F', alpha=0.5)

ax1.set_xlabel('Counting Window (seconds)')
ax1.set_ylabel('Correlation Coefficient')
ax2.legend(loc=4)
ax1.set_ylim([0, 0.20])
plt.savefig(data_path + 'nostim/figures/corr_time_mix.pdf', format='pdf')
plt.close()

# Plot spike rates for running vs. stationary (scatter) as well as PV vs. non-PV to show differences
fig, ax = plt.subplots()
noise_functions.simpleaxis(ax)
unity = np.linspace(0, 30, 1000)
ax.scatter(spk_rates.stat, spk_rates.run, edgecolor='none')
ax.set_xlabel('Spikes per Second (Stationary)')
ax.set_ylabel('Spikes per Second (Running)')
spk_rate_run_sig = sp.stats.wilcoxon(spk_rates.stat, spk_rates.run)
ax.plot(unity, unity, '--', color='#E24A33')
ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
plt.savefig(data_path + 'nostim/figures/spk_rates_locomotion.pdf', format='pdf')
plt.close()

run_rates_mean = [spk_rates.run.mean(),
                  spk_rates[spk_rates.unit.isin(pv_list)].run.mean(),
                  spk_rates[spk_rates.unit.isin(pyr_list)].run.mean()
                  ]

run_rates_stde = [spk_rates.run.std() / spk_rates.shape[0],
                  spk_rates[spk_rates.unit.isin(pv_list)].run.std() / np.sqrt(spk_rates[spk_rates.unit.isin(pv_list)].shape[0]),
                  spk_rates[spk_rates.unit.isin(pyr_list)].run.std() / np.sqrt(spk_rates[spk_rates.unit.isin(pyr_list)].shape[0])
                  ]

stat_rates_mean = [spk_rates.stat.mean(),
                   spk_rates[spk_rates.unit.isin(pv_list)].stat.mean(),
                   spk_rates[spk_rates.unit.isin(pyr_list)].stat.mean()
                   ]

stat_rates_stde = [spk_rates.stat.std() / spk_rates.shape[0],
                   spk_rates[spk_rates.unit.isin(pv_list)].stat.std() / np.sqrt(spk_rates[spk_rates.unit.isin(pv_list)].shape[0]),
                   spk_rates[spk_rates.unit.isin(pyr_list)].stat.std() / np.sqrt(spk_rates[spk_rates.unit.isin(pyr_list)].shape[0])
                   ]

p_vals = np.array([sp.stats.wilcoxon(spk_rates.run, spk_rates.stat),
                   sp.stats.wilcoxon(spk_rates[spk_rates.unit.isin(pv_list)].run, spk_rates[spk_rates.unit.isin(pv_list)].stat),
                   sp.stats.wilcoxon(spk_rates[spk_rates.unit.isin(pyr_list)].run, spk_rates[spk_rates.unit.isin(pyr_list)].stat)])

fig, ax = plt.subplots()
noise_functions.simpleaxis(ax)

N = len(run_rates_mean)
ind = np.arange(N)
width = 0.25

rects1 = ax.bar(ind, stat_rates_mean, width, color='#3852A4', alpha=0.9, yerr=stat_rates_stde, ecolor='k',
                edgecolor='w')
rects2 = ax.bar(ind + width, run_rates_mean, width, color='#ED1F24', alpha=0.9, yerr=run_rates_stde, ecolor='k',
                edgecolor='w')

ax.set_ylabel('Mean Spike Rates (Hz)')
ax.set_xticks(ind + width)
ax.set_xticklabels(('All Units', 'PV', 'non-PV'))
ax.legend((rects1[0], rects2[0]), ('Stationary', 'Running'), loc=4)

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

    height = offset + np.max(run_rates_mean)
    bar_centers = ind[i] + np.array([0.5, 1.5]) * width
    noise_functions.significance_bar(bar_centers[0], bar_centers[1], height, display_string)

plt.savefig(data_path + 'nostim/figures/spk_rates_bar.pdf', format='pdf')
plt.close()

run_rates_mean_layer = [spk_rates[spk_rates.unit.isin(supra_list)].run.mean(),
                        spk_rates[spk_rates.unit.isin(gran_list)].run.mean(),
                        spk_rates[spk_rates.unit.isin(infra_list)].run.mean()
                        ]

run_rates_stde_layer = [spk_rates[spk_rates.unit.isin(supra_list)].run.sem(),
                        spk_rates[spk_rates.unit.isin(gran_list)].run.sem(),
                        spk_rates[spk_rates.unit.isin(infra_list)].run.sem()
                        ]

stat_rates_mean_layer = [spk_rates[spk_rates.unit.isin(supra_list)].stat.mean(),
                         spk_rates[spk_rates.unit.isin(gran_list)].stat.mean(),
                         spk_rates[spk_rates.unit.isin(infra_list)].stat.mean()
                         ]

stat_rates_stde_layer = [spk_rates[spk_rates.unit.isin(supra_list)].stat.sem(),
                         spk_rates[spk_rates.unit.isin(gran_list)].stat.sem(),
                         spk_rates[spk_rates.unit.isin(infra_list)].stat.sem()
                         ]

p_vals_layer = np.array([sp.stats.wilcoxon(spk_rates[spk_rates.unit.isin(supra_list)].run,
                                           spk_rates[spk_rates.unit.isin(supra_list)].stat),
                         sp.stats.wilcoxon(spk_rates[spk_rates.unit.isin(gran_list)].run,
                                           spk_rates[spk_rates.unit.isin(gran_list)].stat),
                         sp.stats.wilcoxon(spk_rates[spk_rates.unit.isin(infra_list)].run,
                                           spk_rates[spk_rates.unit.isin(infra_list)].stat)])

fig, ax = plt.subplots()
noise_functions.simpleaxis(ax)

N = len(run_rates_mean)
ind = np.arange(N)
width = 0.25

rects1 = ax.bar(ind, stat_rates_mean_layer, width, color='#3852A4', alpha=0.9, yerr=stat_rates_stde_layer, ecolor='k')
rects2 = ax.bar(ind + width, run_rates_mean_layer, width, color='#ED1F24', alpha=0.9, yerr=run_rates_stde_layer, ecolor='k')

ax.set_ylabel('Mean Spike Rates (Hz)')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Supragranular', 'Granular', 'Infragranular'))
ax.legend((rects1[0], rects2[0]), ('Stationary', 'Running'), loc=4)

y_lim = ax.get_ylim()
offset = (y_lim[1] - y_lim[0]) / 2
for i, p in enumerate(p_vals_layer):
    if p[1] >= 0.05:
        display_string = r'n.s.'
    elif p[1] < 0.001:
        display_string = r'***'
    elif p[1] < 0.01:
        display_string = r'**'
    else:
        display_string = r'*'

    height = offset + np.max(run_rates_mean)
    bar_centers = ind[i] + np.array([0.5, 1.5]) * width
    noise_functions.significance_bar(bar_centers[0], bar_centers[1], height, display_string)

plt.savefig(data_path + 'nostim/figures/spk_rates_bar_layer.pdf', format='pdf')
plt.close()
