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
# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')

# get list of mat files and unit info from CSV in data folder
bin_size = 0.3
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
all_stat_coeff, all_run_coeff = (pd.DataFrame() for i in range(2))
unit_counter = 0
vis_info = {}
pv_list, pyr_list, supra_list, gran_list, infra_list = ([] for i in range(5))

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
        except TypeError:
            spk_times = pd.DataFrame(vis_info[channel_ind].Spk_times_s, columns=['spike times'])
            stat_times = vis_info[channel_ind].Stat_times
            mov_times = vis_info[channel_ind].Mov_times
            exp_dur = vis_info[channel_ind].Exp_dur

        # get bins for pd.cut function by flattening stat and making sure that stat is first or adjusting accordingly
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

    all_stat_coeff = pd.concat([all_stat_coeff, stat_counts_corr], axis=0)
    all_run_coeff = pd.concat([all_run_coeff, run_counts_corr], axis=0)

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
    noise_functions.main_plots(data_path + 'nostim/', 'n', all_stat_coeff, all_run_coeff, pv_list, pyr_list)

    noise_functions.layer_plots(data_path + 'nostim/', 'n', all_stat_coeff, all_run_coeff, supra_list,
                                gran_list, infra_list)

    noise_functions.layer_type_plots(data_path + 'nostim/', 'n', all_stat_coeff, all_run_coeff, supra_list,
                                     gran_list, infra_list, pv_list, pyr_list)

