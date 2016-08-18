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

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')

bin_size = 0.3
plt_units = 'n'
plt_summary = 'y'
data_path = '/Volumes/anupam/Amanda Data/resort_nostim/'

# create folder for figures
if not os.path.exists(data_path + 'figures'):
    os.makedirs(data_path + 'figures')

mat_files = list(glob.iglob(data_path + '*.mat'))
unit_data = pd.read_csv(data_path + 'unit_info_nostim.csv')

layers = {1: 'supragranular', 2: 'granular', 3: 'infragranular'}
unit_data = unit_data.replace({"layer": layers})

all_stat_coeff = pd.DataFrame()
all_run_coeff = pd.DataFrame()

unit_counter = 0
vis_info = {}
pv_list = []
pyr_list = []
supra_list = []
gran_list = []
infra_list = []

for file in mat_files:
    date_stat_counts = pd.DataFrame()
    date_run_counts = pd.DataFrame()
    date = int(ntpath.basename(file)[:8])

    print(str(datetime.now()) + " Currently processing: " + str(date))

    vis_info = sio.loadmat(file, squeeze_me=True, struct_as_record=False)['info']

    date_units = unit_data[unit_data.date == date]
    vis_index = noise_functions.get_chans(vis_info)

    for unit in date_units.iterrows():
        unit = unit[1]
        layer = unit.layer

        channel = unit['channel']
        unit_number = unit['unit_number']
        channel_ind = int(vis_index[(vis_index.channel == channel) & (vis_index.cluster == unit_number)]['ind1'].values)
        unit_ind = int(vis_index[(vis_index.channel == channel) & (vis_index.cluster == unit_number)]['ind2'].values)

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

        bins = stat_times.flatten('F')

        first = ['stat', 'run']
        if np.max(bins) != exp_dur:
            bins = np.append(bins, exp_dur)
        if bins[0] != 0:
            bins = np.insert(bins, 0, np.zeros(1), axis=0)
            first = ['run', 'stat']

        spk_times['movement'] = np.where(pd.cut(spk_times['spike times'], bins, labels=False) % 2 == 1,
                                         first[1], first[0])

        run_counts = pd.DataFrame()
        stat_counts = pd.DataFrame()
        for i, bin in enumerate(bins[:-1]):
            new_bins = np.arange(bins[i], bins[i+1], bin_size)
            spk_cut = pd.cut(spk_times['spike times'], bins=new_bins)
            spk_counts = pd.value_counts(spk_cut).reindex(spk_cut.cat.categories)

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

    if plt_units == 'y':
        plt.ioff()
        plt.matshow(date_stat_counts.corr())
        plt.savefig(data_path + 'figures/' + str(date) + '_stat.pdf', format='pdf')
        plt.close()

        plt.matshow(date_run_counts.corr())
        plt.savefig(data_path + 'figures/' + str(date) + '_run.pdf', format='pdf')
        plt.close()
        plt.ion()

if plt_summary == 'y':
    noise_functions.main_plots(data_path, all_stat_coeff, all_run_coeff, pv_list, pyr_list)

    noise_functions.layer_plots(data_path, all_stat_coeff, all_run_coeff, supra_list, gran_list, infra_list)

    noise_functions.layer_type_plots(data_path, all_stat_coeff, all_run_coeff, supra_list, gran_list, infra_list,
                                     pv_list, pyr_list)

