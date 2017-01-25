import glob
import ntpath
import os
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')

data_path = 'D:/noise_correlation/resort/'
vstim_path = data_path + 'vstim'
mat_files = list(glob.iglob(vstim_path + '*.mat'))
unit_data = pd.read_csv(data_path + 'unit_info.csv')

# create folder for figures
if not os.path.exists(data_path + 'figures'):
    os.makedirs(data_path + 'figures')

vis_info = {}
for file in mat_files:
    date = int(ntpath.basename(file)[:8])
    print("Currently processing: " + str(date))
    vis_info = sio.loadmat(file, squeeze_me=True, struct_as_record=False)['vis_info']

    # create folder for date
    if not os.path.exists(data_path + str(date)):
        os.makedirs(data_path + str(date))

    date_units = unit_data[unit_data.date == date]

    for unit in date_units.iterrows():
        unit = unit[1]
        unit_info = vis_info[unit['channel_python']][unit['unit_python']]

        trial_num = pd.DataFrame(unit_info.trials_num, index=None, columns=['ori'])

        trials_run = np.sort(np.concatenate(unit_info.trials_ori_Loff_run.tolist()).ravel()) - 1,
        trials_stat = np.sort(np.concatenate(unit_info.trials_ori_Loff_stat.tolist()).ravel()) - 1
        trial_num['movement'] = np.arange(trial_num.count()['ori'])
        trial_num['movement'].loc[trials_run] = 'run'
        trial_num['movement'].loc[trials_stat] = 'stat'
        trial_num['movement'] = trial_num['movement'].replace(trials_run, 'run')
        trial_num['movement'] = trial_num['movement'].replace(trials_stat, 'stat')
        trial_num.loc[trial_num.ori == 256, ['movement']] = 'blank'

        deg = unit_info.deg
        f_sample = unit_info.fsample
        pre_bins = int(np.round(unit_info.stim_time[0] / (1/f_sample)))
        post_bins = int(np.round(unit_info.stim_time[2] / (1/f_sample)))

        subset = trial_num[['ori', 'movement']]
        trial_tuples = [tuple(x) for x in subset.values]

        multi_index = pd.MultiIndex.from_tuples(trial_tuples, names=('ori', 'movement'))
        stim_spk = pd.DataFrame(unit_info.spk_trial[pre_bins:-post_bins], columns=multi_index)

        stim_spk_sum = stim_spk.sum(axis=0)
        stim_spk_rate = stim_spk_sum / unit_info.stim_time[1]

        means = stim_spk_rate.groupby(level=['ori', 'movement']).mean()

        plt.ioff()
        if 256 in means:
            del means[256]
        means.unstack().plot(linewidth=2)
        plt.title('%d, Channel %d, Unit %d' % (date, unit.channel, unit.unit_number))
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Spike Rate (spikes/second)')
        plt.savefig(data_path + str(date) + '/%d_%d_%d.pdf' % (date, unit.channel, unit.unit_number), format='pdf')
        plt.close()

# -----------------------------------------
