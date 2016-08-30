import glob
from datetime import datetime
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

# get list of mat files and unit info from CSV in data folder
bin_size = 0.3
plt_units = 'y'  # plot individual correlelograms?
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
all_stat_coeff, all_run_coeff = (pd.DataFrame for i in range(2))
unit_counter = 0
vis_info = {}
pv_list, pyr_list, supra_list, gran_list, infra_list = ([] for i2 in range(5))

for file in mat_files:
    date_stat_counts, date_run_counts = (pd.DataFrame() for i3 in range(2))

    # get date from filename (first 8 characters)
    date = int(ntpath.basename(file)[:8])
    # print message of date to the console
    print(str(datetime.now()) + " Currently processing: " + str(date))

    # load mat file for date
    vis_info = sio.loadmat(file, squeeze_me=True, struct_as_record=False)['vis_info']

    # filter units that belong to date
    date_units = unit_data[unit_data.date == date]

    # iterate through each unit (fix to remove pandas iteration?)
    for unit in date_units.iterrows():
        unit = unit[1]
        layer = unit.layer  # get unit layer as string
        channel = unit['channel']  # get unit channel number
        unit_number = unit['unit_number']  # get unit number

        channel_ind = channel - 1
        unit_ind = unit_number - 1

        trial_num = pd.DataFrame(vis_info[channel_ind, unit_ind].trials_num)
        spk_times = pd.DataFrame(vis_info[channel_ind, unit_ind].spk_trial)
        stat_trials = vis_info[channel_ind, unit_ind].trials_ori_Loff_stat
        mov_trials = vis_info[channel_ind, unit_ind].trials_ori_Loff_run
        ori = vis_info[channel_ind, unit_ind].deg
        stim_time = vis_info[channel_ind, unit_ind].stim_time
        fsample = vis_info[channel_ind, unit_ind].fsample

        vresp_sig = vis_info[channel_ind, unit_ind].vresp_sig
        osi_sig = vis_info[channel_ind, unit_ind].osi_sig

        stim_start_samp = int(np.round(stim_time[0] * fsample))
        stim_end_samp = int(stim_start_samp + np.round(stim_time[1] * fsample))
        stim_spk_times = spk_times[stim_start_samp:stim_end_samp]

        bin_samples = np.round(bin_size * fsample)
        stim_spk_times_bin = stim_spk_times.groupby(stim_spk_times.index // bin_samples).sum()

