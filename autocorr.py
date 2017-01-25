import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Arial']
plt.style.use('ggplot')

data_path = '/Volumes/anupam/Amanda Data/noise_correlation/resort/'  # path to all data
file = data_path + 'nostim/20130117_info.mat'
unit_data = pd.read_csv(data_path + 'unit_info.csv')  # unit data from CSV in data folder

vis_info = sio.loadmat(file, squeeze_me=True, struct_as_record=False)['info']

all_spikes = vis_info[0, 0].all_spikes
spk_times = vis_info[0, 0].Spk_times_s
exp_dur = vis_info[0, 0].Exp_dur

exp_dur *= 1000
exp_dur += 1
spk_times *= 1000

# bin_size = 1
# n_bins = np.round(exp_dur)
#
# Signal = np.zeros(n_bins)
# Signal[np.floor(spk_times/bin_size).astype('int')] = 1.
# auto_corr = np.correlate(Signal, Signal, mode='full')

ISI_forward = spk_times[1:]-spk_times[:-1]
ISI_reverse = spk_times[0:-1]-spk_times[1:]
ISI = np.concatenate([ISI_forward, ISI_reverse])
bins = np.arange(-15, 16)

plt.ioff()
plt.hist(ISI, bins=bins, color='#000000')
plt.xlabel('Time Lag (ms)')
plt.ylabel('Number of Spikes')
plt.savefig(data_path + 'nostim/figures/corr.pdf', format='pdf')
plt.close()

plt.plot(all_spikes[:, np.random.choice(all_spikes.shape[1], size=75)] * 1e6, color='#000000')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (uV)')
plt.savefig(data_path + 'nostim/figures/wave.pdf', format='pdf')
plt.close()
plt.ion()
