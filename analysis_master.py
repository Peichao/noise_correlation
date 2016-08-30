import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')

data_path = 'D:/noise_correlation/resort/'
vstim_mat = list(glob.iglob(data_path + 'vstim/*.mat'))
nostim_mat = list(glob.iglob(data_path + 'nostim/*.mat'))
unit_data = pd.read_csv(data_path + 'unit_info.csv')
bin_size = 0.3  # bin size for spike counts in seconds

# create folder for figures
if not os.path.exists(data_path + 'figures'):
    os.makedirs(data_path + 'figures')

# analyze nostim data
