import numpy as np
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import dyconnmap

# Setting some backend perameters
plt.matplotlib.use('Qt5Agg')

raw_BIDS = ''   # Path to data
subject_ids = ''    # Title for preprocessing plots

# Mark bad channels and annotate here
raw_BIDS.plot(block=True, title=str(subject_ids))

# Notch filter (50 Hz)
raw_notch = raw_BIDS.copy().notch_filter(np.arange(50, 500, 50))

# high pass filter set to 1 Hz (to 100 Hz)
raw_filtered = raw_notch.copy().filter(l_freq=1, h_freq=100)

# Virtual reference (average referencing here)
raw_avg_ref = raw_filtered.copy().set_eeg_reference(projection=True)
raw_avg_ref.apply_proj()

# downsample data from to 250 Hz
raw_downsampled = raw_filtered.resample(250, npad="auto")

# del raw
del raw_notch
del raw_filtered

# Independent Components Analysis (ICA) for artifact removal
ica = mne.preprocessing.ICA(n_components=32, random_state=0)
ica.fit(raw_downsampled)
ica.plot_sources(raw_downsampled, block=True, title=str(subject_ids))
ica.plot_components(title=str(subject_ids))

raw_ica = ica.apply(raw_downsampled.copy())
raw_ica.plot(block=True, title=str(subject_ids))


    ### Analysis ###

# PSD
raw_ica.psd()

# Connectivity
data = raw_ica.get_data(picks=['eeg'])
_, _, target_array = dyconnmap.analytic_signal(data, fb=[7., 12.], fs=250)
array = dyconnmap.fc.wpli(target_array, fs=250, fb=[7., 12.])

# Plotting connectivity
sns.heatmap(array)
