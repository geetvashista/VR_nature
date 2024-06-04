import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne

# Setting some backend perameters
plt.matplotlib.use('Qt5Agg')

file = r'C:\Users\em17531\Desktop\VR\data.dat'
raw = mne.io.read_raw_curry(file, preload=True)
raw_BIDS = raw.crop(tmin = 60, tmax = 360)
subject_ids = 1

raw_BIDS.plot(block=True, title=str(subject_ids))

# Notch filter (50 Hz)
raw_notch = raw_BIDS.copy().notch_filter(np.arange(50, 500, 50))

# high pass filter set to 1 Hz (to 100 Hz)
raw_filtered = raw_notch.copy().filter(l_freq=8, h_freq=12)

# Virtual reference (average referencing here)
raw_avg_ref = raw_filtered.copy().set_eeg_reference(projection=True)
raw_avg_ref.apply_proj()

# downsample data from to 250 Hz
raw_downsampled = raw_avg_ref.resample(500, npad="auto")

# del raw
del raw_notch
del raw_filtered

# Independent Components Analysis (ICA) for artifact removal
ica = mne.preprocessing.ICA(n_components=20, random_state=0)
ica.fit(raw_downsampled)
ica.plot_sources(raw_downsampled, block=True, title=str(subject_ids))
ica.plot_components(title=str(subject_ids))

raw_ica = ica.apply(raw_downsampled.copy())
raw_ica.plot(block=True, title=str(subject_ids))