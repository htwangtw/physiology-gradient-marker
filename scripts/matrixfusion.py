from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

import matplotlib.colors as colors
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy import interpolate, signal
from scipy.ndimage import median_filter

import nibabel as nb
from nilearn.image import resample_to_img, smooth_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure, vec_to_sym_matrix
from nilearn import plotting

from systole.hrv import time_domain, frequency_domain

from detecta import detect_peaks

from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

from util import interpolate_ibi, resample, spwvd_power, power_ampt


def yeo_sparse(fc_mat, percent=95):
    top = []
    n = fc_mat.shape[0]
    for i in range(n):
        cur = fc_mat[i, :]
        top_edges = cur >= np.percentile(cur, percent)
        top.append(top_edges.astype(int))
    top = np.array(top)
    fc_mat *= top
    return fc_mat

p = Path.home() / "projects/gradient-physiology"

tr = 0.645
window_size = 24
sampling_rate = 62.5

atlas = p / "references/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
physio_path = p / "data/derivatives/sub-A00034074/sub-A00034074_ses-DS2_task-rest_acq-645_physio.tsv.gz"
nii_path = p / "data/derivatives/sub-A00034074/sub-A00034074_scan_rest_acq-645_selector_CSF-2mmE-M_aC-WM-2mm-DPC5_G-M_M-SDB_P-2.nii.gz"
nii = nb.load(str(nii_path))

physio = pd.read_table(physio_path, header=None, names=["cardiac", "respiratory"],
                       compression="gzip")

physio.index = 1 / sampling_rate * np.arange(physio.shape[0])

# use a median filter to despike
physio.cardiac = median_filter(physio.cardiac.values, 5)
cardiac = physio.cardiac
# get peaks
peak_idx = detect_peaks(zscore(cardiac), mph=0, mpd=20)
physio["cardiac_peak"] = 0
physio.iloc[peak_idx, -1] = 1

# prepro
peak_time = physio.query("cardiac_peak == 1").index.tolist()
ibi = interpolate_ibi(np.diff(peak_time))
rr, resample_time = resample(ibi, 1/tr)
stop = np.where(resample_time >= 900 * tr)[0][0]  # should be 900
trf, psd, freq = spwvd_power(rr[:stop], resample_time[:stop], 1/tr)
vlf, lf, hf = power_ampt(trf, psd, freq)

# imaging data
smoothed = smooth_img(nii, 5)
resampled_stat_img = resample_to_img(nii, str(atlas))
masker = NiftiLabelsMasker(str(atlas), standardize=True)
time_series = masker.fit_transform(resampled_stat_img)
n_tr = time_series.shape[0]
time_series = pd.DataFrame(time_series, index=tr * np.arange(n_tr), columns=None)
time_series["hfHRV"] = hf
time_series["lfHRV"] = lf

correlation_measure = ConnectivityMeasure(kind='correlation')
pcorr_matrix = correlation_measure.fit_transform([time_series.values])

def fusion(*args):
    from scipy.stats import rankdata
    from sklearn.preprocessing import minmax_scale

    max_rk = [None] * len(args)
    masks = [None] * len(args)
    for j, a in enumerate(args):
        m = masks[j] = a != 0
        a[m] = rankdata(a[m])
        max_rk[j] = a[m].max()

    max_rk = min(max_rk)
    for j, a in enumerate(args):
        m = masks[j]
        a[m] = minmax_scale(a[m], feature_range=(1, max_rk))

    return np.hstack(args)

fc_fuse = fusion(pcorr_matrix[0, :400, :400], pcorr_matrix[0, :400, 400:])

# Ask for 5 gradients (default)
gm = GradientMaps(kernel='normalized_angle', approach="pca", n_components=3, random_state=0)
gm.fit(fc_fuse)

# First load mean connectivity matrix and Schaefer parcellation
labeling = load_parcellation('schaefer', scale=400, join=True)
surf_lh, surf_rh = load_conte69()

mask = labeling != 0
grad = [None] * 3
for i in range(3):
    # map the gradient to the parcels
    grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan) * -1

# Modified color map from Margulies et al 2016
first = int((128 * 2)-np.round(255 * (1.- 0.7)))
second = (256-first)
colors2 = plt.cm.viridis(np.linspace(0.1, .98, first))
colors3 = plt.cm.YlOrBr(np.linspace(0.25, 1, second))
cols = np.vstack((colors2,colors3))
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)

# visualise the color bar
# num = 256
# cbar = range(num)
# for x in range(5):
#     cbar = np.vstack((cbar, cbar))
# fig, ax = plt.subplots(nrows=1)
# ax.imshow(cbar, cmap=mymap, interpolation='nearest')
# ax.set_axis_off()
# fig.tight_layout()
# plt.show()

# viz
plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap=mymap,
                 color_bar=True, label_text=[f'Grad{i+1}' for i in range(3)], zoom=1.55,
                 screenshot=True, filename=f"static_fusion_95.png")