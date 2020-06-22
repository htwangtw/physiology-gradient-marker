from pathlib import Path
import json

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

from util import interpolate_ibi


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

# download data
# with open(p / "data/neuroimaging_path_645.json", 'r') as f:
#     participants = json.load(f)

# for subject in list(participants.keys())[0:1]:
#     output_path = p / "data/derivatives" / f"sub-{subject}"
#     if not output_path.exists():
#         os.makedirs(p / "data/derivatives" / f"sub-{subject}")
#     items = participants[subject]
#     for s3_path in items:
#         filename = s3_path.split("/")
#         if filename[-1] == "residuals_antswarp.nii.gz":
#             filename = f"sub-{subject}{filename[-3]}{filename[-2]}.nii.gz"
#         else:
#             filename = filename[-1]
#         download_file = output_path / filename
#         with open(download_file, 'wb') as f:
#             s3_client.download_fileobj(s3_bucket_name, s3_path, f)

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

print("load data")
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

# imaging data
smoothed = smooth_img(nii, 5)
resampled_stat_img = resample_to_img(nii, str(atlas))
masker = NiftiLabelsMasker(str(atlas), standardize=True)
time_series = masker.fit_transform(resampled_stat_img)
n_tr = time_series.shape[0]
time_series = pd.DataFrame(time_series, index=tr * np.arange(n_tr), columns=None)

# sliding window
ts_windows = []
hrv_windows = []
for i in range(0, n_tr, int(window_size/2)):
    if (i + window_size) <= n_tr:
        # get the window for the cardiac recording
        t_start = i * tr + 4  # 4 seconds to account for hemodynamic delay
        t_end = (i + window_size) * tr + 4
        window = np.logical_and(physio.index >= t_start, physio.index <= t_end)

        # get time series of both modality
        ts_block = time_series.iloc[i: i + window_size].values
        cardiac_block = physio[window]
        peak_time = cardiac_block.query("cardiac_peak == 1").index.tolist()

        # IBI in milliseconds, clean ibi
        rr = interpolate_ibi(np.diff(peak_time) * 1000)

        # calculate some hrv for this window
        rmssd = np.std(np.diff(rr))

        # hrv in frequency domain        
        time = np.cumsum(rr)
        f = interpolate.interp1d(time, rr, kind='cubic')
        new_time = np.arange(time[0], time[-1], 1000/4)  # Sampling rate = 4 Hz
        rr_4hz = f(new_time)

        fft_sig = np.fft.fft(rr_4hz) / len(rr_4hz)
        frq = np.fft.fftfreq(len(rr_4hz), d=1/4)
        lf = np.trapz(abs(fft_sig[(frq>=0.04) & (frq<=0.15)]))
        hf = np.trapz(abs(fft_sig[(frq>=0.16) & (frq<=0.5)]))
        
        # save data for this window
        hrv_windows.append({"rmssd": rmssd, "lf-hrv": lf, "hf-hrv": hf})
        ts_windows.append(ts_block)

hrv_windows = pd.DataFrame(hrv_windows)

print("complete prerpocessing")
# tangent space
for k in ["tangent", "partial correlation"]:
    print(k)
    measure = ConnectivityMeasure(kind=k, vectorize=True, discard_diagonal=True)
    matrix = measure.fit_transform(ts_windows)
    if k == "tangent":
        mean = measure.mean_

    # fisher r-to-z transformation
    np.arctanh()


    hrv_fc = [] 
    for i in range(matrix.shape[1]):
        r = np.corrcoef(matrix[:, i].T, np.array(hrv_windows)[:, 0].T)[0, 1]
        hrv_fc.append(r) 
    fc_mat = vec_to_sym_matrix(np.array(hrv_fc), diagonal=np.zeros(400))
    np.fill_diagonal(fc_mat, 1)

    gradient_ready = yeo_sparse(fc_mat, percent=95)

    # Ask for 5 gradients (default)
    gm = GradientMaps(kernel='normalized_angle', approach="dm", n_components=3, random_state=0)
    gm.fit(gradient_ready)

    # First load mean connectivity matrix and Schaefer parcellation
    labeling = load_parcellation('schaefer', scale=400, join=True)
    surf_lh, surf_rh = load_conte69()

    mask = labeling != 0
    grad = [None] * 3
    for i in range(3):
        # map the gradient to the parcels
        grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan) * -1

    # viz
    plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap=mymap,
                    color_bar=True, label_text=[f'Grad{i+1}' for i in range(3)], zoom=1.55,
                    screenshot=True, filename=f"{k}_HRV_95.png")

print("static")
# mean tangent
gradient_ready = yeo_sparse(mean, percent=95)

# Ask for 5 gradients (default)
gm = GradientMaps(kernel='normalized_angle', approach="dm", n_components=3, random_state=0)
gm.fit(gradient_ready)

# First load mean connectivity matrix and Schaefer parcellation
labeling = load_parcellation('schaefer', scale=400, join=True)
surf_lh, surf_rh = load_conte69()

mask = labeling != 0
grad = [None] * 3
for i in range(3):
    # map the gradient to the parcels
    grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan) * -1

# viz
plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap=mymap,
                color_bar=True, label_text=[f'Grad{i+1}' for i in range(3)], zoom=1.55,
                screenshot=True, filename="mean_tangent_95.png")


# just gradient
for k in ["correlation", "partial correlation"]:
    correlation_measure = ConnectivityMeasure(kind=k)
    matrix = correlation_measure.fit_transform([time_series.values])
    # mean tangent
    gradient_ready = yeo_sparse(matrix[0], percent=95)

    # Ask for 5 gradients (default)
    gm = GradientMaps(kernel='normalized_angle', approach="dm", n_components=3, random_state=0)
    gm.fit(gradient_ready)

    # First load mean connectivity matrix and Schaefer parcellation
    labeling = load_parcellation('schaefer', scale=400, join=True)
    surf_lh, surf_rh = load_conte69()

    mask = labeling != 0
    grad = [None] * 3
    for i in range(3):
        # map the gradient to the parcels
        grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan) * -1
    
    # viz
    plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap=mymap,
                    color_bar=True, label_text=[f'Grad{i+1}' for i in range(3)], zoom=1.55,
                    screenshot=True, filename=f"static_{k}_95.png")
