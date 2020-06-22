import math

import numpy as np
import pandas as pd

from scipy.stats import zscore
from scipy import interpolate, signal

from systole.detection import rr_outliers

from tftb.processing import smoothed_pseudo_wigner_ville as spwvd

frequency_bands = {'vlf': ['Very low frequency', (0.003, 0.04), 'b'],
                   'lf': ['Low frequency', (0.04, 0.15), 'g'],
                   'hf': ['High frequency', (0.15, 0.4), 'r']}

def interpolate_ibi(ibi):
    '''
    Qucik wrapper for systole outlier detection
    and interpolate bad ibi
    outlier and ectobeats detection with method
    from Lipponen & Tarvainen (2019)
    '''

    ectobeats, outliers = rr_outliers(ibi)
    keep_idx = ~(ectobeats | outliers)
    time = np.cumsum(ibi)
    f = interpolate.interp1d(time[keep_idx],
                                ibi[keep_idx],
                                "cubic",
                                fill_value="extrapolate")
    rr = f(time)
    return rr


def resample(ibi, resample_fs=4):
    '''
    resample ibi to certain frequency with
    spline, 3rd order interpolation function
    '''
    time = np.cumsum(ibi) # in seconds
    # detrend
    detrend_ibi = signal.detrend(ibi, type='linear')
    detrend_ibi -= detrend_ibi.mean()

    # interpolate function (spline, 3rd order)
    f = interpolate.interp1d(time, detrend_ibi,
                            "cubic",
                            fill_value="extrapolate")
    sampling_time = 1 / resample_fs
    resample_time = np.arange(0, time[-1], sampling_time)
    ibi_resampled = f(resample_time)
    ibi_resampled -= ibi_resampled.mean()
    return ibi_resampled, resample_time


def spwvd_power(ibi_resampled, resample_time, resample_fs, tres=None, fres=None):
    '''
    tres :
        desired time resolution in seconds
    fres :
        desired frequency resolution in hz
    '''
    l = len(ibi_resampled)
    nfft = 2 ** _nextpower2(l) # Next power of 2 from length of signal
    nfreqbin = int(nfft / 4)  # number of frequency bins
    freq = (resample_fs / 2) * np.linspace(0, 1, nfreqbin) # normalised frequency 1 is fs / 2

    if all(r is None for r in [tres, fres]):
        print('default')
        # from this paper https://doi.org/10.1016/S1566-0702(00)00211-3
        twin_sample = 16
        fwin_sample = 128
    else:
        # smoothing window size in the number of samples
        delta_freq = np.diff(freq)[0]
        twin_sample = int(resample_fs * tres)
        fwin_sample = int(fres / delta_freq)

    # must be odd number
    twin_sample = round_up_to_odd(twin_sample)
    fwin_sample = round_up_to_odd(fwin_sample)

    # create smoothing window
    twindow = signal.hamming(twin_sample)
    fwindow = signal.hamming(fwin_sample)

    # power spectrum density spwvd
    trf = spwvd(ibi_resampled, resample_time,
                nfreqbin, twindow, fwindow)
    psd = trf ** 2
    return trf, psd, freq

def power_ampt(trf, psd, freq):
    """
    group signal by frequency band along time
    """
    # extract power amptitude in high and low frequency band
    power = []
    for f in frequency_bands.keys():
        lb = frequency_bands[f][1][0]
        ub = frequency_bands[f][1][1]
        idx_freq = np.logical_and(freq >= lb, freq < ub)
        print(idx_freq.shape)
        print(psd[idx_freq, :].shape)
        dx = np.diff(freq)[0]
        amptitude = np.trapz(y=psd[idx_freq, :],
                            dx=dx, axis=0)
        power.append(amptitude)
    vlf = power[0]
    lf = power[1]
    hf = power[2]
    return (vlf, lf, hf)
    
def _nextpower2(x):
    return 0 if x == 0 else math.ceil(math.log2(x))

def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)