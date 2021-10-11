# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:38:17 2021

@author: mason42
"""

import numpy as np, pylab as plt
import nmrglue.fileio.bruker as br
from scipy.ndimage.filters import maximum_filter
import pywt
from phasecorr import *
from wavelet_denoise import *

def extract_windows(data, window_size):
    start = window_size
    sub_windows = (
        start +
        np.expand_dims(np.arange(window_size), 0) +
        np.expand_dims(np.arange(len(data)-2*window_size), 0).T
        )
    return data[sub_windows]

dct, data = br.read_pdata('31//pdata//1')
#data[:,1890:2172] =  np.mean(data[:,2172:]) *  np.random.normal(0, 1, data[:,1890:2172].shape)

test = maximum_filter(data, size = 5)
npts = [data.shape[0]//16, data.shape[1]//16]
tempNoise = np.zeros((16,16))
thresh = 3
for i in range(16):
    for k in range(16):
        tempNoise[i,k] = np.std(test[(i*npts[0]):npts[0]*(i+1),
                               (k*npts[1]):npts[1]*(k+1)])
noise = thresh*np.mean(tempNoise)
wndw = 30
wdw = extract_windows(test, wndw)
wdw = np.concatenate((np.zeros((wndw,wndw,data.shape[1])),
                      wdw, 
                      np.zeros((wndw,wndw,data.shape[1]))), axis = 0)

mn = np.max(np.abs(wdw), axis = 1) - np.min(np.abs(wdw), axis = 1)
region_spec = 1-np.array([mn < noise][0]).astype(int)

wave = 'bior2.2'; level = 5
coeffs=pywt.swt2(data, wave, level = level)
for i in range(len(coeffs)):
     temp = coeffs[i]
     alpha = 1
     lam = calc_lamb(temp[0], region_spec)
     fincomp0 = hard_threshold(temp[0], lam)
     fincomp1 = hard_threshold(temp[1][0], lam)
     fincomp2 = hard_threshold(temp[1][1], lam)
     fincomp3 = hard_threshold(temp[1][2], lam)
     coeffs[i] = (fincomp0, (fincomp1, fincomp2, fincomp3))

final = pywt.iswt2(coeffs, wave)
#%%
plt.plot(data[1140]/np.max(data[1140])+1)
plt.plot(final[1140]/np.max(data[1140]))

#%%
plt.figure()
spc = np.linspace(np.max(final)/10, np.max(final), 20)
plt.contour(final, levels = spc)

plt.figure()
spc2 = np.linspace(np.max(data)/10, np.max(data), 20)
plt.contour(data, levels = spc2)