# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:44:43 2021

@author: mason42
"""

import numpy as np, pylab as plt
import nmrglue.fileio.bruker as br
import matplotlib
from scipy import linalg as la, signal as sig
import pywt
from phasecorr import *
from wavelet_denoise import *


def region_spec_2D(data, nthresh = 4.5, buff_size = 200, filter_size = 4):
    F2proj = np.max(data, axis = 0); F1proj = np.max(data, axis = 1)
    F2reg = region_spec(F2proj, nthresh = nthresh, 
                        buff_size = buff_size, filter_size = filter_size).reshape((1,y))
    F1reg = region_spec(F1proj, nthresh = nthresh, 
                        buff_size = buff_size, filter_size = filter_size).reshape((x,1))
    twoDreg = F2reg*F1reg
    return twoDreg

dct, data = br.read_pdata('2//pdata//1')
#data += 0.05*np.max(data) * np.random.normal(0, 1, data.shape)
x,y = data.shape
offF2= dct['procs']['OFFSET']
swpF2 = dct['procs']['SW_p'] / dct['acqus']['SFO1']
F2 =  np.linspace(offF2, offF2-swpF2, num=y)

offF1= dct['proc2s']['OFFSET']
swpF1 = dct['proc2s']['SW_p'] / dct['acqu2s']['SFO1']
F1 =  np.linspace(offF1, offF1-swpF1, num=x)

dct, datarg = br.read_pdata('501//pdata//1')
region_spec = region_spec_2D(data, nthresh = 4.5, buff_size = 5, filter_size=4)
spc = np.linspace(np.max(data)/2, np.max(data), 15)
#%%
plt.contour(data/np.max(data)+region_spec, )
plt.show()
#%%
wave = 'bior2.2'; level = 5
F1proj = np.max(data, axis = 0)
coeffs=pywt.swt2(data, wave, level = level)
for i in range(len(coeffs)):
     temp = coeffs[i]
     alpha = 1
     lam = calc_lamb(temp[0], region_spec)
     fincomp0 = mod_thresh(temp[0], lam, alpha)
     fincomp1 = mod_thresh(temp[1][0], lam, alpha)
     fincomp2 = mod_thresh(temp[1][1], lam, alpha)
     fincomp3 = mod_thresh(temp[1][2], lam, alpha)
     coeffs[i] = (fincomp0, (fincomp1, fincomp2, fincomp3))

final = pywt.iswt2(coeffs, wave)
#%%
plt.plot(np.max(data, axis = 1))
plt.plot(np.max(final, axis = 1))
#%%
matplotlib.rcParams['xtick.direction']='out'
matplotlib.rcParams['ytick.direction']='out'
matplotlib.rcParams['xtick.major.size']='2'
matplotlib.rcParams['ytick.major.size']='2'
matplotlib.rcParams['xtick.minor.size']='1'
matplotlib.rcParams['ytick.minor.size']='1'
matplotlib.rcParams['xtick.labelsize']='6'
matplotlib.rcParams['ytick.labelsize']='6'
matplotlib.rcParams['font.sans-serif']='Arial'
matplotlib.rcParams['lines.linewidth']='0.2'
matplotlib.rcParams['axes.linewidth'] = '0.5'
matplotlib.rcParams['grid.linewidth'] = '0.1'
spc = np.linspace(np.max(data)/15, np.max(data), 15)
plt.figure(1, figsize=(4,4), dpi=300)
ax1 = plt.axes([.2,.2,.6,0.6])
plt.contourf(F2, F1, final, levels = spc)
plt.contour(F2, F1, final, levels = spc, colors='k')

plt.setp(ax1, xlim=(200,20), ylim=(11,0))
xmajloc = plt.MultipleLocator(20); xminloc = plt.MultipleLocator(2)
ax1.xaxis.set_major_locator(xmajloc); ax1.xaxis.set_minor_locator(xminloc)
ymajloc = plt.MultipleLocator(2); yminloc = plt.MultipleLocator(0.4)
ax1.yaxis.set_major_locator(ymajloc); ax1.yaxis.set_minor_locator(yminloc)
plt.xlabel(r'$^\mathregular{13}$C Chemical Shift (ppm from TMS)', fontsize = 10)
plt.ylabel(r'$^\mathregular{1}$H Chemical Shift (ppm from TMS)', fontsize = 10)

b = plt.axes([.2,.81,.6,.17])
plt.plot(F2, np.max(final, axis = 0), 'k-')
plt.setp(b, xlim=(200,20), frame_on = False, xticks=[], yticks=[])

c = plt.axes([.81,.2,.17,.6])
plt.plot(np.max(final, axis = 1),F1, 'k')
plt.setp(c, ylim=(11,0), frame_on = False, xticks=[], yticks=[])
plt.show()

"""plt.figure()
plt.subplot(121)
plt.plot(np.max(data, axis = 0)/np.max(data)+2)
plt.plot(np.max(data, axis = 0)/np.max(data)+1)
plt.plot(np.max(final, axis = 0)/np.max(final))
plt.plot(np.max(region_spec, axis = 0)+2.1)
plt.subplot(122)
plt.plot(np.max(data, axis = 1)/np.max(data)+2)
plt.plot(np.max(datarg, axis = 1)/np.max(datarg)+1)
plt.plot(np.max(final, axis = 1)/np.max(final))
plt.plot(np.max(region_spec, axis = 1)+2.1)"""
#%%
