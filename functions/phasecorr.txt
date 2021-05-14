# -*- coding: utf-8 -*-
"""
This code contains a series of general automatic processing routines for complex NMR data

Autophasing broadly follows the routine set forth in:
A robust, general automatic phase correction algorithm for high‐resolution NMR data
Zorin et. al. Magn. Reson. Chem. 2017, 55, 738–746



Created on Tue Mar 24 11:14:03 2020

@author: mason42
"""

import numpy as np, pylab as plt
import nmrglue.process.proc_base as proc
from scipy.optimize import fmin,curve_fit


def global_phase(s0, thr=10, buff=200, filt=50, corr=False):
    """
    Final phasing algorithm
    Determines the zero order phase (ph0) needed to phase correct each region of the spectrum
    Plots the ph0 values as a function of the relative position of the region in the spectrum
    Fits the ph0 values to a linear model to determine first order phase correction
    for total spectum
    
    Parameters
    ----------
    s0 : ndarray
        Complex FFT NMR data for phasing
    thr : int
        optional : sets the amount of points to consider for filing gaps
        default  : 10 pts
    buff : int
        optional : sets the amount of signal generated outside of region
        default  : 200 pts
    filt : int
        optional : sets the minimum number of points to be considered a peak
        default  : 50 pts
    corr : bool
        optional : True plots the correlated phase points and best fit line
    
    Returns
    -------
    ph : ndarray
        zero and first order phase in degrees
    pdata : ndarray
        final phased data
    """
    
    data=np.copy(s0)  #copy data so as to not inadvertantly overwrite array
    region = find_peak_region(data.real, thresh=thr)  #call region function
    idx = peak_indices(region, buffer_size=buff, filter_size=filt) #call indexing function
    output = np.zeros((len(idx), 3))  #empty array for storage
    #loop through the data regions, autophase (p0 only), and store range max, pH, 
    for k in range(len(idx)):  
        test = data[idx[k,0]:idx[k,1]]
        x_axis = np.linspace(0, 1, num=len(data))
        xtrun = x_axis[idx[k,0]:idx[k,1]]
        ## subtract a linear baseline from each region prior to autophasing
        pts = np.array([[test[2], test[-2]],
                [xtrun[2], xtrun[-2]]])
        slope = (pts[0,1]-pts[0,0])/(pts[1,1]-pts[1,0])
        inter = pts[0,1]-slope*pts[1,1]
        base = line(slope, inter, xtrun)
        test -= base
        s0sum = np.max(test.real)  #weighting factor essentially the area
        ph0 = calc_phase(test)  #calc ph0 only using acme algorithm
        #need to invert phase if they are negative to get straight line
        if ph0 < 0:
            ph0+=360
        fin = (np.exp(1j*ph0*np.pi/180.)*test).real
        ppmmax=xtrun[np.argmax(fin)]
        output[k] = [ppmmax, ph0, s0sum]
    if len(output[:,0]) > 1:
        # fitting the resulting curve with  weighted linear model: slope= ph1 and intercept = p0
        ph, pcov = curve_fit(line, output[:,0], output[:,1], sigma=1/output[:,2]) #uncomment for weighted fit
        if corr==True:
            plt.figure()
            plt.plot(output[:,0], output[:,1], 'ro')
            plt.plot(x_axis, line(ph[1], ph[0], x_axis))
    else:
        ph = np.array([output[0,1], 0])
    pdata = proc.ps(s0, p0=ph[0], p1=ph[1]) #final phased data
    return ph, pdata

def baseline_corr(s0, prc = True, thr = 10, buff = 200, filt = 50, smpts=1000):
    """
    Automatic baseline correction algorithm
    Determines the baseline from a peak list 
    
    Parameters
    ----------
    s0 : ndarray
        FFT NMR data can be complex or real
    prc : bool
        optional : if False the baseline as produced as the inverse FFT of the baseline for FID subtraction
        default  : True
    thr : int
        optional : sets the amount of points to consider for filing gaps
        default  : 10 pts
    buff : int
        optional : sets the amount of signal generated outside of region
        default  : 200 pts
    filt : int
        optional : sets the minimum number of points to be considered a peak
        default  : 50 pts
    smpts: int
        number of points to use for box smoothing of the baseline points
    
    Returns
    -------
    base_corr_data : ndarray
        prc = True
            The spectrum s0 baseline corrected
        prc = False
            The inverse FFT of the smoothed baseline for use in FID subtraction
    """
    
    baseline = np.copy(s0)
    regions = find_peak_region(s0.real, nthresh=4.5)
    idx = peak_indices(regions, buffer_size=buff, filter_size=filt)
    x_axis = np.linspace(0,len(s0), len(s0))
    for i in range(len(idx)):
        pts = np.array([[s0[idx[i,0]-1], s0[idx[i,1]+1]],
                        [x_axis[idx[i,0]-1], x_axis[idx[i,1]+1]]])
        slope = (pts[0,1]-pts[0,0])/(pts[1,1]-pts[1,0])
        inter = pts[0,1]-slope*pts[1,1]
        baseline[idx[i,0]:idx[i,1]] = line(slope, inter, x_axis[idx[i,0]:idx[i,1]])
    if prc == True:
        base_corr_dta = s0 - smooth(baseline, smpts)
    else:
        base_corr_dta = np.zeros(len(s0)).astype('complex')
        base_corr_dta[:len(s0)//2] = proc.ifft(smooth(baseline, smpts))[:len(s0)//2]
    return base_corr_dta 

def fiddle(sexp, fid, x_axis, finlen,  width=0.007, prange = [0.2, -0.2]):
    """
    FIDDLE algorithm for reference deconvolution
    assumes Lorentzian lineshape
    
    Parameters
    ----------
    sexp : ndarray
        complex processed NMR spectrum
    fid : ndarray
        original FID for correction
    x_axis : ndarray
        x axis scale in ppm for the processed spectrum
    finlen
        final length of FID to output
    width : float
        optional : width of the final reference peak
        default  : 0.007
    prange : ndarray
        optional : ppm range containing reference peak (in NMR order)
        default : [0.2, -0.2] assumes DSS/TSP/DSS peak at 0 ppm
    
    Returns
    -------
    final : ndarray
        complex FID data corrected using FIDDLE
    """

    idx = np.where(np.logical_and(x_axis<prange[0], x_axis>prange[1])) #find range
    idx_max = np.argmax(sexp[idx])  #find maximum intensity for peak
    sidl = lorentz(sexp[idx][idx_max], x_axis[idx][idx_max], width, x_axis)  #generate ideal linehsape
    sspc = np.zeros(len(sexp)).astype('complex')  #empty complex array for storage
    sspc[idx] = sexp[idx]  #place the reference portion in empty array
    sref = proc.ifft(sspc)[:len(sexp)//2] #extract only first half of ifft of ref data
    sinv = proc.ifft(sidl)[:len(sexp)//2] #extract only first half of ifft of ideal data
    trun = fid[:len(sexp)//2]*sinv/sref  #final multiplication for fiddle 
    final = np.zeros(finlen).astype(complex) #empty complex array for storage
    final[:len(sexp)//2] = trun
    return final

def base_deriv(s0):
    """
    Function to determine signal derivative
    Uses equation 5 of Zorin et. al. Magn. Reson. Chem. 2017, 55, 738–746
    
    Parameters
    ----------
    s0  : ndarray
        The real component of a FFT array
    Returns
    -------
    deriv : ndarray
        The derivitive of the data, s0
    """
    deriv = np.zeros((len(s0)))
    for i in range(5,len(s0)-5):
        deriv[i] = (42*(s0[i] - s0[i-1]) + 48*(s0[i+1]-s0[i-2]) + 27*(s0[i+2]-s0[i-3]) + 8*(s0[i+3]-s0[i-4]) + s0[i+4]-s0[i-5])/512
    return deriv

def find_peak_region(signal, thresh = 10, nthresh = 4.5):
    """
    Finds the peaks in the real data
    Follows a hybrid proceedure 
    Uses a moving window approach as used to determine if a region belongs to a peak
    
    Parameters
    ----------
    sign  : ndarray
        derivitive of a FFT NMR data set
    thresh : int
        threshold for gaps in peaks as number of points
        default is 10 pts
    nthresh : float
        threshold use to define peak levels
        default is 4.5 times the noise level
    Returns
    -------
    regions : ndarray
        A binary array where 1 represents a peak region and 0 represents baseline
        
    """
    s0 = np.copy(signal)
    regions = np.zeros(len(s0))
    deriv = base_deriv(s0)
    #determine noise level for thresholding
    sn=np.int(0); nspts = len(s0) // 16
    tempNoise = np.zeros(16)
    for i in range(16):
        tempNoise[i] = np.std(deriv[sn+(i*nspts):nspts*(i+1)])
    noise = np.min(tempNoise)
    noise *= nthresh
    # use moving window to determine peak regions
    window = np.int(len(s0)*0.0015)
    regions = np.zeros(len(s0))
    for i in range(window, len(s0)-window):
        minr = np.min(np.abs(deriv[i-window:i+window]))
        maxr = np.max(np.abs(deriv[i-window:i+window]))
        delt = maxr-minr
        if delt > noise:
            regions[i] = 1
    #fill holes in the binary regions array
    for k in range(thresh,len(regions)-thresh):
        pn = regions[k+thresh-1]; nn = regions[k-thresh]
        #look thresh points away from the point of interest to fill holes and remove outliers
        if np.logical_and(regions[k] != pn, regions[k] != nn):
            regions[k] = pn
    return regions

def peak_indices(regions, buffer_size = 200, filter_size = 50):
    """
    Algorithm to transform binary region array into peak region indices
    and to merge and trim peak regions
    
    Parameters
    ----------
    regions : ndarray
        A binary array where 1 represents a peak region and 0 represents baseline
    buffer_size : int
        number of points to add to either side of a peak region
    filter_size : int
        minium size for a peak region in number of points
    Returns
    -------
    indx : ndarray
        A two by n column matrix defining the start and end of a peak region
    """
    
    init_indx = np.where(regions == 1)[0]
    cons_indx = consecutive(init_indx)
    idx = np.array([]).astype(int)  #empty array for storage
    # loop through the consecutive blocks and find minimum and max values and add buffer points
    for i in range(len(cons_indx)):
        if len(cons_indx[i]) > filter_size:
            idx = np.append(idx, [int(np.min(cons_indx[i])-buffer_size), int(np.max(cons_indx[i])+buffer_size-1)])

    idx = idx.reshape((len(idx)//2, 2))

    # following code determines region overlaps and merges overlaping regions
    n, mn, mx = 1, 0, 0

    for k in range(0, len(idx)):
        if k == 0:
            mn = idx[0,0]
            mx = idx[0,1]
        elif idx[k,0]<=mx:  #there is overlap
            mx = idx[k,1]   #current max is always greater
            n += 1  #increment number in current group
            #print("n=", n, " max=", mx, " min=", mn)
            if k == len(idx)-1:  #last item in the array
                if n>1:
                    for i in range(0, n):  #set all prior items in group
                        idx[k-i, 0] = mn
                        idx[k-i, 1] = mx
        else:  #not overlapping; new group item 1
            if n>1:
                for i in range(1, n+1):  #set all prior items in group
                    idx[k-i, 0] = mn
                    idx[k-i, 1] = mx
            mx = idx[k, 1]
            mn = idx[k, 0]
            n = 1
    idx = np.unique(idx, axis = 0)
    return idx

def ps_acme_score(p0, data):
    """
    Auto phase algorithm ACME 
    Adapted from nmrglue package to only do zero order phase optimization
    
    Parameters
    ----------
    p0 : float
        intial guess for zero order phase correction
    data : ndarray
        complex FFT data to be phased
    
    Returns
    -------
    score : float
        Value of the objective function (phase score)
    """
    
    stepsize = 1

    phc0 = p0*np.pi/180.

    s0 = np.exp(1j*phc0)*data
    data = np.real(s0)

    # Calculation of first derivatives
    ds1 = np.abs((data[1:]-data[:-1]) / (stepsize*2))
    p1 = ds1 / np.sum(ds1)

    # Calculation of entropy
    p1[p1 == 0] = 1

    h1 = -p1 * np.log(p1)
    h1s = np.sum(h1)

    # Calculation of penalty
    pfun = 0.0
    as_ = data - np.abs(data)
    sumas = np.sum(as_)

    if sumas < 0:
        pfun = pfun + np.sum((as_/2) ** 2)

    p = 1000 * pfun

    return h1s + p

def calc_phase(s0, p0=0):
    """
    Minimization routine for zero order phase correction
    uses scipy.fmin for minimization routine
    
    Parameters
    ----------
    s0 : ndarray
        Complex FFT NMR data for phase correction
    Returns
    ----------
    opt : float
        optimized value for zero order correction
    """
    
    opt = p0
    fn = ps_acme_score #peak_minima_score
    opt = fmin(fn, x0=opt, args=(s0, ), disp=0)
    return opt

def consecutive(data, stepsize=1):
    """"
    basic function to blocks of consecutive number in an array
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def lorentz(amp, pos, fwhm, x):
    """
    basic function for a Lorentian lineshape
    """
    return amp/(1.0+((x-pos)/(fwhm/2.0))**2)

def line(m, b, x):
    """
    basic function defining a line
    """
    return m*x + b

def smooth(y, box_pts):
    """
    basic function for box smoothing  of data
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
