# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:23:08 2021

@author: mason42
"""
import sys
sys.path.insert(0,'/Users/SRG/Documents/GitHub/SSNMR/functions')
import functions as proc
import numpy as np
import nmrglue.process.proc_base as ng
# import scipy.sparse as spar
import scipy.sparse.linalg as spla
from scipy.signal import hilbert
# from numba import jit_module
import matplotlib.pyplot as plt


# @jit(nopython=True) 
def delta(y, threshold):
    """Calculates threshold level"""
    s = np.abs(y) - threshold
    s = (s + np.abs(s))/2
    return np.sign(y)*s


# @jit(nopython=True) 
def IST_D(y, N, threshold = 0.99, max_iter = 10):
    """Iterative Soft Thresholding with Deletion
    
    N: int
        F1 zerofill 
    """
    
    nus = np.loadtxt('nuslist').astype('int') #Need to be in EXP folder
    F = np.fft.ifft(np.identity(N).astype('complex'))
    F = (np.fft.fftshift(F, axes = 1))
    A = spla.aslinearoperator(F[nus].T)
    sp_art = A @ y   
    A = spla.aslinearoperator(F[nus])
    t = threshold * np.max(np.abs(sp_art))
    X = np.zeros(sp_art.shape).astype('complex')
    r = y
    print('Iteration:')
    for i in range(max_iter):
        print ('%d / %d'%(i,max_iter))
        X += delta(sp_art, t)
        r = y - A@X[::-1]
        fX = np.zeros(sp_art.shape).astype('complex')
        fX[nus,:] = r
        sp_art =(ng.fft(fX.T).T)[::-1] #, 1, axis = 0)
        t = threshold * np.max(np.abs(sp_art))
        t = t * threshold * (max_iter - i)/max_iter
        
    h = hilbert(X.real, axis=0)  ##X has no imag component
    X = X.real - 1j*h.imag
    return X


def prep(fid, zf2, ph, gb=0, st=0):
    """(temp name).
    Pre-process the 2D NUS sampled FID.
    
    Parameters
    ----------
    fid : ndarray
        complex FID matrix
    zf2 : int
        zerofill on F2
    ph : list
        List of phases
    gb: float
        Gaussian broaden F2 that's centered at t = 0
    st : int
        For hypercomplex data: use st = 0 for standard complex, st = 1 for STATES,
        st = 2 for STATES-TPPI
    """
    
    g=open("acqu2s", mode='r'); lines=g.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$NusTD=': #Full t1 points
            np1 = int(lines[i].split()[1])
            
    nus = np.loadtxt('nuslist').astype('int') 
    
    datacln = np.zeros((np1, zf2)).astype('complex') 
    #Pre-process, standard complex acq
    if st == 0:
        datacln[nus , :fid.shape[1]] = fid  #This has the ordered nus entries 
        
        #FT t2
        f2proc = proc.phase(np.fft.fftshift(np.fft.fft(proc.gauss(datacln, gb, c=0,ax=0),axis=1)
                                        ,axes=1), ph, ax=1)
        fidnus = f2proc[nus,:]
    
    elif st != 0:
        idxn = np.zeros(len(nus)*2).astype('int')
        idxn[::2] = 2*nus; idxn[1::2] = 2*nus+1
        # datacln = np.zeros((zf1, zf2)).astype('complex') #zero-filled spectral matrix
        datacln[idxn, :fid.shape[1]] = fid 
        
        #states-tppi:
        f2proc = proc.phase(np.fft.fftshift(np.fft.fft(proc.gauss(datacln, gb, c=0,ax=1),axis=1)
                                            ,axes=1), ph)
        f1proc = f2proc[::2].real + 1j*f2proc[1::2].real
        if st == 2:
            f1proc[1::2] = -f1proc[1::2]
    
        fidnus = f1proc[nus] #These are the t1 points that were sampled in the FID
    return fidnus

# jit_module(nopython=True, error_model="numpy")

# def IST_S(y, N, threshold = 0.99, max_iter = 10):
#     # Iterative Soft Thresholding with Substitution
#     nus = np.loadtxt('nuslist').astype('int') #Need to be in EXP folder
#     y0 = np.zeros((N,y.shape[1])).astype('complex')
#     y0[nus] = y
#     F = np.fft.ifft(np.identity(N).astype('complex'))
#     F = spar.csc_matrix(np.fft.fftshift(F, axes = 1))
#     A = F[nus]
#     sp_art = A.T @ y
#     t = threshold * np.max(np.abs(sp_art))
#     X = np.zeros(sp_art.shape).astype('complex')
#     Z = np.setdiff1d(np.arange(N), nus)
#     for i in range(max_iter):
#         print(i)
#         X = delta(sp_art, t)
#         fX = F@X[::-1]
#         y0[Z] = fX[Z]
#         sp_art = np.roll(ng.fft(y0.T).T[::-1], 1, axis = 0)
#         t = threshold * np.max(np.abs(sp_art))
#         t = t * threshold * (max_iter - i)/max_iter
#     return sp_art