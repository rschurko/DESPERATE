# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:23:08 2021

@author: mason42
"""

import numpy as np, pylab as plt
import nmrglue.fileio.bruker as br
import nmrglue.process.proc_base as proc
import scipy.sparse as spar
import scipy.sparse.linalg as spla

def delta(y, threshold):
    # Calculates threshold level
    s = np.abs(y) - threshold
    s = (s + np.abs(s))/2
    return np.sign(y)*s

def IST_D(y, nus, N, threshold = 0.99, max_iter = 10):
    # Iterative Soft Thresholding with Deletion
    # Substantially faster than IST_S
    F = np.fft.ifft(np.identity(N).astype('complex'))
    F = (np.fft.fftshift(F, axes = 1))
    A = spla.aslinearoperator(F[nus])
    sp_art = A.T @ y
    t = threshold * np.max(np.abs(sp_art))
    X = np.zeros(sp_art.shape).astype(complex)
    r = y
    for i in range(max_iter):
        print (i)
        X += delta(sp_art, t)
        r = y - A@X[::-1]
        fX = np.zeros(sp_art.shape).astype('complex128')
        fX[nus,:] = r
        sp_art =(proc.fft(fX.T).T)[::-1] #, 1, axis = 0)
        t = threshold * np.max(np.abs(sp_art))
        t = t * threshold * (max_iter - i)/max_iter
    return X

def IST_S(y, nus, N, threshold = 0.99, max_iter = 10):
    # Iterative Soft Thresholding with Substitution
    y0 = np.zeros((N,y.shape[1])).astype('complex')
    y0[nus] = y
    F = np.fft.ifft(np.identity(N).astype('complex'))
    F = spar.csc_matrix(np.fft.fftshift(F, axes = 1))
    A = F[nus]
    sp_art = A.T @ y
    t = threshold * np.max(np.abs(sp_art))
    X = np.zeros(sp_art.shape).astype('complex')
    Z = np.setdiff1d(np.arange(N), nus)
    for i in range(max_iter):
        print(i)
        X = delta(sp_art, t)
        fX = F@X[::-1]
        y0[Z] = fX[Z]
        sp_art = np.roll(proc.fft(y0.T).T[::-1], 1, axis = 0)
        t = threshold * np.max(np.abs(sp_art))
        t = t * threshold * (max_iter - i)/max_iter
    return sp_art

N = 4096; N2 = 4096
dct, data = br.read('24')
nus = np.loadtxt('24/nuslist').astype('int')
idxn = np.zeros(len(nus)*2).astype('int')
idxn[::2] = 2*nus; idxn[1::2] = 2*nus+1
data = data.reshape((26,2048))
datacln = np.zeros((N*2, N2)).astype('complex')
datacln[idxn, :2048] = data 

datacln[:,:4026] = br.remove_digital_filter(dct, datacln)
datacln[:,1024:] = 0
f2proc = proc.ps(proc.fft(proc.em(datacln, lb=0.0005)), p0=-130, p1=100.4)

f1proc = f2proc[::2].real + 1j*f2proc[1::2].real
f1proc[1::2] = -f1proc[1::2]
f1nus = f1proc[nus]

recon = IST_S(f1nus, nus, N, max_iter=20)
