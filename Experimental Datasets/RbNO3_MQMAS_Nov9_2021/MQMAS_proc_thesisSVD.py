# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import scipy
start_time = time.time()

cwd =  os.getcwd()
os.chdir(cwd + '\\' + '21')
fid = proc.loadfid2('ser',plot='no')

#Params:
nzf1 = 256 #512
nzf2 = 2048 #4096
base = 2 #base contour %'age

##SH = +7/9 for this coherence selection
spec = proc.mqproc(fid, SH = 7/9, zf1=nzf1, zf2=nzf2, lb1=0, lb2=5) 

#Phase
ph = [243 + 10, 797420 + 140, 0, 0]
spec = proc.phase(spec,ph,ax=1)

#2D SWT
#spec, coeffin, coeffs = wave.wavelet_denoise2(2, np.real(spec), 0, wave = 'bior2.2', threshold = 'mod', alpha = 0)

#SNRs
a = np.unravel_index(spec.argmax(), spec.shape)
snrF2 = proc.snr(spec[a[0],:],1000)
snrF1 = proc.snr(spec[:,a[1]],150)

##Matrix plots
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14
mpl.rcParams['pdf.fonttype'] = 42

fid = spec[80:200,650:1150] #Remove some chunkz

U, s, Vt = np.linalg.svd(fid) #s is a vector of singular values, not a matrix, Vt is already transposed

fid = fid /np.max(fid)

m,n = fid.shape
sigma = scipy.linalg.diagsvd(s, m, n) #rebuilds s as sigma matrix

##discarding bit
r = 10
s2 = s[0:(r)] #discard r compon.
sigma2 = scipy.linalg.diagsvd(s2, r, r) #rebuilds s as sigma matrix
U2 = U[:,:r]
V2 = Vt[:r,:]

fid2 = np.dot(U2,np.dot(sigma2,V2))
#b = np.dot(sigma, Vh) ##replace with np.matmul or @ 
#z = np.dot(U[:,:r], b[:r,:]) #retain 'r' principal components

cmap = 'jet'
alpha = 1

# plt.figure(1)
# plt.subplot(141)
# plt.imshow(np.abs(fid),cmap)
# plt.subplot(142)
# plt.imshow(np.abs(U),cmap)
# plt.subplot(143)
# plt.imshow(np.abs(sigma),cmap)
# plt.subplot(144)
# plt.imshow(np.abs(Vt),cmap)

# plt.figure(2)
# plt.subplot(141)
# plt.imshow(np.abs(fid2),cmap)
# plt.subplot(142)
# plt.imshow(np.abs(U2),cmap)
# plt.subplot(143)
# plt.imshow(np.abs(sigma2),cmap)
# plt.subplot(144)
# plt.imshow(np.abs(V2),cmap)

plt.subplot(241)
plt.imshow(np.abs(fid),cmap)
plt.subplot(242)
plt.imshow(np.abs(U),cmap)
plt.subplot(243)
plt.imshow(np.abs(sigma),cmap)
plt.subplot(244)
plt.imshow(np.abs(Vt),cmap)

plt.subplot(245)
plt.imshow(np.abs(fid2),cmap)
plt.subplot(246)
plt.imshow(np.abs(U2),cmap)
plt.subplot(247)
plt.imshow(np.abs(sigma2),cmap)

# plt.colorbar()

plt.subplot(248)
plt.imshow(np.abs(V2),cmap)

# plt.colorbar()

os.chdir(cwd)
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))