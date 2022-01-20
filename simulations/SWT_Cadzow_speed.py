# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import pywt
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
#start_time = time.time()

##Time Cadzow and SWT run times

#Iterations to run
kk = 6

k1 = 3 #Levels to decomp.
k2 = 7

cadsize = []; cad = []; wtsize = []; wt = []; wt2 = []
for i in range(kk):
    td = 2**(9+i)
    zf = td*4
    
    t = np.linspace(0,100,td)*1e-3 #sec
    
    T2 = 17*1e-3
    w1 = 0.5e3 #Hz
    w2 = -1e3 #Hz
    
    s = np.exp(-t/T2)*(np.exp(1j*t*w1*2*np.pi) + np.exp(1j*t*w2*2*np.pi) )
    s = simproc.noise(s,0.1)
    sf = proc.fft(s,zf)
    
    cadsize.append(len(s))
    wtsize.append(len(sf))
    
    cad0 = time.time()
    fiddum = proc.cadzow(s)
    cad.append( time.time() - cad0 )
    
    wt0 = time.time()
    specdum, coeffin, coeffs = wave.wavelet_denoise(k1, sf, 0)
    wt.append( time.time() - wt0 )
    
    wt02 = time.time()
    specdum, coeffin, coeffs = wave.wavelet_denoise(k2, sf, 0)
    wt2.append( time.time() - wt02 )
    
######Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14

plt.plot(cadsize,cad,'kx--',label = 'Cadzow/SVD')
plt.plot(wtsize,wt,'mo--',label = 'SWT w/ %d levels'%k1)
plt.plot(wtsize,wt2,'c*--',label = 'SWT w/ %d levels'%k2)
plt.xlabel('Input Data Size (# pts.)')
plt.ylabel('Computation Time (s)')
plt.legend(loc='upper right')
plt.grid(b=True, which='major', axis='both')
plt.yscale('symlog')

print('Finished!')