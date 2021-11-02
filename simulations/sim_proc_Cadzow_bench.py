# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import urQRd1 as proc2
import simpson as simproc
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tabulate import tabulate
start_time = time.time()

T2 = 0.1 #fake T2 global, unitless
fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='no')

ph = [558.67, 92160.09, 0, 0]
gb = 16 ##Amount of global gaussian broadening for coadd

fid = simproc.coadd(fid, 512, 50)
fidref = proc.gauss( fid , lb = gb )
specref = proc.phase(proc.fft(fidref),ph)

freq = simproc.freqaxis(specref,1e6)
a = simproc.nearest(freq,-146.7)
b= simproc.nearest(freq,293.8)

kk = 8 ##Number of noisy data points

snrpin = np.zeros(kk); snrpout =  np.zeros(kk); snrin =  np.zeros(kk)
snrout =  np.zeros(kk);SSIMin =  np.zeros(kk) ; SSIMout =  np.zeros(kk); res =  np.zeros(kk)
specrecon = np.zeros((len(specref),kk),dtype='complex64')
specin = np.zeros((len(specref),kk),dtype='complex64')
for i in range(kk):
    fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='no') #dup for coadd
    fid = simproc.noise(fid,(i+1)*0.1)
    fid = simproc.coadd(fid, 512, 50)                   #dup for coadd
    fid2 = np.copy(fid) #dup for coadd
    fid = proc.gauss(fid, lb = gb)
    spec = proc.phase(proc.fft(fid),ph)
    specin[:,i] = spec
    
    snrpin[i] = simproc.snrp(spec,a,b) 
    snrin[i] = proc.snr(spec,j=0)
    SSIMin[i] = simproc.ssim(specref,spec) #problem here that spec is not GB'd

    fidrecon = proc.cadzow(fid2, p=35) ##Slow Cadzow
    fidrecon = proc.gauss(fidrecon,gb)
    
    specrecon[:,i] = proc.phase(proc.fft(fidrecon),ph)
    snrpout[i] = simproc.snrp(specrecon[:,i],a,b)
    SSIMout[i] = simproc.ssim(specref,specrecon[:,i])
    

c = SSIMin[0] - 1
SSIMin -= c; SSIMout -= c

######Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14

plt.figure(1)
for i in range(len(snrin)):
    plt.plot(freq,np.real(np.flipud(specin[:,i])) - i*np.max(np.real(specin[:,i])), label = 'SNRpp_in = %.1f , SSIM_in = %.4f' % (snrpin[i],SSIMin[i]))
plt.legend(loc='upper right')
plt.title('Noisy Spectra')
plt.xlabel('Frequency (kHz)')
plt.xlim((250, -400))
plt.minorticks_on()
plt.yticks([])

plt.figure(2)
for i in range(len(snrin)):
    plt.plot(freq,np.real(np.flipud(specrecon[:,i])) - i*np.max(np.real(specrecon[:,i])), label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
plt.legend(loc='upper right')
plt.title('Denoised Spectra')
plt.xlabel('Frequency (kHz)')
plt.xlim((250, -400))
plt.minorticks_on()
plt.yticks([])

##Table SNR peak-peak
data=[]
for i in range(len(snrin)):
    data.append( ["%.1f"%snrpin[i], "%.1f"%snrpout[i], "%.4f"%SSIMin[i],  "%.4f"%SSIMout[i]] )
# create header
head = ['SNRpp_in', 'SNRpp_out', 'SSIM_in', 'SSIM_out']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))

##Table SNR
# data=[]
# for i in range(len(snrin)):
#     data.append( ["%.1f"%snrin[i], "%.4f"%SSIMin[i], "%.1f"%snrout[i], "%.4f"%SSIMout[i]] )
# # create header
# head = ['SNR_in','SSIM_in', 'SNR_out', 'SSIM_out']
# # display table
# print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))



print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))