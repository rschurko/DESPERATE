# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import urQRd1 as proc2
import simpson as simproc
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
start_time = time.time()

def snrp(spec,i,j):
    """Peak-to-peak SNR. Need to know the indicies of the max [i] and min[j] peaks."""
    
    spec = np.real(spec)
    sn = ( spec[i] - spec[j] ) / np.std(spec[0:100])
    print('SNRp = %.3f' %sn)
    return sn

def ssim(specref, measure):
    X = np.real( measure )
    Y = np.real( specref )
    SSIM = (2*np.mean(X)*np.mean(Y) + 0)*( 2*np.cov(X,Y)[0][1] + 0) /( (np.mean(X)**2 + np.mean(Y)**2 +0)*(np.std(X)**2 +np.std(Y)**2 + 0))
    return SSIM

T2 = 0.1 #fake T2 global, unitless
fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='no')

ph = [558.67, 92160.09, 0, 0]

gb = 10 ##Amount of global gaussian broadening for coadd

fid, spec = simproc.coadd(fid, 512, 50)
fidref = proc.gauss( fid , lb = gb )
specref = proc.phase(proc.fft(fidref),ph)[0,:]

kk = 11 ##Number of noisy data points

snrpin = np.zeros(kk)
snrpout =  np.zeros(kk)
snrin =  np.zeros(kk)
snrout =  np.zeros(kk)
SSIMin =  np.zeros(kk)
SSIMout =  np.zeros(kk)
res =  np.zeros(kk)
specrecon = np.zeros((4096,kk),dtype='complex64')
specin = np.zeros((4096,kk),dtype='complex64')
for i in range(kk):
    fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='no') #dup for coadd
    fid = simproc.noise(fid,(i+1)*0.1)
    fid, spec = simproc.coadd(fid, 512, 50)                   #dup for coadd
    fid = proc.gauss(fid , lb = gb)
    spec = proc.phase(proc.fft(fid),ph)[0,:]
    specin[:,i] = spec
    
    #spec = proc.phase(spec,ph)[0,:]
    snrpin[i] = snrp(spec,1446,3249)
    snrin[i] = proc.snr(spec,j=0)
    
    #fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='no') #dup for coadd
    #fid = simproc.noise(fid,(i+1)*0.1)
    #fid = proc.gauss(fid , lb = gb )
    #spec = proc.phase(proc.fft(fid),ph)[0,:]
    SSIMin[i] = ssim(specref,spec) #problem here that spec is not GB'd

    fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='no') #dup again for cadzow
    fid = simproc.noise(fid,(i+1)*0.1)
    fid, spec = simproc.coadd(fid, 512, 50)                   #dup again for cadzow
    #fid = proc.gauss(fid , lb = gb)

    fidrecon = simproc.cadzow(fid,ph,SW,lb = gb) ##Slow Cadzow
    plt.close()
    specrecon[:,i] = proc.phase(proc.fft(fidrecon),ph)[0,:]
    snrpout[i] = snrp(specrecon[:,i],1446,3249)

    SSIMout[i], res[i], snrout[i] = simproc.residual(fidref,fidrecon,SW,ph,plot='no')
    
    del fid
    del fidrecon
    #del spec
    #del specrecon
#specrecon = simproc.coadd(fidrecon,512,50)
#plt.plot(np.abs(specrecon))
plt.close()

plt.figure(1)
for i in range(len(snrin)):
    plt.plot(np.real(specin[:,i]) - i*np.max(np.real(specin[:,i])), label = 'SNRpp_in = %.1f , SSIM_in = %.4f' % (snrpin[i],SSIMin[i]))
plt.legend(loc='upper right')
plt.title('Noisey Spectra')

plt.figure(2)
for i in range(len(snrin)):
    plt.plot(np.real(specrecon[:,i]) - i*np.max(np.real(specrecon[:,i])), label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
plt.legend(loc='upper right')
plt.title('Denoised Spectra')

##Table SNR
data=[]
for i in range(len(snrin)):
    data.append( ["%.1f"%snrin[i], "%.4f"%SSIMin[i], "%.1f"%snrout[i], "%.4f"%SSIMout[i]] )
# create header
head = ['SNR_in','SSIM_in', 'SNR_out', 'SSIM_out']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))

##Table SNR peak-peak
data=[]
for i in range(len(snrin)):
    data.append( ["%.1f"%snrpin[i], "%.4f"%SSIMin[i], "%.1f"%snrpout[i], "%.4f"%SSIMout[i]] )
# create header
head = ['SNRpp_in', 'SSIM_in', 'SNRpp_out', 'SSIM_out']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))