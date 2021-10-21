import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from tabulate import tabulate
start_time = time.time()

##Script will loop over array of Sn spectra and denoise with SWT

##Get High SNR ref spectrum for SSIM:
zf = 0#2**16 #global ZF
os.chdir( 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\10' ) #1k scan EXP
fid, SW = proc.loadfid('fid',plot='no')  
#fid = proc.em(fid,0.3)
#cpmg, fidcoadd, dumspec = proc.coadd(fid,lb = 0, plot='no')
#freq = proc.freqaxis(fidcoadd,SW)
ph = [95, 80480, -1750, 5600]
spec = proc.phase(proc.fft(fid,zf),ph)
#ph = proc.autophase(spec, 50, phase2='yes')
spec = np.real(spec)/np.max(np.real(spec))

freq = proc.freqaxis(fid,SW,zf)
a = simproc.nearest(freq,23.38) ##Allows me to change ZF and retain peak indices
b = simproc.nearest(freq,-175.94)

##Move on to main loop
path = 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\'
m = 8

SSIMin =  np.zeros(m)
SSIMout =  np.zeros(m)
specrecon = np.zeros((len(spec),m),dtype='complex64')
specin = np.zeros((len(spec),m),dtype='complex64')
snrpin = np.zeros(m)
snrpout =  np.zeros(m)
for i in range(m):
    kk = 10 + i
    print(kk)
    os.chdir( path + str(kk) )
    fid, SW = proc.loadfid('fid',plot='no')  
    
    specintemp = proc.phase(proc.fft(fid,zf),ph)
    specin[:,i] = np.real(specintemp)/np.max(np.real(specintemp))
    SSIMin[i] = simproc.ssim(spec,specin[:,i]) #SSIM of raw spec
    snrpin[i] = simproc.snrp(specin[:,i],a,b)

    ##Wavelet Denoising
    #fin_reg = wave.region_spec(np.real(spec), nthresh = 4.5, buff_size = 200, filter_size = 2)
    specrecont, coeffin, coeffs = wave.wavelet_denoise(3, np.real(specintemp), 0, wave = 'bior2.2', threshold = 'mod', alpha = 0)
    
    specrecon[:,i] = np.real(specrecont)/np.max(np.real(specrecont))
    SSIMout[i] = simproc.ssim(spec,specrecon[:,i])
    snrpout[i] = simproc.snrp(specrecon[:,i],a,b)

c =0.00003046# 0.00097758 #SSIMin of 1024 against itself has this difference from 1.0000
SSIMin = SSIMin - c
SSIMout = SSIMout - c

##Plotting
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14

plt.figure(1)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specin[:,i]) - i*np.max(np.real(specin[:,i])), label = 'SNRpp_in = %.1f , SSIM_in = %.4f' % (snrpin[i],SSIMin[i]))
plt.legend(loc='upper right')
plt.title('Noisey Spectra')
plt.xlabel('Frequency (kHz)')
plt.gca().invert_xaxis()
plt.yticks([])
plt.minorticks_on()

plt.figure(2)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specrecon[:,i]) - i*np.max(np.real(specrecon[:,i])), label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
plt.legend(loc='upper right')
plt.title('Denoised Spectra')
plt.xlabel('Frequency (kHz)')
plt.gca().invert_xaxis()
plt.yticks([])
plt.minorticks_on()

##Table SNR peak-peak
ns = [1024, 512, 256, 128, 64, 32, 16, 8]
data=[]
for i in range(len(snrpin)):
    data.append( ["%d"%ns[i],  "%.1f"%snrpin[i], "%.4f"%SSIMin[i], "%.1f"%snrpout[i], "%.4f"%SSIMout[i]] )
# create header
head = ['ns', 'SNRpp_in', 'SSIM_in', 'SNRpp_out', 'SSIM_out']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))
#generate ns counter at end

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))