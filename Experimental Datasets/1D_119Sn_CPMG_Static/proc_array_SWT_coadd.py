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

##Get High SNR ref spectrum for SSIM:
zf = 2048
gb = 16 #global gaussian broaden
os.chdir( 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\10' ) #1k scan EXP
fid, SW = proc.loadfid('fid',plot='no')  
cpmg, fidcoadd, dumspec = proc.coadd(fid,lb = 0, plot='no')
#freq = proc.freqaxis(fidcoadd,SW)
fidcoadd = proc.gauss(fidcoadd,gb)
#ph = [371.283066651318, 44952.976, -1690.231670275709, 99.559]
ph = [368, 45314, -1636, 199]
spec = proc.phase(proc.fft(fidcoadd,zf),ph)
spec = np.real(spec)/np.max(np.real(spec))  ##UNCOMMENT
#plt.plot(np.real(spec))
#sys.exit()
freq = proc.freqaxis(spec,SW,zf)
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

    cpmg, fidcoadd, dumspec = proc.coadd(fid,lb = 0, plot='no') #save lb for cadzow
    
    specintemp = proc.phase(proc.fft(proc.gauss(fidcoadd,gb),zf),ph)
    specin[:,i] = np.real(specintemp)/np.max(np.real(specintemp))
    
    SSIMin[i] = simproc.ssim(spec,specin[:,i]) #SSIM of raw spec
    snrpin[i] = simproc.snrp(specin[:,i] ,a,b)

    #Denoise
    specrecont, coeffin, coeffs = wave.wavelet_denoise(7, np.real(specintemp), 0, wave = 'bior2.2', threshold = 'mod', alpha = 0)
    
    specrecon[:,i] = np.real(specrecont)/np.max(np.real(specrecont))
    SSIMout[i] = simproc.ssim(spec,specrecon[:,i])
    snrpout[i] = simproc.snrp(specrecon[:,i],a,b)

c = 0.0004889999999999617 #SSIMin of 1024 against itself has this difference from 1.0000
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

os.chdir(path)
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))