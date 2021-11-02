import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from tabulate import tabulate
start_time = time.time()

##Script will loop over array of Sn spectra
##Get High SNR ref spectrum for SSIM:
gb = 22 #global gaussian broaden
zf = 0
os.chdir( 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\10' ) #1k scan EXP
fid, SW = proc.loadfid('fid',plot='no')  
cpmg, fidcoadd = proc.coadd(fid, plot='no')
#cpmg, fidcoadd, dumspec = proc.coaddg(fid, plot='no')
fidcoadd = fidcoadd[1:]  #Weird glitch, need to throw out first point to balance
fidcoadd = proc.gauss(fidcoadd,gb)
ph = [367, 52161, -1636, 100]
#ph = [367, 52161, -1636, 100]
spec = proc.phase(proc.fft(fidcoadd),ph)
#plt.plot(np.real(spec))
#sys.exit()
spec = np.real(spec)/np.max(np.real(spec))
freq = proc.freqaxis(spec)
b = simproc.nearest(freq,-170.9)
a= simproc.nearest(freq,24.38)
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

    cpmg, fidcoadd = proc.coadd(fid, plot='no') #save lb for cadzow
    fidcoadd = fidcoadd[1:]
    
    specintemp = proc.phase(proc.fft(proc.gauss(fidcoadd,gb)),ph)
    specin[:,i] = np.real(specintemp)/np.max(np.real(specintemp))
    SSIMin[i] = simproc.ssim(spec,specin[:,i]) #SSIM of raw spec
    snrpin[i] = simproc.snrp(specin[:,i] ,a,b)

    #Denoise
    fidrecon = proc.cadzow(fidcoadd, 10)
    fidrecon = proc.gauss(fidrecon,gb)
    
    specrecont = proc.phase(proc.fft(fidrecon),ph)
    specrecon[:,i] = np.real(specrecont)/np.max(np.real(specrecont))
    SSIMout[i] = simproc.ssim(spec,specrecon[:,i])
    snrpout[i] = simproc.snrp(specrecon[:,i],a,b)

c = SSIMin[0] - 1 #SSIMin of 1024 against itself has this difference from 1.0000

SSIMin = SSIMin - c
SSIMout = SSIMout - c

plt.figure(1)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specin[:,i]) - i*np.max(np.real(specin[:,i])), label = 'SNRpp_in = %.1f , SSIM_in = %.4f' % (snrpin[i],SSIMin[i]))
plt.legend(loc='upper right')
plt.title('Noisy Spectra')
plt.xlabel('Frequency (kHz)')
plt.xlim((100, -250))
plt.minorticks_on()
plt.yticks([])

plt.figure(2)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specrecon[:,i]) - i*np.max(np.real(specrecon[:,i])), label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
plt.legend(loc='upper right')
plt.title('Denoised Spectra')
plt.xlabel('Frequency (kHz)')
plt.xlim((100, -250))
plt.minorticks_on()
plt.yticks([])


##Table SNR peak-peak
ns = [1024, 512, 256, 128, 64, 32, 16, 8]
data=[]
for i in range(len(snrpin)):
    data.append( ["%d"%ns[i],  "%.1f"%snrpin[i], "%.1f"%snrpout[i], "%.4f"%SSIMin[i],  "%.4f"%SSIMout[i]] )
# create header
head = ['ns', 'SNRpp_in', 'SNRpp_out', 'SSIM_in', 'SSIM_out']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))
#generate ns counter at end

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))