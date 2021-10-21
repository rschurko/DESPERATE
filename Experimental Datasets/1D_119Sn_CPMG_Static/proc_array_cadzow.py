import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import urQRd1 as proc2
import time
import matplotlib.pyplot as plt
from tabulate import tabulate
start_time = time.time()

##Script will loop over array of Sn spectra
def snrp(spec,i,j):
    """Peak-to-peak SNR. Need to know the indicies of the max [i] and min[j] peaks."""
    
    spec = np.real(spec)
    sn = np.abs( spec[i] - spec[j] ) / np.std(spec[0:100])
    print('SNRp = %.3f' %sn)
    return sn

##Also used highest signal avg as SSIM 
def ssim(specref, measure):
    X = np.real( measure )
    Y = np.real( specref )
    SSIM = (2*np.mean(X)*np.mean(Y) + 0)*( 2*np.cov(X,Y)[0][1] + 0) /( (np.mean(X)**2 + np.mean(Y)**2 +0)*(np.std(X)**2 +np.std(Y)**2 + 0))
    return SSIM

##Get High SNR ref spectrum for SSIM:
gb = 16 #global gaussian broaden
os.chdir( 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\10' ) #1k scan EXP
fid, SW = proc.loadfid('fid',plot='no')  
cpmg, fidcoadd, dumspec = proc.coadd(fid,lb = 0, plot='no')
freq = proc.freqaxis(fidcoadd,SW)
fidcoadd = fidcoadd[1:]  #Weird glitch, need to throw out first point to balance
fidcoadd = proc.gauss(fidcoadd,gb)
ph = [371.283066651318, 44952.976, -1690.231670275709, 99.559]
spec = proc.phase(proc.fft(fidcoadd),ph)
spec = np.real(spec)/np.max(np.real(spec))
plt.plot(np.real(spec))
sys.exit()
##Move on to main loop

path = 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\'
m = 8

SSIMin =  np.zeros(m)
SSIMout =  np.zeros(m)
specrecon = np.zeros((1024,m),dtype='complex64')
specin = np.zeros((1024,m),dtype='complex64')
snrpin = np.zeros(m)
snrpout =  np.zeros(m)
for i in range(m):
    kk = 10 + i
    print(kk)
    os.chdir( path + str(kk) )
    fid, SW = proc.loadfid('fid',plot='no')  

    cpmg, fidcoadd, dumspec = proc.coadd(fid,lb = 0, plot='no') #save lb for cadzow
    fidcoadd = fidcoadd[1:]
    
    specintemp = proc.phase(proc.fft(proc.gauss(fidcoadd,gb)),ph)[0,:]
    specin[:,i] = np.real(specintemp)/np.max(np.real(specintemp))
    SSIMin[i] = ssim(spec,specin[:,i]) #SSIM of raw spec
    snrpin[i] = snrp(specin[:,i] ,312,721)

    #Denoise
    fidrecon = proc.cadzow(fidcoadd, ph, SW,lb = gb)
    
    plt.close()
    
    specrecont = proc.phase(proc.fft(fidrecon),ph)[0,:]
    specrecon[:,i] = np.real(specrecont)/np.max(np.real(specrecont))
    SSIMout[i] = ssim(spec,specrecon[:,i])
    snrpout[i] = snrp(specrecon[:,i],312,721)

plt.figure(1)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specin[:,i]) - i*np.max(np.real(specin[:,i])), label = 'SNRpp_in = %.1f , SSIM_in = %.4f' % (snrpin[i],SSIMin[i]))
plt.legend(loc='upper right')
plt.title('Noisey Spectra')
plt.xlabel('Frequency (kHz)')
plt.gca().invert_xaxis()

plt.figure(2)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specrecon[:,i]) - i*np.max(np.real(specrecon[:,i])), label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
plt.legend(loc='upper right')
plt.title('Denoised Spectra')
plt.xlabel('Frequency (kHz)')
plt.gca().invert_xaxis()

c = 0.00097758 #SSIMin of 1024 against itself has this difference from 1.0000

SSIMin = SSIMin - c
SSIMout = SSIMout - c

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