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

##Script will loop over array of Pt spectra
##Also used highest signal avg as SSIM 

##Get High SNR ref spectrum for SSIM:
gb = 5 #global gaussian broaden
p = 15 # global cadzow % cut off
zf = 2**14
q = 3 #levels to use
path = 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\Pt(NH3)4Cl2_Sept28_2021\\' 
os.chdir( path ) #1k scan EXP
fidcoadd = np.load('fid2k.npy')  ##Load the pre-processed 2k scan FID
fidcoadd = proc.gauss(fidcoadd,gb)
#ph = [326, 680344, -9343, 455]
ph = [326, 680344, -9343, 910]
spec = proc.phase(proc.fft(fidcoadd,zf),ph)
spec = np.real(spec)/np.max(np.real(spec))

specout, coeffin, coeffs = wave.wavelet_denoise(q, np.real(spec), 0, wave = 'bior2.2', threshold = 'mod', alpha = 0)
specout = np.real(specout)/np.max(np.real(specout))
##Move on to main loop
m = 6

os.chdir( path + '50' )
freq = proc.freqaxis(spec,zf)
os.chdir( path )
#sys.exit()
l= simproc.nearest(freq,232.657) ##Allows me to change ZF and retain peak indices
k = simproc.nearest(freq,-617.789)

SSIMin =  np.zeros(m)
SSIMout =  np.zeros(m)
specrecon = np.zeros((len(spec),m),dtype='complex64')
specin = np.zeros((len(spec),m),dtype='complex64')
snrpin = np.zeros(m)
snrpout =  np.zeros(m)

#Entries for 2K scan:
SSIMin[0] = simproc.ssim(spec,spec)
snrpin[0] = simproc.snrp(spec,k,l)
specin[:,0] = spec
SSIMout[0] = simproc.ssim(spec,specout)
snrpout[0] = simproc.snrp(specout,k,l)
specrecon[:,0] = specout

for i in range(m-1):
    if i == 4:
        i = 5
    kk = 50 + i
    print(kk)
    os.chdir( path + str(kk) )
    fid, SW = proc.loadfid('fid',plot='no') 
    if i == 5:
        i = 4
    cpmg, fidcoadd = proc.coadd(fid,MAS='yes') #save lb for cadzow
    
    specintemp = proc.phase(proc.fft(proc.gauss(fidcoadd,gb),zf),ph)
    specin[:,i+1] = np.real(specintemp)/np.max(np.real(specintemp))
    SSIMin[i+1] = simproc.ssim(spec,specin[:,i+1]) #SSIM of raw spec
    snrpin[i+1] = simproc.snrp(specin[:,i+1] ,k,l)

    #Denoise
    specrecont, coeffin, coeffs = wave.wavelet_denoise(q, np.real(specin[:,i+1]), 0, wave = 'bior2.2', threshold = 'mod', alpha = 0)

    specrecon[:,i+1] = np.real(specrecont)/np.max(np.real(specrecont))
    SSIMout[i+1] = simproc.ssim(spec,specrecon[:,i+1])
    snrpout[i+1] = simproc.snrp(specrecon[:,i+1],k,l)

##Plotting
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14

c = 1 - SSIMin[0]#0.000061038881768 #SSIMin of 2048 against itself has this difference from 1.0000
SSIMin = SSIMin + c
SSIMout = SSIMout + c

#plt.figure(1)
plt.subplot(121)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specin[:,i]) - i*np.max(np.real(specin[:,i])), label = 'SNRpp_in = %.1f , SSIM_in = %.4f' % (snrpin[i],SSIMin[i]))
plt.legend(loc='upper right')
plt.title('Noisey Spectra')
plt.xlabel('Frequency (kHz)')
plt.xlim((-900,500))
plt.gca().invert_xaxis()
plt.yticks([])
plt.minorticks_on()

#plt.figure(2)
plt.subplot(122)
for i in range(len(snrpin)):
    plt.plot(freq,np.real(specrecon[:,i]) - i*np.max(np.real(specrecon[:,i])), label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
plt.legend(loc='upper right')
plt.title('Denoised Spectra')
plt.xlabel('Frequency (kHz)')
plt.xlim((-900,500))
plt.gca().invert_xaxis()
plt.yticks([])
plt.minorticks_on()

##Table SNR peak-peak
ns = [2048, 1024, 512, 256, 128, 64]
data=[]
for i in range(len(snrpin)):
    data.append( ["%d"%ns[i],  "%.1f"%snrpin[i], "%.4f"%SSIMin[i], "%.1f"%snrpout[i], "%.4f"%SSIMout[i]] )
# create header
head = ['ns', 'SNRpp_in', 'SSIM_in', 'SNRpp_out', 'SSIM_out']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))
#generate ns counter at end
os.chdir( path )
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))