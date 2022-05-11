import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/GitHub/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from tabulate import tabulate
import nmrglue.fileio.bruker as br
import nmrglue.process.pipe_proc as pipe
start_time = time.time()

cwd =  os.getcwd()

##Script will loop over array of Na spectra
##Get High SNR ref spectrum for SSIM:

q = 1.2 ##Spacing spectra ()
xi = 10; xj = -35

gb = 0 #global gaussian broaden
c = 0
zf = 2**13 #2**16
k = 5 #8

# th = 0.2

#256 scan EXP
os.chdir( cwd+ '\\' + '108')
dic, fid = br.read(cwd + '\\' + '108')
fid = br.remove_digital_filter(dic, fid)

# fidcoadd = fidcoadd[1:]  #Weird glitch, need to throw out first point to balance

fid = proc.gauss(fid,gb,c)

ph = [-129, -220, 0, 0]
spec = proc.phase(proc.fft(fid,zf),ph)

spec = np.real(spec)/np.max(np.real(spec))
freq = proc.freqaxis(spec)
b = simproc.nearest(freq,-4)
a = simproc.nearest(freq,-2.779)

# plt.plot(freq,spec)
# sys.exit()
#SSIM bounds
o = simproc.nearest(freq,-6.8)
p = simproc.nearest(freq,3.68)

#noise bounds
o1 = simproc.nearest(freq,-9.28)
p1 = simproc.nearest(freq,-4.5)

freq = proc.freqaxis(spec,unit='ppm')

##Move on to main loop
m = 9#9

SSIMin =  np.zeros(m); SSIMout =  np.zeros(m); SSIMoutWT =  np.zeros(m)
specrecon = np.zeros((len(spec),m),dtype='complex64')
specreconWT = np.zeros((len(spec),m),dtype='complex64')
specin = np.zeros((len(spec),m),dtype='complex64')
snrpin = np.zeros(m); snrpout =  np.zeros(m) ; snrpoutWT =  np.zeros(m)
for i in range(m):
    kk = 108 - i
    print(kk)
    
    os.chdir( cwd+ '\\' + str(kk))
    dic, fid = br.read(cwd + '\\' + str(kk))
    fid = br.remove_digital_filter(dic, fid)
    
    specintemp = proc.phase(proc.fft(proc.gauss(fid,gb,c),zf),ph)
    
    # ndic, specintemp = pipe.med(dic,specintemp.real, nw=500, sf=16, sigma=5.0) #baseline correct
    # specintemp = specintemp[400:-400] #baseline
    # plt.plot(specintemp)
    # sys.exit()
    
    specin[:,i] = np.real(specintemp)/np.max(np.real(specintemp))
    SSIMin[i] = simproc.ssim(spec[o:p],specin[o:p,i]) #SSIM of raw spec
    # snrpin[i] = simproc.snrp(specin[:,i] ,a,b, th)
    snrpin[i] = (specin[a,i] - specin[b,i]) / np.std(specin[o1:p1,i])

    #Denoise
    fidrecon = proc.cadzow(fid, 1)
    fidrecon = proc.gauss(fidrecon,gb)
    
    specrecont = proc.phase(proc.fft(fidrecon,zf),ph)
    specrecon[:,i] = np.real(specrecont)/np.max(np.real(specrecont))
    SSIMout[i] = simproc.ssim(spec[o:p],specrecon[o:p,i])
    # snrpout[i] = simproc.snrp(specrecon[:,i],a,b, th)
    snrpout[i] = (specrecon[a,i] - specrecon[b,i]) / np.std(specrecon[o1:p1,i])
    
    #WT
    # wdw = wave.region_spec(np.real(specintemp), nthresh = 2, buff_size = 100, filter_size = 10)
    specrecont, coeffin, coeffs = wave.wavelet_denoise(k, np.real(specintemp), 0, wave='bior2.4', alpha=0)
    # specrecont, coeffin, coeffs = wave.wavelet_denoise(k, np.real(specintemp), wdw)
    
    specreconWT[:,i] = np.real(specrecont)/np.max(np.real(specrecont))
    SSIMoutWT[i] = simproc.ssim(spec[o:p],specreconWT[o:p,i])
    # snrpoutWT[i] = simproc.snrp(specreconWT[:,i],a,b, th)
    snrpoutWT[i] = (specreconWT[a,i] - specreconWT[b,i]) / np.std(specreconWT[o1:p1,i])

c1 = SSIMin[0] - 1 #SSIMin of 1024 against itself has this difference from 1.0000

SSIMin -= c1; SSIMout -= c1 ; SSIMoutWT -= c1

# freq = freq[400:-400]

#PLotting
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14
mpl.rcParams['pdf.fonttype'] = 42

plt.subplot(131)
for i in range(len(snrpin)):
    plt.plot(freq,np.real((specin[:,i])) - i*np.max(np.real(specin[:,i]))/q, label = 'SNRpp_in = %.1f , SSIM_in = %.4f' % (snrpin[i],SSIMin[i]))
#plt.legend(loc='upper right')
plt.title('Input')
plt.xlabel('Frequency (ppm)')
plt.xlim((xi, xj))
plt.minorticks_on()
plt.yticks([])

plt.subplot(132)
for i in range(len(snrpin)):
    plt.plot(freq,np.real((specrecon[:,i])) - i*np.max(np.real(specrecon[:,i]))/q, label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
#plt.legend(loc='upper right')
plt.title('Cadzow Denoised')
plt.xlabel('Frequency (ppm)')
plt.xlim((xi, xj))
plt.minorticks_on()
plt.yticks([])

plt.subplot(133)
for i in range(len(snrpin)):
    plt.plot(freq,np.real((specreconWT[:,i])) - i*np.max(np.real(specreconWT[:,i]))/q, label = 'SNRpp_out = %.1f , SSIM_out = %.4f' % (snrpout[i],SSIMout[i]))
#plt.legend(loc='upper right')
plt.title('WT Denoised')
plt.xlabel('Frequency (ppm)')
plt.xlim((xi, xj))
plt.minorticks_on()
plt.yticks([])


##Table SNR peak-peak
ns = [256, 128, 64, 32, 16, 8, 4, 2, 1]
data=[]
for i in range(len(snrpin)):
    data.append( ['%d'%ns[i], "%.1f"%snrpin[i], "%.1f"%snrpout[i], "%.1f"%snrpoutWT[i], "%.4f"%SSIMin[i], "%.4f"%SSIMout[i], "%.4f"%SSIMoutWT[i]] )
# create header
head = ['ns', 'SNRpp_in', 'SNRpp_out_Cz', 'SNRpp_out_WT', 'SSIM_in', 'SSIM_out', 'SSIM_out_WT']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))
#generate ns counter at end

g = np.asarray(data,dtype='float64')
np.savetxt("23Na_C_WT_comp.csv", g, fmt='%5.4f', delimiter=',')

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))