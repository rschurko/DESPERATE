# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
start_time = time.time()


fid,SW = simproc.read('MQMAS_echo_sync.fid', lb=0, plot='no') #This normalizes FID
fid = simproc.noise(fid, 0.02) #Adds noise as a % of max intensity

#Simulation params:
np1 = 128
np2 = 1024
zf1 = 512
zf2 = 4096
dwt1 = 100e-6
swF1 = 1/dwt1

data = np.reshape(fid,(np1,np2))

lb = 50 #t1 LB
for i in range(np2):
    data[:,i] = proc.gauss(data[:,i],lb,c=0)

#FT t2
spec1 = np.zeros((np1,zf2),dtype='complex')
for i in range(np1):
    spec1[i,:] = proc.fft(data[i,:])
    
#Shearing
SH = -7/9
freq2 = np.linspace(-SW/2,SW/2,zf2) #F2 freq. (Hz)
t1 = np.arange(0,np1*dwt1,dwt1)             #t1 time vector (s)
spec1 = np.multiply(spec1, np.exp(1j*SH*2*np.pi*np.outer(t1,freq2)))

#iFT t2, centered GB, FT t2 again
gb = 17 #Amount of gaussian broadening 
for i in range(np1):
    spec1[i,:] = (np.fft.ifft(spec1[i,:]))
    # spec1[i,:] = proc.cadzow(spec1[i,:])   ##DENOISE
    spec1[i,:] = proc.gauss(spec1[i,:],gb,c=(np2/2)/zf2)
    spec1[i,:] = (np.fft.fft(spec1[i,:],zf2))

#FT t1
spec = np.zeros((zf1,zf2),dtype='complex')
for i in range(zf2):
    spec[:,i] = proc.fft(spec1[:,i])

#PCA Denoise
# spec = proc.PCA(spec,5)
# plt.close()
# plt.close()
#sys.exit()

#Phase
ph = [278, 184280, 0, 0]
#ph = proc.autophase(spec[410,:],50,phase2='no')
spec = proc.phase(spec,ph,ax=1)

#SNRs
a = np.unravel_index(spec.argmax(), spec.shape)
snrF2 = proc.snr(spec[a[0],:],1000)
snrF1 = proc.snr(spec[:,a[1]],250)

snrp1 = simproc.snrp(spec[290,:],2293,2464)
snrp2 = simproc.snrp(spec[422,:],2130,2408)
snrp3 = simproc.snrp(spec[433,:],1836,2118)


print('SNR over F2 = %5.1f' %snrF2)
print('SNR over F1 = %5.1f' %snrF1)

#Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14


SF = 196.3478212 #Larmor for ppm
#SF = 196.3420918
SW1 = 1/dwt1
off = -6.1e3
freq2 = np.linspace(-SW/2,SW/2,zf2)+off #F2 freq. (Hz)
freq1 = np.linspace(-SW1/2,SW1/2,zf1)
fiso = freq1/((SF)*(3-SH))+(off/SF)#Fiso in ppm

h = np.max(np.real(spec))
lvls = np.linspace(0.02*h,h,30)

# Set up the axes with gridspec 
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2/SF,fiso,np.fliplr(np.real(spec)),lvls,cmap='jet')
main_ax.set_xlabel('F2 Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.set_xlim(-24, -40)
main_ax.set_ylim(-25, -32)
main_ax.minorticks_on()

xplot.plot(freq2/SF,np.flipud(np.sum(np.real(spec),0)),'k')

yplot.plot(np.real(np.sum(spec,1)),fiso,'k')
yplot.invert_xaxis()
yplot.invert_yaxis()
yplot.set_ylim(-25, -32) 

#Plot the sub-spectra
s1 = fig.add_subplot(grid[1, 4], yticklabels=[])
s2 = fig.add_subplot(grid[2, 4], yticklabels=[],sharex=s1)
s3 = fig.add_subplot(grid[3, 4], yticklabels=[],sharex=s1)
s1.invert_xaxis()
#s1.minorticks_on()
s1.set_xlim(-24,-40)
s3.set_xlabel('F2 Frequency (ppm)')

s1.plot(freq2/SF,np.flipud(np.real(spec[290,:])),'c')
s2.plot(freq2/SF,np.flipud(np.real(spec[422,:])),'b')
s3.plot(freq2/SF,np.flipud(np.real(spec[433,:])),'m')

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))