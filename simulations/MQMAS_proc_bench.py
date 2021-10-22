# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tabulate import tabulate
start_time = time.time()

#Simulation params:
np1 = 128
np2 = 1024
zf1 = 512
zf2 = 4096
dwt1 = 100e-6
swF1 = 1/dwt1

##1. Obtain noise-free spectrum
fid,SW = simproc.read('MQMAS_echo_sync.fid', lb=0, plot='no') #This normalizes FID
#fid = simproc.noise(fid, 0.1) #Adds noise as a % of max intensity
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
    spec1[i,:] = proc.gauss(spec1[i,:],gb,c=(np2/2)/zf2)
    spec1[i,:] = (np.fft.fft(spec1[i,:],zf2))

#FT t1
spec = np.zeros((zf1,zf2),dtype='complex')
for i in range(zf2):
    spec[:,i] = proc.fft(spec1[:,i])

#Phase
ph = [278, 184280, 0, 0]
#ph = proc.autophase(spec[410,:],50,phase2='no')
spec = proc.phase(spec,ph,ax=1)

##Get SSIM normalization constants
SSIM1 = simproc.ssim(spec[290,:],spec[290,:])
SSIM2 = simproc.ssim(spec[422,:],spec[422,:])
SSIM3 = simproc.ssim(spec[433,:],spec[433,:])
c1 = SSIM1 -1
c2 = SSIM2 -1
c3 = SSIM3 -1
##Noise-free done

##2. Obtain looped array of noisey spec
m = 6
noisey = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
SSIMin1 = np.zeros(m); SSIMin2 = np.zeros(m); SSIMin3 = np.zeros(m); SSIMout1 = np.zeros(m);
SSIMout2 = np.zeros(m); SSIMout3 = np.zeros(m); snrF1in = np.zeros(m); snrF2in = np.zeros(m);
snrF2out = np.zeros(m); snrF1out = np.zeros(m)
specrecon = np.zeros((zf1,zf2,m),dtype='complex')
specin = np.zeros((zf1,zf2,m),dtype='complex')
for j in range(m):
    fid,SW = simproc.read('MQMAS_echo_sync.fid', lb=0, plot='no') #This normalizes FID
    fid = simproc.noise(fid, noisey[j]) #Adds noise as a % of max intensity
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
        spec1[i,:] = proc.gauss(spec1[i,:],gb,c=(np2/2)/zf2)
        spec1[i,:] = (np.fft.fft(spec1[i,:],zf2))
    
    #FT t1
    spec2 = np.zeros((zf1,zf2),dtype='complex')
    for i in range(zf2):
        spec2[:,i] = proc.fft(spec1[:,i])
    
    #Phase
    ph = [278, 184280, 0, 0]
    #ph = proc.autophase(spec[410,:],50,phase2='no')
    spec2 = proc.phase(spec2,ph,ax=1)
    specin[:,:,j] = spec2 
    
    #PCA Denoise
    # spec = proc.PCA(spec,3)
    # plt.close()
    # plt.close()
    #sys.exit()
    
    #Denoise 2D SWT
    specrecon[:,:,j], coeffs = wave.wavelet_denoise2(2, np.real(spec2), 0, wave = 'bior2.2', threshold = 'mod', alpha = 0)

    #SNRs
    a = np.unravel_index(spec.argmax(), spec.shape)
    snrF2in[j] = proc.snr(spec2[a[0],:],1000)
    snrF1in[j] = proc.snr(spec2[:,a[1]],250)
    snrF2out[j] = proc.snr(specrecon[a[0],:,j],1000)
    snrF1out[j] = proc.snr(specrecon[:,a[1],j],250)
    
    #Simplify with SNR only
    #snrp1 = simproc.snrp(spec[290,:],2293,2464)
    #snrp2 = simproc.snrp(spec[422,:],2130,2408)
    #snrp3 = simproc.snrp(spec[433,:],1836,2118)
    
    ##SSIMs for the 2nd order patterns
    SSIMin1[j] = simproc.ssim(spec[290,:],spec2[290,:]) - c1
    SSIMin2[j] = simproc.ssim(spec[422,:],spec2[422,:]) - c2
    SSIMin3[j] = simproc.ssim(spec[433,:],spec2[433,:]) - c3
    
    SSIMout1[j] = simproc.ssim(spec[290,:],specrecon[290,:,j]) - c1
    SSIMout2[j] = simproc.ssim(spec[422,:],specrecon[422,:,j]) - c2
    SSIMout3[j] = simproc.ssim(spec[433,:],specrecon[433,:,j]) - c3
    
    print('%.0f Percent Done' %(100*(j+1)/m))


#############Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14

SF = 196.3478212 #Larmor for ppm
#SF = 196.3420918
SW1 = 1/dwt1
off = -6.1e3
freq2 = np.linspace(-SW/2,SW/2,zf2)+off #F2 freq. (Hz)
freq1 = np.linspace(-SW1/2,SW1/2,zf1)
fiso = freq1/((SF)*(3-SH))+(off/SF)#Fiso in ppm

#Pick just two figures to compare
aa = 4 #Spectrum index
#################FirstFig
#plt.figure(1) 
h = np.max(np.real(specin[:,:,aa]))
lvls = np.linspace(0.02*h,h,30) #2% base contour

# Set up the axes with gridspec 
fig = plt.figure(figsize=(11, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2/SF,fiso,np.fliplr(np.real(specin[:,:,aa])),lvls,cmap='jet')
main_ax.set_xlabel('F2 Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.set_xlim(-24, -40)
main_ax.set_ylim(-25, -32)
main_ax.minorticks_on()

xplot.plot(freq2/SF,np.flipud(np.sum(np.real(specin[:,:,aa]),0)),'k')

yplot.plot(np.real(np.sum(specin[:,:,aa],1)),fiso,'k')
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

s1.plot(freq2/SF,np.flipud(np.real(specin[290,:,aa])),'c')
s2.plot(freq2/SF,np.flipud(np.real(specin[422,:,aa])),'b')
s3.plot(freq2/SF,np.flipud(np.real(specin[433,:,aa])),'m')

####################Second fig
#plt.figure(2)
h = np.max(np.real(specrecon[:,:,aa]))
lvls = np.linspace(0.02*h,h,30) #2% base contour

# Set up the axes with gridspec 
fig = plt.figure(figsize=(11, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2/SF,fiso,np.fliplr(np.real(specrecon[:,:,aa])),lvls,cmap='jet')
main_ax.set_xlabel('F2 Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.set_xlim(-24, -40)
main_ax.set_ylim(-25, -32)
main_ax.minorticks_on()

xplot.plot(freq2/SF,np.flipud(np.sum(np.real(specrecon[:,:,aa]),0)),'k')

yplot.plot(np.real(np.sum(specrecon[:,:,aa],1)),fiso,'k')
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

s1.plot(freq2/SF,np.flipud(np.real(specrecon[290,:,aa])),'c')
s2.plot(freq2/SF,np.flipud(np.real(specrecon[422,:,aa])),'b')
s3.plot(freq2/SF,np.flipud(np.real(specrecon[433,:,aa])),'m')

##Table of results
data=[]
for i in range(len(SSIMin1)):
    data.append( ["%.1f"%snrF1in[i],"%.1f"%snrF1out[i], "%.1f"%snrF2in[i],
                  "%.1f"%snrF2out[i], "%.4f"%SSIMin1[i], "%.4f"%SSIMout1[i],
                  "%.4f"%SSIMin2[i], "%.4f"%SSIMout2[i], "%.4f"%SSIMin3[i],
                  "%.4f"%SSIMout3[i] ])
# create header
head = ['SNRF1_in','SNRF1_out', 'SNRF2_in','SNRF2_out', 'SSIM_in1', 'SSIM_out1', 
        'SSIM_in2', 'SSIM_out2', 'SSIM_in3', 'SSIM_out3']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))