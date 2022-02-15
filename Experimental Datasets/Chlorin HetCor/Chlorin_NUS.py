# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
start_time = time.time()

cwd =  os.getcwd()
os.chdir(cwd + '\\' + '22')
fid = proc.loadfid2('ser',plot='no')

#Hypercomplex (STATES-TPPI)
re = fid[0::2,:]
im = fid[1::2,:]  #fid[-1,:]

fid2 = re - 1j*im

#Params:
nzf1 = 512
nzf2 = 4096*2
gb1 = 5
gb2 = 5

s = 2 #projections switch
base = 2 #base contour %'age
xi = 500; xj = -50 #xlims
yi = 42; yj = -5 #ylims

fid2 = proc.gauss(fid2, gb1, c=0, ax=0)
fid2 = proc.gauss(fid2, gb2, c=0, ax=1)

spec = np.fft.fftshift(np.fft.fft2(fid2, s = (nzf1, nzf2)),axes = 1)

# plt.contour(np.real(spec),40)
# sys.exit()

##Need to flip F2 for some reason and shift F1
#spec = np.roll(spec,100,axis=0)
# spec = np.flip(spec,0)
#plt.contour(abs(spec))

#Phase
# ph=[178 - 39 -13 -23 , 819553 - 8940, 0, 0]
ph = [-77, 24520, 0, 0]
# ph = proc.autophase(spec[120,:],50,phase2='no')
# sys.exit()
spec = proc.phase(spec,ph,ax=1)
# proc.mphase(spec,fine=100)
# plt.contour(np.real(spec),40)
# sys.exit()

#2D SWT
# fin_reg = wave.region_spec2(np.real(spec),17,8)
# plt.contour(fin_reg*np.real(spec.max()),40,cmap='binary')
# plt.contour(np.real(spec),40,cmap='jet')
# sys.exit()
# specrecon, coeffin, coeffs = wave.wavelet_denoise2(2, np.real(spec), fin_reg)
# specrecon, coeffin, coeffs = wave.wavelet_denoise2(2, np.real(spec), 0)

#PCA Denoise
# specrecon = proc.PCA(specrecon,10)
# plt.close()

#SNRs
a = np.unravel_index(spec.argmax(), spec.shape)
snrF2 = proc.snr(spec[a[0],:],1500)
# snrF2r = proc.snr(specrecon[a[0],:],1500)
snrF1 = proc.snr(spec[:,a[1]],100)
# snrF1r = proc.snr(specrecon[:,a[1]],100)

print('SNR over F2 = %5.1f' %snrF2)
# print('SNR over F2 recon= %5.1f' %snrF2r)
print('SNR over F1 = %5.1f' %snrF1)
# print('SNR over F1 recon= %5.1f' %snrF1r)

######Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14
mpl.rcParams['pdf.fonttype'] = 42

freq2 = proc.freqaxis(spec[0,:],unit='ppm')
# fiso = proc.fiso(spec[:,0], SH = SH, q=3, unit='kHz',s=nzf1/np.shape(fid)[0]) 
fiso = proc.fiso(spec[:,0], SH = 7/9, q=3, unit='kHz',s=1) 

#2nd order pattern indices
# a1 = simproc.nearest(fiso,-9.56); a2 = simproc.nearest(fiso,-6.8); a3 = simproc.nearest(fiso,-4.36)
# a4 = simproc.nearest(fiso,-2.25)

h = np.max(np.real(spec))
lvls = np.linspace((base*1e-2)*h,h,30)

##########Figure 1
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,fiso,(np.real(spec)),lvls,cmap='jet')
main_ax.set_xlabel('F$_{2}$ (ppm)')#, fontfamily = 'Arial')
#main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel("F$_{iso}$ (ppm)",labelpad=-429)
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.tick_params(right = True,left = False,labelleft = False, 
                    labelright=True, which = 'both')
main_ax.set_xlim(xi, xj) 
main_ax.set_ylim(yi, yj)  ##CHK BACK
yplot.set_ylim(yj, yi)
main_ax.minorticks_on()


if s == 0:
    xplot.plot(freq2,(np.sum(np.real(spec),0)),'k') #sum
    yplot.plot(np.real(np.sum(spec,1)),fiso,'k')
elif s==1:
    xplot.plot(freq2,(np.max(np.real(spec),0)),'k') #skyline
    yplot.plot(np.real(np.max(spec,1)),fiso,'k') #Skyline
else:
    xplot.plot(freq2,(np.max(np.real(spec),0) / np.max(np.max(np.real(spec),0)) )+0.5,'k') #both
    xplot.plot(freq2,(np.sum(np.real(spec),0) / np.max(np.sum(np.real(spec),0))),'r') #
    yplot.plot(np.real(np.max(spec,1) / np.max(np.max(spec,1)))+0.5,fiso,'k') 
    yplot.plot(np.real(np.sum(spec,1) / np.max(np.sum(spec,1))),fiso,'r')
    
yplot.invert_xaxis()
yplot.invert_yaxis()

#Plot the sub-spectra
# s1 = fig.add_subplot(grid[0, 4], yticklabels=[])
# s2 = fig.add_subplot(grid[1, 4], yticklabels=[],sharex=s1)
# s3 = fig.add_subplot(grid[2, 4], yticklabels=[],sharex=s1)
# s4 = fig.add_subplot(grid[3, 4], yticklabels=[],sharex=s1)
# s1.invert_xaxis()
# #s1.minorticks_on()
# s1.set_xlim(xi, xj)  #Change
# s4.set_xlabel('F$_{2}$ (ppm)')

# s1.plot(freq2,(np.real(spec[a1,:])),'c')
# s2.plot(freq2,(np.real(spec[a2,:])),'b')
# s3.plot(freq2,(np.real(spec[a3,:])),'m')
# s4.plot(freq2,(np.real(spec[a4,:])),'r')

##########Figure 2
# spec = specrecon
# fig = plt.figure(figsize=(12, 8)) # figure size w x h
# grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
# main_ax = fig.add_subplot(grid[1:, 1:4]) 

# yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
# xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

# main_ax.contour(freq2,fiso,(np.real(spec)),lvls,cmap='jet')
# main_ax.set_xlabel('F$_{2}$ (ppm)')#, fontfamily = 'Arial')
# #main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
# main_ax.set_ylabel("F$_{iso}$ (ppm)",labelpad=-429)
# main_ax.invert_yaxis()
# main_ax.invert_xaxis()
# main_ax.tick_params(right = True,left = False,labelleft = False, 
#                     labelright=True, which = 'both')
# main_ax.set_xlim(xi, xj) 
# main_ax.set_ylim(yi, yj)  ##CHK BACK
# yplot.set_ylim(yj, yi)
# main_ax.minorticks_on()


# if s == 0:
#     xplot.plot(freq2,(np.sum(np.real(spec),0)),'k') #sum
#     yplot.plot(np.real(np.sum(spec,1)),fiso,'k')
# elif s==1:
#     xplot.plot(freq2,(np.max(np.real(spec),0)),'k') #skyline
#     yplot.plot(np.real(np.max(spec,1)),fiso,'k') #Skyline
# else:
#     xplot.plot(freq2,(np.max(np.real(spec),0) / np.max(np.max(np.real(spec),0)) )+0.5,'k') #both
#     xplot.plot(freq2,(np.sum(np.real(spec),0) / np.max(np.sum(np.real(spec),0))),'r') #
#     yplot.plot(np.real(np.max(spec,1) / np.max(np.max(spec,1)))+0.5,fiso,'k') 
#     yplot.plot(np.real(np.sum(spec,1) / np.max(np.sum(spec,1))),fiso,'r')
    
# yplot.invert_xaxis()
# yplot.invert_yaxis()

#Plot the sub-spectra
# s1 = fig.add_subplot(grid[0, 4], yticklabels=[])
# s2 = fig.add_subplot(grid[1, 4], yticklabels=[],sharex=s1)
# s3 = fig.add_subplot(grid[2, 4], yticklabels=[],sharex=s1)
# s4 = fig.add_subplot(grid[3, 4], yticklabels=[],sharex=s1)
# s1.invert_xaxis()
# #s1.minorticks_on()
# #s1.set_xlim(xi, xj) #change
# s4.set_xlabel('F$_{2}$ (ppm)')

# s1.plot(freq2,(np.real(spec[a1,:])),'c')
# s2.plot(freq2,(np.real(spec[a2,:])),'b')
# s3.plot(freq2,(np.real(spec[a3,:])),'m')
# s4.plot(freq2,(np.real(spec[a4,:])),'r')

os.chdir(cwd)
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))