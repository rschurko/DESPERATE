# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/GitHub/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import nmrglue.fileio.bruker as br
import NUS as nus
# from scipy.signal import hilbert
start_time = time.time()

cwd =  os.getcwd()
os.chdir(cwd + '\\' + '24')
# fid = proc.loadfid2('ser',plot='no')

# dct, data = br.read(os.getcwd())
# sys.exit()
# fid = br.remove_digital_filter(dct, fid)

#Params:
nzf1 = 512
nzf2 = 4096*4
gb1 = 0
gb2 = 3

s = 0 #projections switch
base = 10 #base contour %'age
nc = 20 # number contours
xi = 210; xj = -30 #xlims
yi = 20; yj = -18 #ylims

ph = [-90,0,0,0]

#NUS tings
data = proc.loadfid2('ser',plot='no')
dct, d = br.read(cwd+ '\\' + '24')
# sys.exit()
# lst = np.loadtxt('nuslist').astype('int')
# data = np.reshape(data,(len(lst)*2,int(data.shape[0]/(len(lst)*2)))) 
data = br.remove_digital_filter(dct, data)

##NUS fcn
f1nus = nus.prep(data, nzf1, nzf2, ph, gb2, st=2)    

#IST recon
spec = nus.IST_D(f1nus, nzf1, 1, max_iter=10) #IST_S or _D

spec = np.flipud(spec)

#Phase
# spec = proc.phase(spec,[90,0,0,0],ax=1)
# proc.mphase(spec,fine=100)
# sys.exit()

#2D SWT
# fin_reg = wave.region_spec2(np.real(spec),7,8)
# specrecon, coeffin, coeffs = wave.wavelet_denoise2(2, np.real(spec), fin_reg)
# spec, coeffin, coeffs = wave.wavelet_denoise2(2, np.real(spec), 0)

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
freq1 = proc.freqaxis1(spec[:,0],unit='ppm')

h = np.max(np.real(spec))
lvls = np.linspace((base*1e-2)*h,h,nc)

##########Figure 1
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.05, wspace=0.05) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:]) 

# yplot = fig.add_subplot(grid[1:, 0], yticklabels=[], sharey=main_ax)
yplot = fig.add_subplot(grid[1:, 0], sharey=main_ax)
xplot = fig.add_subplot(grid[0, 1:], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,freq1,(np.real(spec)),lvls,cmap='jet')
main_ax.set_xlabel('F$_{2}$ (ppm)')
main_ax.set_ylabel("F$_{1}$ (ppm)", labelpad=-590)
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.tick_params(right = True,left = False,labelleft = False, 
                    labelright=True, which = 'both')
main_ax.set_xlim(xi, xj) 
main_ax.set_ylim(yi, yj)
main_ax.minorticks_on()


if s == 0:
    xplot.plot(freq2,(np.sum(np.real(spec),0)),'k') #sum
    yplot.plot(np.real(np.sum(spec,1)),freq1,'k')
elif s==1:
    xplot.plot(freq2,(np.max(np.real(spec),0)),'k') #skyline
    yplot.plot(np.real(np.max(spec,1)),freq1,'k') #Skyline
else:
    xplot.plot(freq2,(np.max(np.real(spec),0) / np.max(np.max(np.real(spec),0)) )+0.5,'k') #both
    xplot.plot(freq2,(np.sum(np.real(spec),0) / np.max(np.sum(np.real(spec),0))),'r') #
    yplot.plot(np.real(np.max(spec,1) / np.max(np.max(spec,1)))+0.5,freq1,'k') 
    yplot.plot(np.real(np.sum(spec,1) / np.max(np.sum(spec,1))),freq1,'r')
    
yplot.invert_xaxis()
yplot.axis('off')
xplot.axis('off')

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