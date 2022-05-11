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
import nmrglue as ng
import nmrglue.fileio.bruker as br
import NUS as nus
start_time = time.time()

EXP = '31'

cwd =  os.getcwd()
os.chdir(cwd + '\\' + EXP)

# Params:
base = 2 #base contour %'age
s=0
nzf1 = 512
nzf2 = 4096
gb1 = 30 #GB on F1
gb2 = 0 #GB on F2

ph = [243 + 10 -6, 797420 + 140, 0, 0] #phases for F2
# ph = [194+21, 788450+9140, 0, 0] #phases for F2

xi = -24; xj = -40 #xlims
yi = -22; yj = -38 #ylims

dct, data = br.read(cwd+ '\\' + EXP) #load FID

##NUS fcn
f1nus = nus.prep(data, nzf2, ph)  ##NOTE, not nzf1 but np1

##Shear first to find phases? Comment out 
# f = np.fft.ifft(f1nus,axis=1)
# g = proc.mqproc(f,7/9,0,data.shape[0],nzf2)
# g = np.fft.fftshift(g,axes=1)
# # proc.autophase(g[0,:],25,phase2='no')
# g = proc.phase(g,[194+21, 788450+9140, 0, 0],ax=1)
# proc.mphase(g,fine= 100)
# sys.exit()

#IST recon
spec = nus.IST_D(f1nus, nzf1, threshold = 0.99,max_iter=20) #IST_S or _D

spec = np.fliplr(spec)

fid = np.fft.ifft2(spec)
fid = np.fft.fftshift(np.fft.fft(fid, axis = 1 ),axes = 1)
fid = np.fft.ifft(fid, axis = 1)
# sys.exit()
##end 

##SH = +7/9 for this coherence selection
spec = proc.mqproc(fid, SH = 7/9, zf1=nzf1, zf2=nzf2, lb1=gb1, lb2=gb2) 

#Need a shift and flips n such
spec = np.fft.fftshift(spec,axes=0)
spec = np.fliplr(spec)
spec = np.flipud(spec)

#Phase
ph = [180+3, 0, 0, 0]
# ph = proc.autophase(spec[196,:],50,phase2='no')
spec = proc.phase(spec,ph,ax=1)
# proc.mphase(spec,ax=1)

#2D SWT
# spec, coeffin, coeffs = wave.wavelet_denoise2(2, np.real(spec), 0, wave = 'bior2.2', threshold = 'mod', alpha = 0)

#SNRs
a = np.unravel_index(spec.argmax(), spec.shape)
snrF2 = proc.snr(spec[a[0],:],1000)
snrF1 = proc.snr(spec[:,a[1]],150)

#snrp1 = simproc.snrp(spec[290,:],2293,2464)
#snrp2 = simproc.snrp(spec[422,:],2130,2408)
#snrp3 = simproc.snrp(spec[433,:],1836,2118)

#print('SNRp 1 = %.1f' %snrp1)
#print('SNRp 2 = %.1f' %snrp2)
#print('SNRp 3 = %.1f' %snrp3)
print('SNR over F2 = %5.1f' %snrF2)
print('SNR over F1 = %5.1f' %snrF1)

#####3Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14
mpl.rcParams['pdf.fonttype'] = 42

freq2 = proc.freqaxis(spec[0,:],unit='ppm')
freq1 = proc.fiso(spec[:,0],unit='ppm')

h = np.max(np.real(spec))
lvls = np.linspace((base*1e-2)*h,h,30)

# Set up the axes with gridspec 
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], sharey=main_ax)
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,freq1,(np.real(spec)),lvls,cmap='jet')
main_ax.set_xlabel('F$_{2}$ (ppm)')
main_ax.set_ylabel("F$_{1}$ (ppm)", labelpad=-429)
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

#Plot the sub-spectra
s1 = fig.add_subplot(grid[1, 4], yticklabels=[])
s2 = fig.add_subplot(grid[2, 4], yticklabels=[],sharex=s1)
s3 = fig.add_subplot(grid[3, 4], yticklabels=[],sharex=s1)
s1.invert_xaxis()
#s1.minorticks_on()
s1.set_xlim(xi,xj) ##CHK BACK
s3.set_xlabel('F$_{2}$ (ppm)')

s1.plot(freq2,(np.real(spec[196,:])),'c')
s2.plot(freq2,(np.real(spec[338,:])),'b')
s3.plot(freq2,(np.real(spec[356,:])),'m')

# yplot.invert_xaxis()
# yplot.axis('off')
# xplot.axis('off')
yplot.invert_xaxis()
# s1.axis('off')
yplot.axis('off')
xplot.axis('off')

os.chdir(cwd)
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))