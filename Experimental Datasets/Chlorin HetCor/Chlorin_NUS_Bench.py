# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/GitHub/SSNMR/functions')
import numpy as np
import functions as proc
import wavelet_denoise as wave
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import nmrglue.fileio.bruker as br
import NUS as nus
# from scipy.signal import hilbert
start_time = time.time()

cwd =  os.getcwd()

aa = 2 #recon index to look at

##1. Params:
nzf1 = 512
nzf2 = 4096*4
gb1 = 0
gb2 = 2.5

s = 0 #projections switch
base = 10 #base contour %'age
nc = 20 # number contours
xi = 210; xj = -30 #xlims
yi = 20; yj = -18 #ylims

ph = [-90,0,0,0]


##2. Obtain all exp spec over a loop
m = 3
exp = [22, 23, 24, 25, 26]
SSIMin1 = np.zeros(m); SSIMin2 = np.zeros(m); SSIMout1 = np.zeros(m);
SSIMout2 = np.zeros(m);  snrF1in = np.zeros(m); snrF2in = np.zeros(m);
snrF2out = np.zeros(m); snrF1out = np.zeros(m)
specrecon = np.zeros((nzf1,nzf2,m),dtype='complex')
specin = np.zeros((nzf1,nzf2,m),dtype='complex')

for j in range(m):
    print(j)
    if j < 2:
        os.chdir(cwd + '\\' + str(exp[j]))
        data = proc.loadfid2('ser',plot='no')
        dct, d = br.read(cwd + '\\' + str(exp[j]))
        fid = br.remove_digital_filter(dct, data)
        #FT t2 Dim
        fidF2 = proc.gauss(fid, gb2, c=0, ax=1)
        fidF2 = np.fft.fftshift(np.fft.fft(fidF2,nzf2,axis=1),axes=1)
        # fidF2 = proc.fft(fid,nzf2,ax=1)
        fidF2 = proc.phase(fidF2,ph,ax=1)
        
        #Hypercomplex (STATES-TPPI)
        fidF1 = fidF2[::2].real + 1j*fidF2[1::2].real
        fidF1[1::2] = -fidF1[1::2]
        
        #FT t1 dim
        fidF1 = proc.gauss(fidF1, gb1, c=0, ax=0)
        specin[:,:,j] = np.fft.fftshift(np.fft.fft(fidF1,nzf1,axis=0),axes=0)
        
        #2D SWT
        fin_reg = wave.region_spec2(np.real(specin[:,:,j]),7,8)
        specrecon[:,:,j], coeffin, coeffs = wave.wavelet_denoise2(2, np.real(specin[:,:,j]), fin_reg)
        
    else:
        os.chdir(cwd + '\\' + str(exp[j]))
        data = proc.loadfid2('ser',plot='no')
        dct, d = br.read(cwd + '\\' + str(exp[j]))
        data = br.remove_digital_filter(dct, data)
        ##NUS fcn
        f1nus = nus.prep(data, nzf1, nzf2, ph, gb2, st=2)    
        
        #IST recon
        spec = nus.IST_D(f1nus, nzf1, 1, max_iter=20) #IST_S or _D
        # spec = nus.IST_S(f1nus, nzf1, 1, max_iter=40) #IST_S or _D
        
        specin[:,:,j] = np.flipud(spec)

        #2D SWT
        specrecon[:,:,j] = specin[:,:,j]
        # fin_reg = wave.region_spec2(np.real(specin[:,:,j]),7,8)
        # specrecon[:,:,j], coeffin, coeffs = wave.wavelet_denoise2(2, np.real(specin[:,:,j]), fin_reg)
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

h = np.max(np.real(specin[:,:,0]))
lvls = np.linspace((base*1e-2)*h,h,nc)

##########Figure 1
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.05, wspace=0.05) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:]) 

# yplot = fig.add_subplot(grid[1:, 0], yticklabels=[], sharey=main_ax)
yplot = fig.add_subplot(grid[1:, 0], sharey=main_ax)
xplot = fig.add_subplot(grid[0, 1:], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,freq1,(np.real(specin[:,:,0])),lvls,cmap='jet')
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
    xplot.plot(freq2,(np.sum(np.real(specin[:,:,0]),0)),'k') #sum
    yplot.plot(np.real(np.sum(specin[:,:,0],1)),freq1,'k')
elif s==1:
    xplot.plot(freq2,(np.max(np.real(specin[:,:,0]),0)),'k') #skyline
    yplot.plot(np.real(np.max(specin[:,:,0],1)),freq1,'k') #Skyline
else:
    xplot.plot(freq2,(np.max(np.real(specin[:,:,0]),0) / np.max(np.max(np.real(specin[:,:,0]),0)) )+0.5,'k') #both
    xplot.plot(freq2,(np.sum(np.real(specin[:,:,0]),0) / np.max(np.sum(np.real(specin[:,:,0]),0))),'r') #
    yplot.plot(np.real(np.max(specin[:,:,0],1) / np.max(np.max(specin[:,:,0],1)))+0.5,freq1,'k') 
    yplot.plot(np.real(np.sum(specin[:,:,0],1) / np.max(np.sum(specin[:,:,0],1))),freq1,'r')
    
yplot.invert_xaxis()
yplot.axis('off')
xplot.axis('off')



##########Figure 2
h = np.max(np.real(specrecon[:,:,aa]))
lvls = np.linspace((base*1e-2)*h,h,nc)

fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.05, wspace=0.05) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:]) 

# yplot = fig.add_subplot(grid[1:, 0], yticklabels=[], sharey=main_ax)
yplot = fig.add_subplot(grid[1:, 0], sharey=main_ax)
xplot = fig.add_subplot(grid[0, 1:], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,freq1,(np.real(specrecon[:,:,aa])),lvls,cmap='jet')
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
    xplot.plot(freq2,(np.sum(np.real(specrecon[:,:,aa]),0)),'k') #sum
    yplot.plot(np.real(np.sum(specrecon[:,:,aa],1)),freq1,'k')
elif s==1:
    xplot.plot(freq2,(np.max(np.real(specrecon[:,:,aa]),0)),'k') #skyline
    yplot.plot(np.real(np.max(specrecon[:,:,aa],1)),freq1,'k') #Skyline
else:
    xplot.plot(freq2,(np.max(np.real(specrecon[:,:,aa]),0) / np.max(np.max(np.real(specrecon[:,:,aa]),0)) )+0.5,'k') #both
    xplot.plot(freq2,(np.sum(np.real(specrecon[:,:,aa]),0) / np.max(np.sum(np.real(specrecon[:,:,aa]),0))),'r') #
    yplot.plot(np.real(np.max(specrecon[:,:,aa],1) / np.max(np.max(specrecon[:,:,aa],1)))+0.5,freq1,'k') 
    yplot.plot(np.real(np.sum(specrecon[:,:,aa],1) / np.max(np.sum(specrecon[:,:,aa],1))),freq1,'r')
    
yplot.invert_xaxis()
yplot.axis('off')
xplot.axis('off')

os.chdir(cwd)
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))