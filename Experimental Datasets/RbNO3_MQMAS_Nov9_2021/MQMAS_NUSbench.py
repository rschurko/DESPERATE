# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/GitHub/SSNMR/functions')
import numpy as np
import nmrglue as ng
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import NUS as nus
from tabulate import tabulate

start_time = time.time()

cwd =  os.getcwd()

##1. Params:
#Pick just two figures to compare
aa = 2 #Spectrum index
base = 0.5 #base contour %'age
nc = 40 #number of contours
s = 0 #switch on sum (0) or skyline (1) projection or both (2)

# Params:
nzf1 = 512
nzf2 = 4096
gb1 = 30 #GB on F1
gb2 = 0 #GB on F2

ph0 = [243 + 10 -6, 797420 + 140, 0, 0] #phases for F2
# ph0 = [-148, 796480, 0, 0] #phases for F2

a1 = 196; a2 = 340; a3 = 356 #F1 indices for the 2nd order spectra
# a3 = simproc.nearest(fiso,-26.6504)
# a2 = simproc.nearest(fiso,-27.43)
# a1 = simproc.nearest(fiso,-33.82)

##2. Obtain all exp spec over a loop
m = 3
exp = [20, 30, 31, 32]
SSIMin1 = np.zeros(m); SSIMin2 = np.zeros(m); SSIMin3 = np.zeros(m); SSIMout1 = np.zeros(m);
SSIMout2 = np.zeros(m); SSIMout3 = np.zeros(m); snrF1in = np.zeros(m); snrF2in = np.zeros(m);
snrF2out = np.zeros(m); snrF1out = np.zeros(m)
specrecon = np.zeros((nzf1,nzf2,m),dtype='complex')
specin = np.zeros((nzf1,nzf2,m),dtype='complex')

for j in range(m):
    if j == 0:
        fid = np.load('128t1_ref_FID_EXP20.npy')
        os.chdir(cwd + '\\' + str(exp[j]))
    else:
        os.chdir(cwd + '\\' + str(exp[j]))
        dct, data = ng.bruker.read(cwd+ '\\' + str(exp[j])) #load FID
        ##NUS fcn
        f1nus = nus.prep(data, nzf2, ph0)
        
        #IST recon
        spec = nus.IST_D(f1nus, nzf1, threshold = 0.99,max_iter=30) #IST_S or _D
        
        spec = np.fliplr(spec)
        
        fid = np.fft.ifft2(spec)
        fid = np.fft.fftshift(np.fft.fft(fid, axis = 1 ),axes = 1)
        fid = np.fft.ifft(fid, axis = 1)
        ##end 

    ## Shear
    spec = proc.mqproc(fid, SH = 7/9,zf1=nzf1, zf2=nzf2, lb1 = gb1, lb2 = gb2) 
    
    #Phase
    if j == 0:
        # ph = [243 + 10, 797420 + 140, 0, 0]
        ph = ph0
        specin[:,:,j] = proc.phase(spec,ph,ax=1)
    else:
        # proc.mphase(spec, fine = 100, ax=1) #determine initial phases 
        #Need a shift and flips n such
        spec = np.fft.fftshift(spec,axes=0)
        spec = np.fliplr(spec)
        spec = np.flipud(spec)
        #Phase
        # proc.mphase(spec, fine = 100, ax=1) #determine initial phases 
        ph = [180+3, 0, 0, 0]
        # ph = [0, 0, 0, 0]
        specin[:,:,j] = proc.phase(spec,ph,ax=1)
    
    #Denoise 2D SWT
    fin_reg = wave.region_spec2(np.real(spec), thresh = 22, wndw = 8)
    specrecon[:,:,j], coeffin, coeffs = wave.wavelet_denoise2(2, np.real(specin[:,:,j]), fin_reg)

    #PCA Denoise
    # specrecon[:,:,j] = proc.PCA(specrecon[:,:,j],10)
    # plt.close()
    
    #Normalize
    specin[:,:,j] = np.real(specin[:,:,j]) / np.max(np.real(specin[:,:,j]))
    specrecon[:,:,j] = np.real(specrecon[:,:,j]) / np.max(np.real(specrecon[:,:,j]))
    
    #SNRs
    a = np.unravel_index(spec.argmax(), spec.shape)
    snrF2in[j] = proc.snr(specin[a[0],:,j],1000)
    snrF1in[j] = proc.snr(specin[:,a[1],j],150)
    snrF2out[j] = proc.snr(specrecon[a[0],:,j],1000)
    snrF1out[j] = proc.snr(specrecon[:,a[1],j],150)
    
    ##SSIMs for the 2nd order patterns
    SSIMin1[j] = simproc.ssim(specin[a1,:,0],specin[a1,:,j])
    SSIMin2[j] = simproc.ssim(specin[a2,:,0],specin[a2,:,j]) 
    SSIMin3[j] = simproc.ssim(specin[a3,:,0],specin[a3,:,j])
    
    SSIMout1[j] = simproc.ssim(specin[a1,:,0],specrecon[a1,:,j])
    SSIMout2[j] = simproc.ssim(specin[a2,:,0],specrecon[a2,:,j])
    SSIMout3[j] = simproc.ssim(specin[a3,:,0],specrecon[a3,:,j])
    
    print('%.0f Percent Done' %(100*(j+1)/m))

c = simproc.ssim(specin[a1,:,0],specin[a1,:,0]) - 1
SSIMin1 = SSIMin1 - c; SSIMin2 = SSIMin2 - c; SSIMin3 = SSIMin3 - c; 
SSIMout1 = SSIMout1 - c; SSIMout2 = SSIMout2 - c; SSIMout3 = SSIMout3 - c; 

#############Plotting Stuff
#Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14
mpl.rcParams['pdf.fonttype'] = 42

freq2 = proc.freqaxis(spec[0,:],unit='ppm')
fiso = proc.fiso(spec[:,0],unit='ppm')

h = np.max(np.real(specin[:,:,0]))
lvls = np.linspace((base*1e-2)*h,h,nc)

#############First Fig
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,fiso,(np.real(specin[:,:,0])),lvls,cmap='jet')
main_ax.set_xlabel('F$_{2}$ (ppm)')#, fontfamily = 'Arial')
#main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel("F$_{iso}$ (ppm)",labelpad=-429)
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.set_xlim(-24, -40) ##CHK BACK
main_ax.set_ylim(-22, -39)  ##CHK BACK
main_ax.tick_params(right = True,left = False,labelleft = False, 
                    labelright=True, which = 'both')
main_ax.minorticks_on()

if s == 0:
    xplot.plot(freq2,(np.sum(np.real(specin[:,:,0]),0)),'k') #sum
    yplot.plot(np.real(np.sum(specin[:,:,aa],1)),fiso,'k')
elif s==1:
    xplot.plot(freq2,(np.max(np.real(specin[:,:,0]),0)),'k') #skyline
    yplot.plot(np.real(np.max(specin[:,:,aa],1)),fiso,'k') #Skyline
else:
    xplot.plot(freq2,(np.max(np.real(specin[:,:,0]),0) / np.max(np.max(np.real(specin[:,:,0]),0)) )+0.5,'k') #both
    xplot.plot(freq2,(np.sum(np.real(specin[:,:,0]),0) / np.max(np.sum(np.real(specin[:,:,0]),0))),'r') #
    yplot.plot(np.real(np.max(specin[:,:,aa],1) / np.max(np.max(specin[:,:,aa],1)))+0.5,fiso,'k') 
    yplot.plot(np.real(np.sum(specin[:,:,aa],1) / np.max(np.sum(specin[:,:,aa],1))),fiso,'r')

# xplot.plot(freq2,(np.sum(np.real(specin[:,:,aa]),0)),'k')
# yplot.plot(np.real(np.sum(specin[:,:,aa],1)),fiso,'k')
yplot.invert_xaxis()
yplot.invert_yaxis()
yplot.set_ylim(-22, -39)  ##CHK BACK

#Plot the sub-spectra
s1 = fig.add_subplot(grid[1, 4], yticklabels=[])
s2 = fig.add_subplot(grid[2, 4], yticklabels=[],sharex=s1)
s3 = fig.add_subplot(grid[3, 4], yticklabels=[],sharex=s1)
s1.invert_xaxis()
#s1.minorticks_on()
s1.set_xlim(-24,-40) ##CHK BACK
s3.set_xlabel('F$_{2}$ (ppm)')

s1.plot(freq2,(np.real(specin[a1,:,0])),'c')
s2.plot(freq2,(np.real(specin[a2,:,0])),'b')
s3.plot(freq2,(np.real(specin[a3,:,0])),'m')

yplot.axis('off')
xplot.axis('off')
####################Second fig
h = np.max(np.real(specrecon[:,:,aa]))
lvls = np.linspace((base*1e-2)*h,h,nc)

fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,fiso,(np.real(specrecon[:,:,aa])),lvls,cmap='jet')
main_ax.set_xlabel('F$_{2}$ (ppm)')#, fontfamily = 'Arial')
#main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel("F$_{iso}$ (ppm)",labelpad=-429)
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.set_xlim(-24, -40) ##CHK BACK
main_ax.set_ylim(-22, -39)  ##CHK BACK
main_ax.tick_params(right = True,left = False,labelleft = False, 
                    labelright=True, which = 'both')
main_ax.minorticks_on()

if s == 0:
    xplot.plot(freq2,(np.sum(np.real(specrecon[:,:,aa]),0)),'k') #sum
    yplot.plot(np.real(np.sum(specrecon[:,:,aa],1)),fiso,'k')
elif s==1:
    xplot.plot(freq2,(np.max(np.real(specrecon[:,:,aa]),0)),'k') #skyline
    yplot.plot(np.real(np.max(specrecon[:,:,aa],1)),fiso,'k') #Skyline
else:
    xplot.plot(freq2,(np.max(np.real(specrecon[:,:,aa]),0) / np.max(np.max(np.real(specin[:,:,aa]),0)) )+0.5,'k') #both
    xplot.plot(freq2,(np.sum(np.real(specrecon[:,:,aa]),0) / np.max(np.sum(np.real(specin[:,:,aa]),0))),'r') #
    yplot.plot(np.real(np.max(specrecon[:,:,aa],1) / np.max(np.max(specrecon[:,:,aa],1)))+0.5,fiso,'k') 
    yplot.plot(np.real(np.sum(specrecon[:,:,aa],1) / np.max(np.sum(specrecon[:,:,aa],1))),fiso,'r')
    

# xplot.plot(freq2,(np.sum(np.real(specrecon[:,:,aa]),0)),'k')
# yplot.plot(np.real(np.sum(specrecon[:,:,aa],1)),fiso,'k')

yplot.invert_xaxis()
yplot.invert_yaxis()
yplot.set_ylim(-22, -39)  ##CHK BACK

#Plot the sub-spectra
s1 = fig.add_subplot(grid[1, 4], yticklabels=[])
s2 = fig.add_subplot(grid[2, 4], yticklabels=[],sharex=s1)
s3 = fig.add_subplot(grid[3, 4], yticklabels=[],sharex=s1)
s1.invert_xaxis()
#s1.minorticks_on()
s1.set_xlim(-24,-40) ##CHK BACK
s3.set_xlabel('F$_{2}$ (ppm)')

s1.plot(freq2,(np.real(specrecon[a1,:,aa])),'c')
s2.plot(freq2,(np.real(specrecon[a2,:,aa])),'b')
s3.plot(freq2,(np.real(specrecon[a3,:,aa])),'m')

yplot.axis('off')
xplot.axis('off')
##Table of results
##N.B.: dont want SNR measure and want SSIM against no NUS
# ns = ['ns=384', 'ns=192', 'ns=96', '75 kHz', 'No SPAM']
ns = [100,60,40,20]
data=[]
for i in range(len(SSIMin1)):
    data.append( [ns[i],"%.1f"%snrF1in[i],"%.1e"%snrF1out[i], "%.1e"%snrF2in[i],
                  "%.1e"%snrF2out[i], "%.4f"%SSIMin1[i], "%.4f"%SSIMout1[i],
                  "%.4f"%SSIMin2[i], "%.4f"%SSIMout2[i], "%.4f"%SSIMin3[i],
                  "%.4f"%SSIMout3[i] ])
# create header
head = ['NUS %','SNRF1_in','SNRF1_out', 'SNRF2_in','SNRF2_out', 'SSIM_in1', 'SSIM_out1', 
        'SSIM_in2', 'SSIM_out2', 'SSIM_in3', 'SSIM_out3']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))

os.chdir(cwd)

a = np.asarray(data,dtype='float64')
np.savetxt("8_MQ_exp.csv",a, fmt='%5.4f', delimiter=',')

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))