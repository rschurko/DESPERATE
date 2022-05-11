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
from tabulate import tabulate
start_time = time.time()

cwd =  os.getcwd()

##1. Params:
#Pick just two figures to compare
aa = 2 #Spectrum index
base = 0.5 #base contour %'age
nc = 40 #number of contours
s = 2 #switch on sum (0) or skyline (1) projection or both (2)

nzf1 = 512
nzf2 = 4096
gb1 = 0
gb2 = 5

a1 = 198; a2 = 340; a3 = 358 #F1 indices for the 2nd order spectra

##2. Obtain all exp spec over a loop
m = 5
exp = [202, 201, 200, 203, 205]
SSIMin1 = np.zeros(m); SSIMin2 = np.zeros(m); SSIMin3 = np.zeros(m); SSIMout1 = np.zeros(m);
SSIMout2 = np.zeros(m); SSIMout3 = np.zeros(m); snrF1in = np.zeros(m); snrF2in = np.zeros(m);
snrF2out = np.zeros(m); snrF1out = np.zeros(m)
specrecon = np.zeros((nzf1,nzf2,m),dtype='complex')
specin = np.zeros((nzf1,nzf2,m),dtype='complex')
for j in range(m):
    
    os.chdir(cwd + '\\' + str(exp[j]))
    fid = proc.loadfid2('ser',plot='no')
    ## SH = +7/9 for this coherence selection
    spec = proc.mqproc(fid, SH = 7/9, zf1=nzf1, zf2=nzf2, lb1 = gb1, lb2 = gb2) 
    
    #PCA Denoise
    # spec = proc.PCA(spec,10)
    # plt.close()
    # plt.close()
    #sys.exit()
    
    #Phase
    ph = [243 + 10, 797420 + 140, 0, 0]
    #ph = proc.autophase(spec[196,:],50,phase2='no')
    specin[:,:,j] = proc.phase(spec,ph,ax=1)
    
    #Denoise 2D SWT
    wdw = wave.region_spec2(np.real(spec), thresh = 12, wndw = 12)
    specrecon[:,:,j], coeffin, coeffs = wave.wavelet_denoise2(2, np.real(specin[:,:,j]), wdw)

    #PCA Denoise
    # specrecon[:,:,j] = proc.PCA(specrecon[:,:,j],15)
    # plt.close()
    # plt.close()

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

h = np.max(np.real(specin[:,:,aa]))
lvls = np.linspace((base*1e-2)*h,h,nc)

#############First Fig
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,fiso,(np.real(specin[:,:,aa])),lvls,cmap='jet')
main_ax.set_xlabel('{F}$_{2}$ (ppm)')#, fontfamily = 'Arial')
#main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel("F$_{1}$ (ppm)",labelpad=-429)
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.set_xlim(-24, -40) ##CHK BACK
main_ax.set_ylim(-24, -40)  ##CHK BACK
main_ax.tick_params(right = True,left = False,labelleft = False, 
                    labelright=True, which = 'both')
main_ax.minorticks_on()

if s == 0:
    xplot.plot(freq2,(np.sum(np.real(specin[:,:,aa]),0)),'k') #sum
    yplot.plot(np.real(np.sum(specin[:,:,aa],1)),fiso,'k')
elif s==1:
    xplot.plot(freq2,(np.max(np.real(specin[:,:,aa]),0)),'k') #skyline
    yplot.plot(np.real(np.max(specin[:,:,aa],1)),fiso,'k') #Skyline
else:
    xplot.plot(freq2,(np.max(np.real(specin[:,:,aa]),0) / np.max(np.max(np.real(specin[:,:,aa]),0)) )+0.5,'k') #both
    xplot.plot(freq2,(np.sum(np.real(specin[:,:,aa]),0) / np.max(np.sum(np.real(specin[:,:,aa]),0))),'r') #
    yplot.plot(np.real(np.max(specin[:,:,aa],1) / np.max(np.max(specin[:,:,aa],1)))+0.5,fiso,'k') 
    yplot.plot(np.real(np.sum(specin[:,:,aa],1) / np.max(np.sum(specin[:,:,aa],1))),fiso,'r')

# xplot.plot(freq2,(np.sum(np.real(specin[:,:,aa]),0)),'k')
# yplot.plot(np.real(np.sum(specin[:,:,aa],1)),fiso,'k')
yplot.invert_xaxis()
yplot.invert_yaxis()
yplot.set_ylim(-24, -40)  ##CHK BACK

#Plot the sub-spectra
s1 = fig.add_subplot(grid[1, 4], yticklabels=[])
s2 = fig.add_subplot(grid[2, 4], yticklabels=[],sharex=s1)
s3 = fig.add_subplot(grid[3, 4], yticklabels=[],sharex=s1)
s1.invert_xaxis()
#s1.minorticks_on()
s1.set_xlim(-24,-40) ##CHK BACK
s3.set_xlabel('F$_{2}$ (ppm)')

s1.plot(freq2,(np.real(specin[a1,:,aa])),'c')
s2.plot(freq2,(np.real(specin[a2,:,aa])),'b')
s3.plot(freq2,(np.real(specin[a3,:,aa])),'m')

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
main_ax.set_ylim(-24, -40)  ##CHK BACK
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
yplot.set_ylim(-24, -40)  ##CHK BACK

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

##Table of results
ns = ['ns=384', 'ns=192', 'ns=96', '75 kHz', 'No SPAM']
ns = [1,2,3,4,5]
data=[]
for i in range(len(SSIMin1)):
    data.append( [ns[i],"%.1f"%snrF1in[i],"%.1f"%snrF1out[i], "%.1f"%snrF2in[i],
                  "%.1f"%snrF2out[i], "%.4f"%SSIMin1[i], "%.4f"%SSIMout1[i],
                  "%.4f"%SSIMin2[i], "%.4f"%SSIMout2[i], "%.4f"%SSIMin3[i],
                  "%.4f"%SSIMout3[i] ])
# create header
head = ['Exp','SNRF1_in','SNRF1_out', 'SNRF2_in','SNRF2_out', 'SSIM_in1', 'SSIM_out1', 
        'SSIM_in2', 'SSIM_out2', 'SSIM_in3', 'SSIM_out3']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.4f"))

os.chdir(cwd)

a = np.asarray(data,dtype='float64')
np.savetxt("8_MQ_exp.csv",a, fmt='%5.4f', delimiter=',')

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))