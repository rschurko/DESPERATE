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
import NUS as nus
start_time = time.time()

cwd =  os.getcwd()
os.chdir(cwd + '\\16')

##Recover FID from NUS TS spectrum##
# os.chdir(os.getcwd() + '\\pdata\\1')
# dic, data = ng.bruker.read_pdata(dir=os.getcwd(), bin_files = ['2rr', '2ii'], 
#                                  all_components=False)
# os.chdir(cwd + '\\16')
# fid = data[0] +1j*data[1]
# fid = np.fft.ifft2(fid)
# fid = fid[0:256,0:512] ##Not totally needed, but can remove 0-fill
# fid = np.fft.fftshift(np.fft.fft(fid, axis = 1 ),axes = 1)
# fid = np.fft.ifft(fid, axis = 1)
##End FID recovery

#Params:
nzf1 = 512
nzf2 = 4096
base = 2 #base contour %'age
# ph0 = [0, 0, 0, 0]
# ph0 = [696+168, 94750+2820, 0, 0]
# ph0 = [359 +32.5, 700794 +240, 0, 0]
# ph0 = [359 +32.5+94, 700794 +240-96960+420, 0, 0]
ph0 = [-43, 798080-240, 0, 0]

# os.chdir(cwd + '\\' + str(exp[j]))
dct, data = ng.bruker.read(cwd+ '\\' + '16') #load FID
##NUS fcn
f1nus = nus.prep(data, nzf2, ph0)

##Shear first to find phases? Comment out 
# f = np.fft.ifft(f1nus,axis=1)
# g = proc.mqproc(f,7/9,0,data.shape[0],nzf2)

# g = np.fft.ifft(g,axis=0)
# g = np.roll(g,1,axis=0)   #shift 1 point in t1 to correct phase since t1 = 0 is not acq'd
# g = np.fft.fft(g,axis=0)
# g = np.fft.fftshift(g,axes=1)
# # proc.autophase(g[0,:],25,phase2='no')
# g = proc.phase(g,[192, 687551, 0, 0],ax=1)
# proc.mphase(g[0,:],fine= 10)
# sys.exit()

#IST recon
spec = nus.IST_D(f1nus, nzf1, threshold = 0.99,max_iter=10) #IST_S or _D

spec = np.fliplr(spec)

fid = np.fft.ifft2(spec)
fid = np.fft.fftshift(np.fft.fft(fid, axis = 1 ),axes = 1)
fid = np.fft.ifft(fid, axis = 1)
##end 

##Shearing
spec = proc.mqproc(fid, SH = -7/9, zf1=nzf1, zf2=nzf2, lb1=0, lb2=0) 

fid2 = np.fft.ifft(spec,axis=0)
fid2 = np.roll(fid2,1,axis=0)   #shift 1 point in t1 to correct phase since t1 = 0 is not acq'd
spec = np.fft.fft(fid2,axis=0)

spec = np.flip(np.fft.fftshift(spec,axes = 0)) #Need for NUS*

spec = np.flip(spec,axis=0)

#PCA Denoise
# spec = proc.PCA(spec,3)
# plt.close()
# plt.close()
#sys.exit()

#Phase
#ph = [0,0, 0, 0]
# ph = [359 +32.5, 700794 +240, 0, 0]
#ph = [44, 700524, 0, 0]
#ph = [341, 701583, 0, 0]
# ph = [100-28, 679480 -2340, 0, 0]
#ph = proc.autophase(spec[169,:],50,phase2='no')
# spec = proc.phase(spec,ph,ax=1)
# proc.mphase(spec,fine=100)
# sys.exit()

#2D SWT
#spec, coeffin, coeffs = wave.wavelet_denoise2(2, np.real(spec), 0)

#SNRs
a = np.unravel_index(spec.argmax(), spec.shape)
snrF2 = proc.snr(spec[a[0],:],1000)
snrF1 = proc.snr(spec[:,a[1]],100)

print('SNR over F2 = %5.1f' %snrF2)
print('SNR over F1 = %5.1f' %snrF1)

#Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14
mpl.rcParams['pdf.fonttype'] = 42

freq2 = proc.freqaxis(spec[0,:],unit='ppm')
fiso = proc.fiso(spec[:,0],SH = 7/9,unit='ppm')

h = np.max(np.real(spec))
lvls = np.linspace((base*1e-2)*h,h,30)

# Set up the axes with gridspec 
fig = plt.figure(figsize=(12, 8)) # figure size w x h
grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
main_ax = fig.add_subplot(grid[1:, 1:4]) 

yplot = fig.add_subplot(grid[1:, 0], yticklabels=[])
xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)

main_ax.contour(freq2,fiso,(np.real(spec)),lvls,cmap='jet')
main_ax.set_xlabel('F$_{2}$ (ppm)')#, fontfamily = 'Arial')
#main_ax.set_ylabel('F1_iso Frequency (ppm)')#, fontfamily = 'Arial')
main_ax.set_ylabel("F$_{iso}$ (ppm)",labelpad=-429)
main_ax.tick_params(right = True,left = False,labelleft = False, 
                    labelright=True, which = 'both')
main_ax.invert_yaxis()
main_ax.invert_xaxis()
main_ax.set_xlim(-24, -40) ##CHK BACK
main_ax.set_ylim(-22, -39)  ##CHK BACK
main_ax.minorticks_on()

#xplot.plot(freq2,(np.sum(np.real(spec),0)),'k') #sum
xplot.plot(freq2,(np.max(np.real(spec),0)),'k') #skyline

yplot.plot(np.real(np.sum(spec,1)),fiso,'k')
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

s3.plot(freq2,(np.real(spec[simproc.nearest(fiso,-26.51),:])),'c')
s2.plot(freq2,(np.real(spec[simproc.nearest(fiso,-27.3),:])),'b')
s1.plot(freq2,(np.real(spec[simproc.nearest(fiso,-33.71),:])),'m')

os.chdir(cwd)
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))