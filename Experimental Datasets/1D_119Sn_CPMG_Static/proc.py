import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import urQRd1 as proc2
import time
import matplotlib.pyplot as plt
start_time = time.time()

##Script will loop over array of Sn spectra
##Also used highest signal avg as SSIM 
def ssim(specref, measure):
    X = np.real( measure )
    Y = np.real( specref )
    SSIM = (2*np.mean(X)*np.mean(Y) + 0)*( 2*np.cov(X,Y)[0][1] + 0) /( (np.mean(X)**2 + np.mean(Y)**2 +0)*(np.std(X)**2 +np.std(Y)**2 + 0))
    return SSIM

##Get High SNR ref spectrum for SSIM:
gb = 15 #global gaussian broaden
os.chdir( 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\10' ) #1k scan EXP
fid, SW = proc.loadfid('fid',plot='no')  
cpmg, fidcoadd, dumspec = proc.coadd(fid,lb = gb, plot='no')
fidcoadd = fidcoadd[1:]
#phases = proc.autophase(dumspec, 50, phase2 = 'yes')
#sys.exit()
ph = [371.283066651318, 44952.976, -1690.231670275709, 99.559]
spec = proc.phase(proc.fft(fidcoadd),ph)[0,:]
spec = np.real(spec)/np.max(np.real(spec))
##Move on to main loop

path = 'C:\\Users\\SRG\\Documents\\Adam\\Spectromter_Data\\SnO_May30_2021\\'
m = 1
SSIMin =  np.zeros(m)
SSIMout =  np.zeros(m)
for i in range(m):
    kk = 10 + i
    print(kk)
    os.chdir( path + str(kk) )
    fid, SW = proc.loadfid('fid',plot='no')  

    #spec = proc.fmc(fid,SW)

    cpmg, fidcoadd, dumspec = proc.coadd(fid,lb = 0, plot='no') #save lb for cadzow
    fidcoadd = fidcoadd[1:]
    
    specin = proc.phase(proc.fft(proc.gauss(fidcoadd,gb)),ph)[0,:]
    specin = np.real(specin)/np.max(np.real(specin))
    SSIMin[i] = ssim(spec,specin) #SSIM of raw spec

    fidrecon = proc.cadzow(fidcoadd, ph, SW,lb = gb)
    #plt.close()
    specrecon = proc.phase(proc.fft(fidrecon),ph)[0,:]
    specrecon = np.real(specrecon)/np.max(np.real(specrecon))
    
    SSIMout[i] = ssim(spec,specrecon)
    

#tau = proc.T2fit(fid,SW)
#proc.mesh(np.real(cpmg))
#z = proc.PCA(cpmg,2)
#proc.deT2fit(z,tau)

#ph = proc.autophase(proc.fft(fidcoadd), 50, phase2='yes')
#fidrecon = proc2.urQRd(fid, 120, orda=None)  ##Fast Cadzow


#generate ns counter at end

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))