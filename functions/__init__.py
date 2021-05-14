# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy 

def loadFID(name,plot):
    #loads Topspin FID and all other useful info
    
    f=open(name, mode='rb') #open(path + "fid", mode='rb')
    fid = np.frombuffer(f.read(), dtype = float) #float to avoid condvta nonsense
    l = int(len(fid))
    Re = fid[0:l:2]
    Im = 1j*fid[1:l:2]
    fid = Re +Im
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    SW = float(lines[268].split()[1])
    DW = 1/SW
    SFO1 = float(lines[227].split()[1])
    
    td = len(fid)
    time = np.linspace(0, DW*td, num=td)
    
    if plot == 'yes':
        plt.plot(time,np.real(fid),'b')
        plt.plot(time,np.imag(fid),'r')
        plt.xlabel('Time (s)')
    return fid, SW, DW, SFO1

def fmc(fid,SW):
    #loads FID and does 'fmc'
    
    td = len(fid)
    zf = [2**n for n in range(24)] #auto zero-fill
    for i in range(24):
        if td < zf[i]:
            a = i
            break
    zfi =int(zf[a+1])
    spec = np.fft.fftshift(scipy.fft(fid,n=zfi))
    freq = np.linspace(-SW/2e3, SW/2e3, num=zfi)
    
    plt.gca().invert_xaxis()
    plt.plot(freq,np.abs(spec),'m')
    plt.xlabel('Frequency (kHz)')
    return spec