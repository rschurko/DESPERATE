# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import scipy 
from scipy.optimize import curve_fit
import pywt
import sys

def loadfidold(name,plot='no'):
    """loads Topspin FID and other useful info"""
    
    f=open(name, mode='rb') #open(path + "fid", mode='rb')
    fid = np.frombuffer(f.read(), dtype = int) #float to avoid condvta nonsense
    l = int(len(fid))
    Re = fid[0:l:2]
    Im = 1j*fid[1:l:2]
    fid = Re + Im
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    SW = float(lines[269].split()[1]) #269
    DW = 1/SW
    
    td = len(fid)
    time = np.linspace(0, DW*td, num=td)
    
    if plot == 'yes':
        plt.subplot(211)
        plt.plot(time,np.real(fid),'b')
        plt.subplot(212)
        plt.plot(time,np.imag(fid),'r')
        plt.xlabel('Time (s)')
    return fid, SW

def loadfid(name,plot='no'):
    """loads Topspin FID and other useful info"""
    
    f=open(name, mode='rb') #open(path + "fid", mode='rb')
    fid = np.frombuffer(f.read(), dtype = float) #float to avoid condvta nonsense
    l = int(len(fid))
    Re = fid[0:l:2]
    Im = 1j*fid[1:l:2]
    fid = Re + Im
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW = float(lines[i].split()[1])
    DW = 1/SW
    td = len(fid)
    time = np.linspace(0, DW*td, num=td)
    
    if plot == 'yes':
        plt.subplot(211)
        plt.plot(time,np.real(fid),'b')
        plt.subplot(212)
        plt.plot(time,np.imag(fid),'r')
        plt.xlabel('Time (s)')
    return fid, SW

def freqaxis(fid,zf=0):
    "Generate the referenced frequency axis (in kHz) as an array"
    
    zfi = autozero(fid,zf)
    cwd = os.getcwd()
    path = cwd + "\pdata\\1"

    os.chdir(path)
    h=open("procs", mode='r')
    lines=h.readlines()
    SW = float(lines[119].split()[1])
    SF = float(lines[107].split()[1])
    OFFSET = float(lines[87].split()[1])
    
    off = ((OFFSET*SF)-SW/2)*1e-3
    freq = np.linspace(-SW/2e3+off, SW/2e3+off, num=zfi)
    os.chdir(cwd)
    
    return freq

def autozero(fid,n=0):
    """Automatically zero fill fid"""
    if n == 0:
        td = len(fid)
        zf = [2**n for n in range(28)] #auto zero-fill
        for i in range(24):
            if td < zf[i]:
                a = i
                break
        zfi = int(zf[a+1])
    else:
        zfi = int(n)
    return zfi

def gauss(fid,lb,c=0.5):
    """Gaussian line broadening for whole of half echo fid
    Parameters
    ----------
    lb : int or float
        Amount of line-broadening
    c : float
        center of the gaussian curve, between 0 and 1, 0.5 is center (default)
    """
    td = len(fid)
    if lb != 0: 
        sd = 1e3/(lb)
        n = np.linspace(-int(c*td)/2,int((1-c)*td)/2,td)
        gauss = ((1/(2*np.pi*sd))*np.exp(-((n)**2)/(2*sd**2)))
        gbfid = np.multiply(fid,gauss)
    else:
        gbfid = fid
    return gbfid

def em(fid,lb):
    """Exponential line broadening for half echo fid"""
    td = len(fid)
    if lb != 0: #line broadening bit
        sd = 1e3/(lb)
        n = np.linspace(0,td,td)
        lbfid = np.multiply(fid,np.exp(-(n/sd)))
    else:
        lbfid = fid
    return lbfid

def fft(fid,n=0):
    zfi = autozero(fid,n)
    spec = np.fft.fftshift(np.fft.fft(fid,n=zfi))
    return spec

def fft2(fid,zf1,zf2):
    spec = np.fft.fftshift(np.fft.fft2(fid,(zf1,zf2)))
    return spec

def fmc(fid,SW):
    """Calculates a FT and magnitude calculation of the FID"""
    
    zfi = autozero(fid)
    spec = np.fft.fftshift(scipy.fft(fid,n=zfi))
    freq = freqaxis(fid,SW)
    
    plt.gca().invert_xaxis()
    plt.plot(freq,np.abs(spec),'m')
    plt.xlabel('Frequency (kHz)')
    return spec

def mesh(matrix):
    """3D mesh plot of a 2D matrix"""
    
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf=ax.plot_surface(x.T, y.T, matrix, cmap='jet')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()
    return

def phase(spec, phases, ax=0):
    """Phase 1D spectrum up to 2nd-order
    Parameters
    ----------
    spec : numpy array
        Complex spectrum (or FID)
    phases : list
        [ph0,ph1,ph2,offset]
    ax : int
        For 2D phasing choose which axis to apply phases over (default = 0)
    """
    ph0 = phases[0]
    ph1 = phases[1]
    ph2 = phases[2]
    off = phases[3]
    if spec.ndim == 1:
        m = spec.size
        a = np.transpose(np.arange(-m/2, m/2, 1))/m
        b = (np.transpose(np.arange((-m/2-off),(m/2-off),1)**2)) / ((m**2)/2)
        phase = ph0 + a*ph1 + b*ph2
        spec= np.multiply( spec, np.exp((phase*(3.14159j/180))))
    else:
        if ax == 1:
            spec = np.transpose(spec)
        m,n = spec.shape
        a = np.transpose(np.arange(-m/2, m/2, 1))/m
        #a = np.reshape(a,(m,1))
        b = (np.transpose(np.arange((-m/2-off),(m/2-off),1)**2)) / ((m**2)/2)
        #b = np.reshape(b,(m,1))
        phase = ph0 + a*ph1 + b*ph2
        spec= np.multiply( spec, np.outer(np.exp((phase*(3.14159j/180))),np.ones((1,n))) )
        if ax == 1:
            spec = np.transpose(spec)
    return spec

def autophase(spec,n,phase2='no'):
    """Automatically phases spectrum up to second order. Reccommended to use heavy line broadening prior to use"""
    
    td = spec.size
    size = td
    if size > 512:
        size = 1024
    i = np.linspace(0,td,size)     # ph1 variables (in pts)
    j = np.linspace(0,360,size)     # ph0 variables  (in degrees)
    l = np.linspace(600*90,-600*90,size)  # ph2 variables (in degrees)
    ph0_1 = np.zeros(n)
    BestOffset = np.zeros(n)
    #offsets = np.arange(td,-td+1,-4)
    offsets = np.linspace(-td/2,td/2,10)
    Iter1 = np.zeros((n,size))
    M_ph1 = np.zeros(n)
    M_ph0 = np.zeros(n)
    ph1_1 = np.zeros(n)
    Iter3 = np.zeros((n,size))
    area1 = np.zeros(n)
    ph2_1 = np.zeros(n)
    if phase2 == 'yes':
        Iter2 = np.zeros((n,len(offsets),size))
        M_ph2 = np.zeros((n,len(offsets)))
        
    for k in range(n):
        #Find first order phase 
        for r in range(len(i)):
            Iter1[k,r] = np.sum(np.real(phase(spec,[ph0_1[k],360*i[r],ph2_1[k],round(BestOffset[k])])))
        M_ph1[k] = np.max(np.abs(Iter1[k,:]))
        ph1_1[k] = i[np.argwhere(abs(Iter1[k,:]) == M_ph1[k])]
        
        #Find zero order phase
        for r in range(len(j)):
            Iter3[k,r] = np.sum(np.real(phase(spec,[j[r],360*ph1_1[k],ph2_1[k],round(BestOffset[k])])))
        M_ph0[k] = np.max(np.real(Iter3[k,:]))
        a = np.argwhere(np.real(Iter3[k,:]) == M_ph0[k])
        if len(a) > 1:
            a = a[1]
        ph0_1[k] = j[a]
        
        if phase2 == 'yes':
        #Find second order phase
            for mm in range(len(offsets)):
                for r in range(len(l)):
                    Iter2[k,mm,r] = np.sum(np.real(phase(spec,[ph0_1[k],360*ph1_1[k],l[r],(offsets[mm])])))
                M_ph2[k,mm] = np.max(abs(Iter2[k,mm,:]))
        
            BestOffsetIndex = int( np.min( np.argwhere(abs(M_ph2[k,:]) == np.max(abs(M_ph2[k,:]))) ))
            BestOffset[k] = offsets[BestOffsetIndex]
            ph2_1[k] = l[np.argwhere((abs(Iter2[k,BestOffsetIndex])) == abs(M_ph2[k,BestOffsetIndex]))]
    
        else:
            BestOffset[k] = 0
        
        area1[k] = np.sum(np.real(phase(spec,[ph0_1[k],360*ph1_1[k],ph2_1[k],round(BestOffset[k])])))
        
        a = np.argmax(area1) 
        ph0 = ph0_1[a]
        ph1 = ph1_1[a]
        ph2 = ph2_1[a]
    
        i = ph1 + np.linspace(-64/(k+1), 64/(k+1), round(size/(k+1))) #MJJ shrinks these sizes each run, but mismatches with spec
        j = ph0 + np.linspace(0, 360/(k+1), round(size/(k+1)))
        l = ph2 + np.linspace(400*90/(k+1), -400*90/(k+1), round(size/(k+1)))
        offsets = BestOffset[a] + np.linspace(128,-128, round(len(offsets)))
        #offsets = BestOffset[a] + np.linspace(round(td/(2*(k+1))),-round(td/(2*(k+1))), round(td/(k+1)))
        
        print('Iteration %d/%d' % (k+1,n))
        
    off = BestOffset[a]
    phases = [ph0,ph1*360,ph2,off]
    plt.plot(np.real(phase(spec,phases)) + 1*np.max(np.abs(spec)),'m',label='Phased')
    plt.plot(np.abs(spec),'k',label='Magnitude')
    plt.plot(np.abs(spec) - np.real(phase(spec,phases)) + 2*np.max(np.abs(spec)),'r--',label='Difference')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper right')
    
    phases = np.round_(phases,0)
    print('[%d, %d, %d, %d]' % (phases[0],phases[1],phases[2],phases[3]))
    return phases

def coadd(fid,MAS='no',plot='no'):
    """Automatically coadd all spin echos and FT"""
    """Specifically works for WCPMG data acquired on NEO with inconsistent spacings"""
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i] == '##$D= (0..63)\n': #D index -1
            d3 = float(lines[i+1].split()[3])
            d6 = float(lines[i+1].split()[6])
        
        if lines[i] == '##$L= (0..31)\n': #loop index -1
            l22 = int(lines[i+1].split()[22]) #CPMG loops
            l15 = int(lines[i+1].split()[15]) #MAS rotor sync integer
        
        if lines[i] == '##$P= (0..63)\n': #pulse width index -1
            tp = float(lines[i+1].split()[11])
        
        if lines[i].split()[0] == '##$DECIM=': #Decim actual index
            decim = int(round(float(lines[i].split()[1])))
        
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW = float(lines[i].split()[1])
    DW = 1/SW #(s)
    #par = [d3,d6,l22,l15,tp,decim,SW]
    
    fid = fid[(2*decim-1):] #remove decimation points
    fid = fid[:int((l22)*np.round((d6 + 2*d3 + 2e-6 + tp*1e-6)/DW))] #remove trailing pts
    cpmg = np.transpose( np.reshape(fid, (l22, int(len(fid)/l22) )) )
    
    #For echo trains that aren't equally spaced, this roll algorithm will align them
    if MAS == 'yes':
        M = 2*l15-1
        rot = int((d6/DW)/(M)) #If MAS, this is the approx pts. per rotor echo
        l = int(len(cpmg[:,0])/2)
        q = np.argmax(np.abs(np.real( gauss(cpmg[l-rot:l+rot,0],10 ) ))) + (l-rot)
        for i in range(l22):
            r = np.argmax(np.abs(np.real( gauss(cpmg[l-rot:l+rot,i],10) ))) + (l-rot)
            cpmg[:,i] = np.roll(cpmg[:,i],(q-r))
    else:
        q = np.argmax(np.abs(np.real(cpmg[:,0])))
        for i in range(l22):
            r = q - np.argmax(np.abs(np.real(cpmg[:,i]))) #amount to roll by is difference of index of echo tops
            cpmg[:,i] = np.roll(cpmg[:,i],r)
        
    #mesh(np.abs(cpmg))
    #sys.exit()
    fidcoadd = np.sum(cpmg, axis=1) #note that it's more robust to do 2D FT and then coadd
 
    if plot=='yes':
        plt.plot(np.real(fidcoadd),'m')
        plt.title('Real Coadded FID')
        #plt.xlabel('Time (s)')
    
    return cpmg, fidcoadd

def coaddgen(fid,MAS='no',plot='no'):
    """Automatically coadd all spin echos. Attempt at generic coaddition.
    
    parameters:
        dring: int
        dwindow: int
        loop: int
        pw: int
    
    """
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i] == '##$D= (0..63)\n': #D index -1
            d3 = float(lines[i+1].split()[3])
            d6 = float(lines[i+1].split()[6])
        
        if lines[i] == '##$L= (0..31)\n': #loop index -1
            l22 = int(lines[i+1].split()[22]) #CPMG loops
            l15 = int(lines[i+1].split()[15]) #MAS rotor sync integer
        
        if lines[i] == '##$P= (0..63)\n': #pulse width index -1
            tp = float(lines[i+1].split()[11])
        
        if lines[i].split()[0] == '##$DECIM=': #Decim actual index
            decim = int(round(float(lines[i].split()[1])))
        
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW = float(lines[i].split()[1])
    DW = 1/SW #(s)
    #par = [d3,d6,l22,l15,tp,decim,SW]
    
    fid = fid[(2*decim-1):] #remove decimation points
    fid = fid[:int((l22)*np.round((d6 + 2*d3 + 2e-6 + tp*1e-6)/DW))] #remove trailing pts
    cpmg = np.transpose( np.reshape(fid, (l22, int(len(fid)/l22) )) )
    
    #For echo trains that aren't equally spaced, this roll algorithm will align them
    if MAS == 'yes':
        M = 2*l15-1
        rot = int((d6/DW)/(M)) #If MAS, this is the approx pts. per rotor echo
        l = int(len(cpmg[:,0])/2)
        q = np.argmax(np.abs(np.real( gauss(cpmg[l-rot:l+rot,0],10 ) ))) + (l-rot)
        for i in range(l22):
            r = np.argmax(np.abs(np.real( gauss(cpmg[l-rot:l+rot,i],10) ))) + (l-rot)
            cpmg[:,i] = np.roll(cpmg[:,i],(q-r))
    else:
        q = np.argmax(np.abs(np.real(cpmg[:,0])))
        for i in range(l22):
            r = q - np.argmax(np.abs(np.real(cpmg[:,i]))) #amount to roll by is difference of index of echo tops
            cpmg[:,i] = np.roll(cpmg[:,i],r)
        
    #mesh(np.abs(cpmg))
    #sys.exit()
    fidcoadd = np.sum(cpmg, axis=1) #note that it's more robust to do 2D FT and then coadd
 
    if plot=='yes':
        plt.plot(np.real(fidcoadd),'m')
        plt.title('Real Coadded FID')
        #plt.xlabel('Time (s)')
    
    return cpmg, fidcoadd

def T2fit(fid,SW):
    """Fit the monoexponential T2 / T2eff of the WCPMG echo train"""
    
    fid = np.abs(fid)
    DW = 1/SW #(s)
    g=open("acqus", mode='r')
    lines=g.readlines()
    d3 = float(lines[42].split()[3]) #d3 dead time (s)
    d6 = float(lines[42].split()[6]) #d6 acq time (s)
    decim = int(lines[47].split()[1]) #decimation number
    l22 = int(lines[118].split()[22]) #number of echoes
    tp = float(lines[168].split()[11]) #p11 pulse width in WCPMG (us)*
    
    #want the index so you can figure out the time with the dwell
    indx1 = np.argmax(fid[:int(2*decim + (d6+ 2*d3 + 2e-6 + tp*1e-6)/DW -1)]) 
    #print(tau1, indx1)
    
    indx=[indx1]
    tops = []
    tau  = []
    group = int((d6/2+ 2*d3 + 2e-6 + tp*1e-6)/DW) #group delay of dead time and half echo
    for i in range(l22):
        run = np.argmax(fid[ indx[i] + group : indx[i] + group + int((d6)/DW)-1])
        indx.append( indx[i] + group + run )
        tops.append(fid[indx[i]])
        tau.append(indx[i]*DW) #index*DW gives the times of the echo tops
        
    tops = tops/np.amax(tops)
    tau = np.asarray(tau)
    plt.plot(tau,tops,'k')
    plt.xlabel('Time (ms)')
    def f(x,a,T2):
        return a*np.exp(-x/T2)
    
    popt, pcov = curve_fit(f, tau, tops, bounds= (0.0, [3., 1000.]) )
    print('a = %5.3f' % popt[0])
    print('T2 = %5.5f (s)' % popt[1])
    
    rmse = np.sqrt(np.mean(np.square(f(tau, popt[0], popt[1])-tops)))
    print('RMSE = %5.4f' % rmse)
    
    plt.plot(tau, f(tau, popt[0], popt[1]),'r',
             label='fit a*exp[-t/T2]: a=%5.3f, T2=%5.3f (s), RMSE = %5.3f' % (popt[0],popt[1],rmse))        
    plt.legend()
    return tau
    
def snr(spec,j):
    """SNR measure in the frequency domain. Specturm needs to be phased for accurate measure"""
    
    spec = np.real(spec)
    #plt.plot(spec) #plot the spectrum to determine the index #'s of noise basline
    if j!=0:    
        sn = np.max(spec) / np.std(spec[0:j])
    else:
        sn = np.max(spec) / np.std(spec[0:100])
    #print('SNR = %.3f' %sn)
    return sn

def PCA(matrix,r):
    """Denoises m x n complex spectal matrix with PCA reconstruction using r number of components"""
    
    m,n = matrix.shape
    for j in range(n): #'center' the matrix to have mean 0 on the columns
        matrix[:,j] = matrix[:,j] - np.mean(matrix[:,j])
    
    U, s, Vh = scipy.linalg.svd(matrix) #s is a vector of singular values, not a matrix
    plt.figure(1)
    plt.plot(np.abs(s))
    plt.yscale('log')
    plt.ylabel('Singular Value Magnitude')
    plt.xlabel('Singular Value Entry')
    sigma = scipy.linalg.diagsvd(s, m, n) #rebuilds s as sigma matrix
    b = np.dot(sigma, Vh) ##replace with np.matmul or @ 
    z = np.dot(U[:,:r], b[:r,:]) #retain 'r' principal components
    
    ##Run the orthogonal denoising
    # matrix = np.transpose(z)
    # m,n = matrix.shape
    # for j in range(n): #'center' the matrix to have mean 0 on the columns
    #     matrix[:,j] = matrix[:,j] - np.mean(matrix[:,j])
    # U, s, Vh = scipy.linalg.svd(matrix) #s is a vector of singular values, not a matrix
    # sigma = scipy.linalg.diagsvd(s, m, n) #rebuilds s as sigma matrix
    # b = np.dot(sigma, Vh) ##replace with np.matmul or @ 
    # z1 = np.dot(U[:,:r], b[:r,:]) #retain 'r' principal components
    
    return z #np.transpose(z1)

def deT2fit(matrix,tau):
    """Fit the T2 of denoised CPMG data"""
    #can either reccomend to run T2fit first or use T2fit funciton in this function to get tau
    #and either suppress the plot or use as subplot somehow
    matrix = np.abs(matrix)
    m,n = matrix.shape
    tops=[]
    for i in range(n):
        tops.append(np.max(matrix[:,i]))
        
    tops = tops/np.amax(tops)
    #plt.plot(tops)
    plt.plot(tau,tops,'k')
    plt.xlabel('Time (ms)')
    def f(x,a,T2):
        return a*np.exp(-x/T2)
    
    popt, pcov = curve_fit(f, tau, tops, bounds= (0.0, [3., 1000.]) )
    print('a = %5.3f' % popt[0])
    print('T2 = %5.5f (s)' % popt[1])
    
    rmse = np.sqrt(np.mean(np.square(f(tau, popt[0], popt[1])-tops)))
    print('RMSE = %5.4f' % rmse)
    
    plt.plot(tau, f(tau, popt[0], popt[1]),'r',
             label='fit a*exp[-t/T2]: a=%5.3f, T2=%5.3f (s), RMSE = %5.3f' % (popt[0],popt[1],rmse))        
    plt.legend()
    return

def cadzow(fid,p=10):
    """Create Hankel matrix and use SVD to denoise fid
    
    Parameters
    ----------
    fid : 1d array
        complex NMR time-domain FID to be denoised
    p : int
        percentage of first-amount of singular values to ignore when determining how many to discard
        Default = 10 %
    """
    l = round(len(fid)/2)
    #print(l)
    a = fid[:l+1] ##Note that in hankel(c,r) r[0] is ignored*
    ##...so need to incude r[0] as the last point in c!!
    b = fid[l:]
    
    hank = scipy.linalg.hankel(a,b)
    m,n = hank.shape
    print('Hanekl Dimensions = ',m,n)
    #U, s, Vt = scipy.linalg.svd(hank) #s is a vector of singular values, not a matrix, Vt is already transposed
    U, s, Vt = np.linalg.svd(hank) #s is a vector of singular values, not a matrix, Vt is already transposed
    s = np.array(s)
    s1 = np.flipud(np.diff(np.flipud(s)))
    #plt.figure(1)
    #plt.plot(s,'*')
    # plt.plot(s1,'m.')
    # #plt.yscale('log')
    # plt.ylabel('Singular Value Magnitude')
    # plt.xlabel('Singular Value Entry')
    # sys.exit()
    
    r = np.argmax(s1[round((p/100)*len(s1)):]) + round((p/100)*len(s1)) #% of SV's to not look at
    print('Ignoring first %d percent of singular values for cut-off' %p)
    print('Retaining %d singular values' % r)
    
    s[(r):] = 0
    sigma = scipy.linalg.diagsvd(s, m, n) #rebuilds s as sigma matrix
    #plt.imshow(np.abs(sigma))
    hankrecon = np.matmul( np.matmul(U,sigma), Vt )
    
    #plt.imshow(np.abs(hank))
    #plt.imshow(np.abs(hankrecon))
    
    ad = [] #welcome to the inefficient anti-diagonal averaging algorithm
    for i in range(m-1):
        ad.append( np.mean( np.fliplr( hankrecon[:i+1,:i+1] ).diagonal()))
    ad = np.array(ad)

    ad2 = []
    for i in range(n):
        ad2.append( np.mean( np.fliplr( hankrecon[(m-1-i):,(n-1-i):] ).diagonal()))
    ad2 = np.flip(ad2)
    
    fidrecon = np.append( ad, ad2 ) #instead of rebuilding hank, just extract the fid
    return fidrecon
