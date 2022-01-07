# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
from mpl_toolkits.mplot3d import Axes3D
import scipy 
from scipy.optimize import curve_fit
import pywt
import sys

def loadfid(name,plot='no'):
    """loads Topspin FID with optional plotting"""
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW = float(lines[i].split()[1])
        if lines[i].split()[0] == '##$DTYPA=': #TD actual index
            dt = int(lines[i].split()[1])
    DW = 1/SW
    
    f=open(name, mode='rb')
    fid = np.frombuffer(f.read(), dtype = [np.dtype(np.int32), np.dtype(np.float32), np.dtype(np.float64)][dt]) #float or int #Need to look for byt
    l = int(len(fid))
    Re = fid[0:l:2]
    Im = 1j*fid[1:l:2]
    fid = Re + Im
    
    td = len(fid)
    time = np.linspace(0, DW*td, num=td)
    
    if plot == 'yes':
        fig=plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex = ax1)
        ax1.plot(time,np.real(fid),'b')
        ax2.plot(time,np.imag(fid),'r')
        plt.xlabel('Time (s)')
    return fid

def loadfid2(name,plot='no'):
    """loads Topspin 2D SER"""

    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW2 = float(lines[i].split()[1])
            
        if lines[i].split()[0] == '##$TD=': #TD actual index
            td2 = int(int(lines[i].split()[1]) / 2)
            
        if lines[i] == '##$TD_INDIRECT= (0..7)\n': #TD1 index -1
            td1 = int(lines[i+1].split()[1])
            
        if lines[i] == '##$IN= (0..63)\n': #Assumes in0 for DW1
            DW1 = float(lines[i+1].split()[0])
        if lines[i].split()[0] == '##$DTYPA=': #TD actual index
            dt = int(lines[i].split()[1])
            
    f=open(name, mode='rb') #open(path + "fid", mode='rb')
    fid = np.frombuffer(f.read(), dtype = [np.dtype(np.int32), np.dtype(np.float32), np.dtype(np.float64)][dt]) #float or int #Need to look for byt
    l = int(len(fid))
    fid = fid[0:l:2] + 1j*fid[1:l:2]
            
    DW2 = 1/SW2
    td1 = int(len(fid)/td2)  ##TD1 from acqus seems to miss
    fid = np.reshape(fid,(td1,td2))
    
    t2 = np.linspace(0, DW2*td2, num=td2)
    t1 = np.linspace(0, DW1*td1, num=td1)
    if plot == 'yes':
        plt.contour(t2,t1,np.abs(fid),40)
        plt.title('Magnitude 2D FID')
        plt.xlabel('t$_2$ (s)')
        plt.ylabel('t$_1$ (s)')
    return fid

def freqaxis(spec,unit='kHz'):
    "Generate the referenced frequency axis (in kHz or ppm) as an array"
    
    zfi = spec.size
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW = float(lines[i].split()[1])
            
    cwd = os.getcwd()
    path = cwd + "\pdata\\1"
    os.chdir(path)
    h=open("procs", mode='r')
    lines=h.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$OFFSET=': #Offset actual index
            OFFSET = float(lines[i].split()[1])
        
        if lines[i].split()[0] == '##$SF=': #SW actual index
            SF = float(lines[i].split()[1])
   
    off = ((OFFSET*SF)-SW/2)*1e-3
    freq = np.linspace(-SW/2e3+off, SW/2e3+off, num=zfi)
    if unit == 'ppm':
        freq = (freq*1e3)/SF
    os.chdir(cwd)
    return freq

def autozero(fid,n=0):
    """Automatically calculate zero fill amount for fid"""
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
    """3D mesh plot of a 2D matrix just like Matlab's mesh function"""
    
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf=ax.plot_surface(x.T, y.T, matrix, cmap='cool')
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
        b = (np.transpose(np.arange((-m/2-off),(m/2-off),1)**2)) / ((m**2)/2)
        phase = ph0 + a*ph1 + b*ph2
        spec= np.multiply( spec, np.outer(np.exp((phase*(3.14159j/180))),np.ones((1,n))) )
        if ax == 1:
            spec = np.transpose(spec)
    return spec

def mphase(data,fine=0,ax=1):
    """Manual phasing with matplotlib widget for 1D or 2D data"""
    
    if data.ndim == 1:
        fig = plt.figure()
        plt.subplots_adjust(bottom=0.35)
        ax = fig.subplots()
        p, = ax.plot(np.real(data),'k')
        
        but = plt.axes([0.25, 0.25, 0.65, 0.03])
        off_slide = plt.axes([0.25, 0.2, 0.65, 0.03])
        ph0_slide = plt.axes([0.25, 0.15, 0.65, 0.03])
        ph1_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
        ph2_slide = plt.axes([0.25, 0.05, 0.65, 0.03])
        
        off = Slider(off_slide, 'Offset', valmin= 0, valmax = len(data), 
                          valinit=0, valstep=10)
        ph0 = Slider(ph0_slide, "Zero", valmin= 0, valmax = 360, 
                          valinit=0, valstep=0.5)
        if fine == 0:
            ph1 = Slider(ph1_slide, 'First', valmin= -1e6, valmax = 1e6, 
                              valinit=0, valstep=20)
        else:
            ph1 = Slider(ph1_slide, 'First', valmin= -1e6/fine, valmax = 1e6/fine, 
                              valinit=0, valstep=20)
        ph2 = Slider(ph2_slide, 'Second', valmin= -1e4, valmax = 1e4, 
                          valinit=0, valstep=10)
        b = Button(but, 'Print Phases')
        
        def update(val):
            y = phase(data,[ph0.val,ph1.val,ph2.val,off.val])
            p.set_ydata(np.real(y))
            fig.canvas.draw()
            
        def press(val):
            print('[%d, %d, %d, %d]' % (ph0.val,ph1.val,ph2.val,off.val))
        
        ph0.on_changed(update); ph1.on_changed(update)
        ph2.on_changed(update); off.on_changed(update)
        b.on_clicked(press)
        plt.show(block=True)
        
    elif data.ndim == 2:
        fig = plt.figure()
        grid = plt.GridSpec(4, 3, hspace=0.3, wspace=0.3) #4x5 grid of subplots #spacings for h and w
        main_ax = fig.add_subplot(grid[1:3, 1:3]) 
        yplot = fig.add_subplot(grid[1:3, 0], yticklabels=[])
        xplot = fig.add_subplot(grid[0, 1:3], yticklabels=[], sharex=main_ax)
        
        main_ax.contour(np.real(data),10,cmap='jet')
        xplot.plot(np.flipud(np.sum(np.real(data),0)),'k') #sum
        yplot.plot(np.real(np.sum(data,1)),np.arange((len(np.sum(data,1)))),'k') #sum
        yplot.invert_xaxis()
        
        but = plt.axes([0.25, 0.25, 0.65, 0.03])
        off_slide = plt.axes([0.25, 0.2, 0.65, 0.03])
        ph0_slide = plt.axes([0.25, 0.15, 0.65, 0.03])
        ph1_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
        ph2_slide = plt.axes([0.25, 0.05, 0.65, 0.03])
        
        off = Slider(off_slide, 'Offset', valmin= 0, valmax = len(data), 
                          valinit=0, valstep=10)
        ph0 = Slider(ph0_slide, "Zero", valmin= -180, valmax = 180, 
                          valinit=0, valstep=0.5)
        if fine == 0:
            ph1 = Slider(ph1_slide, 'First', valmin= -1e6, valmax = 1e6, 
                              valinit=0, valstep=20)
        else:
            ph1 = Slider(ph1_slide, 'First', valmin= -1e6/fine, valmax = 1e6/fine, 
                              valinit=0, valstep=20)
        ph2 = Slider(ph2_slide, 'Second', valmin= -1e4, valmax = 1e4, 
                          valinit=0, valstep=10)
        b = Button(but, 'Print Phases and Exit')
        # Updating the plot
        def update(val):
            y = phase(data,[ph0.val,ph1.val,ph2.val,off.val],ax=ax)
            main_ax.clear()
            xplot.clear()
            yplot.clear()
            main_ax.contour(np.real(y),10,cmap='jet')
            xplot.plot(np.flipud(np.sum(np.real(y),0)),'k') #sum
            yplot.plot(np.real(np.sum(y,1)),np.arange((len(np.sum(y,1)))),'k') #sum
            yplot.invert_xaxis()
            fig.canvas.draw()
            
        def press(val):
            print('[%d, %d, %d, %d]' % (ph0.val,ph1.val,ph2.val,off.val))
            plt.close()
            
        # Calling the function "update" when the value of the slider is changed
        ph0.on_changed(update)
        ph1.on_changed(update)
        ph2.on_changed(update)
        off.on_changed(update)
        b.on_clicked(press)
        plt.show(block=True)
    return

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
    M_ph1 = np.zeros(n); M_ph0 = np.zeros(n)
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
    """Automatically coadd all spin echos and FT;
    Specifically works for WCPMG data acquired on NEO with
    or without inconsistent spacings"""
    
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
        
    fidcoadd = np.sum(cpmg, axis=1) 
 
    if plot=='yes':
        plt.plot(np.real(fidcoadd),'m')
        plt.title('Real Coadded FID')
    return cpmg, fidcoadd

def coaddgen(fid,pw=11,dring=3,dwindow=6,loop=22,lrot = 15,MAS='no',plot='no'):
    """Automatically coadd all spin echos. Attempt at generic coaddition.
    Assumes that the CPMG pulse sequence has the following structure:
        
    pulse - tau2 - [tau2 - pulse - tau2 - tau_echo]_N
    
    parameters:
    dring: int
        Index of the ring down delay (tau2) in topsin (e.g., 3 for d3)
    dwindow: int
        Index of the windowed echo delay (tau_echo) in topsin (e.g., 6 for d6)
    loop: int
        Index of the loop counter (N) in topsin (e.g., 20 for l20)
    lrot: int
        Index of the MAS loop counter (M) in topsin (e.g., 15 for l15)
    pw: int
        Index of the pulse width (pulse) in topsin (e.g., 11 for p11)
    
    """
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i] == '##$D= (0..63)\n': #D index -1
            d3 = float(lines[i+1].split()[dring])
            d6 = float(lines[i+1].split()[dwindow])
        
        if lines[i] == '##$L= (0..31)\n': #loop index -1
            l22 = int(lines[i+1].split()[loop]) #CPMG loops
            l15 = int(lines[i+1].split()[lrot]) #MAS rotor sync integer
        
        if lines[i] == '##$P= (0..63)\n': #pulse width index -1
            tp = float(lines[i+1].split()[pw])
        
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

def mqproc(fid, SH = -7/9,q = 0, zf1=0, zf2=0, lb1=0, lb2=0):
    """Process whole-echo MQMAS data with shearing.
    
    Parameters
    ----------
    fid : ndarray
        complex FID matrix
    SH : float
        Shearing constant; default = -7/9 for I = 3/2
    q : int
        Optional input to do Q-shearing, will use zf1 for freq. zero-fill
    zf1 : int
        0-fill F1
    zf2 : int
        0-fill F2
    lb1 : int
        Gaussian broaden t1 without mean (shift) 0
    lb2 : int
        Gaussian broaden t2
    """
    np1 = fid.shape[0]
    np2 = fid.shape[1]
    if zf2 == 0:
        zf2 = autozero(fid[0,:])
    if zf1 == 0:
        zf1 = autozero(fid[:,0])
    for i in range(np2):
        fid[:,i] = gauss(fid[:,i],lb1,c=0)
    
    #FT t2
    spec1 = np.zeros((np1,zf2),dtype='complex')
    for i in range(np1):
        spec1[i,:] = fft(fid[i,:],zf2)
    
    #Shearing t1-F2 matrix
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW2 = float(lines[i].split()[1])
        if lines[i] == '##$IN= (0..63)\n': #Assumes in0 for DW1
            DW1 = float(lines[i+1].split()[0])
    
    freq2 = np.linspace(-SW2/2,SW2/2,zf2) #F2 freq. (Hz)
    t1 = np.arange(0,np1*DW1,DW1)             #t1 time vector (s)
    
    if q != 0:
        spec1 = np.multiply(spec1, np.exp(1j*q*2*np.pi*np.outer(t1,freq2)) )
        #FT t1
        spec = np.zeros((zf1,zf2),dtype='complex') ##rename spec
        for i in range(zf2):
            spec[:,i] = fft(spec1[:,i],np1) ##rename spec #dont ZF the FT here
        
        
        
    
    spec1 = np.multiply(spec1, np.exp(1j*SH*2*np.pi*np.outer(t1,freq2)) )
    
    #iFT t2, centered GB, FT t2 again
    dum = np.fft.fftshift((np.fft.ifft(spec1[0,:])))
    c2 = np.argmax(dum) / zf2
    for i in range(np1):
        spec1[i,:] = np.fft.fftshift( (np.fft.ifft(spec1[i,:])) )
        spec1[i,:] = (np.fft.fft( gauss(spec1[i,:],lb2,c2) ,zf2))

    #FT t1
    spec = np.zeros((zf1,zf2),dtype='complex')
    for i in range(zf2):
        spec[:,i] = fft(spec1[:,i],zf1)
    return spec

def fiso(spec,SH = 7/9,q=3,unit='kHz'):
    """Generate the referenced isotropic F1 axis for MQMAS.
    q: int
        MQC coherence order (default = 3)
    """
    
    zfi = spec.size
    cwd = os.getcwd()
    
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i] == '##$IN= (0..63)\n': #Assumes in0 for DW1
            DW1 = float(lines[i+1].split()[0])
    SW1 = 1/DW1 #Hz
            
    path = cwd + "\pdata\\1"
    os.chdir(path)
    h=open("proc2s", mode='r') ##Pulling from proc2
    lines=h.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$OFFSET=': #Offset actual index
            OFFSET = float(lines[i].split()[1])
        
        if lines[i].split()[0] == '##$SF=': #SW actual index
            SF = float(lines[i].split()[1])
    
    off = ((OFFSET*SF)-SW1/2)  #ref off in Hz
    #freq = np.linspace(-SW/2e3+off, SW/2e3+off, num=zfi)
    
    freq1 =  np.linspace(-SW1/2, SW1/2, zfi) ##F1 in Hz 
    fiso =  ( (freq1+off) / (1 + abs(SH)) )/1e3 ##F1iso in kHz 
    #freq1 = np.linspace(-SW1/2,SW1/2,zfi)
    
    if unit == 'ppm':
        fiso = (freq1)/((SF)*(q-SH)) + (off/SF) #F1iso in ppm
    os.chdir(cwd)
    return fiso

def T2fit(tops,tau):
    """Fit the monoexponential T2 / T2eff of the (W)CPMG echo train"""
        
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
    
def snr(spec,j=0):
    """SNR measure in the frequency domain. Specturm needs to be phased for accurate measure"""
    
    spec = np.real(spec)
    #plt.plot(spec) #plot the spectrum to determine the index #'s of noise basline
    if j!=0:    
        sn = np.max(spec) / np.std(spec[0:j])
    else:
        sn = np.max(spec) / np.std(spec[0:int(0.1*len(spec))])
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
    
    return z

def cadzow(fid,p=10, plot ='no'):
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
    if plot == 'yes':
        #plt.figure(1)
        #plt.plot(s,'*')
        plt.plot(s1,'m.')
        #plt.yscale('log')
        plt.ylabel('Deriv. Singular Value Magnitude')
        plt.xlabel('Singular Value Entry')
        sys.exit()
    
    r = np.argmax(s1[round((p/100)*len(s1)):]) + round((p/100)*len(s1)) #% of SV's to not look at
    print('Ignoring first %d percent of singular values for cut-off' %p)
    print('Retaining %d singular values' % r)
    
    s[(r):] = 0
    sigma = scipy.linalg.diagsvd(s, m, n) #rebuilds s as sigma matrix
    hankrecon = np.matmul( np.matmul(U,sigma), Vt )
    
    ad = [] #anti-diagonal averaging algorithm
    for i in range(m-1):
        ad.append( np.mean( np.fliplr( hankrecon[:i+1,:i+1] ).diagonal()))
    ad = np.array(ad)

    ad2 = []
    for i in range(n):
        ad2.append( np.mean( np.fliplr( hankrecon[(m-1-i):,(n-1-i):] ).diagonal()))
    ad2 = np.flip(ad2)
    
    fidrecon = np.append( ad, ad2 ) #instead of rebuilding hank, just extract the fid
    return fidrecon
