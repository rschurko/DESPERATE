# Adam Altenhof

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
import os
from mpl_toolkits.mplot3d import Axes3D
import scipy 
from scipy.optimize import curve_fit
import scipy.signal as sig
import sys
from scipy.signal import find_peaks
import pybaselines as bl
import wavelet_denoise as wave

#####Plotting Stuff
mpl.rcParams['font.family'] = "arial"
mpl.rcParams['font.size'] = 14
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['figure.dpi'] = 100
#####

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
    fid = fid[0:l:2] + 1j*fid[1:l:2]
    
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
            
        # if lines[i].split()[0] == '##$TD=': #TD actual index
            # td2 = int(int(lines[i].split()[1]) / 2)
            
        if lines[i] == '##$IN= (0..63)\n': #Assumes in0 for DW1
            DW1 = float(lines[i+1].split()[0])
            
        if lines[i].split()[0] == '##$DTYPA=': #TD actual index
            dt = int(lines[i].split()[1])
        # if lines[i] == '##$TD_INDIRECT= (0..7)\n': #TD1 index -1
            # td1 = int(lines[i+1].split()[1])
            
    h=open("acqu2s", mode='r')
    lines=h.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$TD=': #TD actual index
            td1 = int(lines[i].split()[1])
            
    f=open(name, mode='rb')
    fid = np.frombuffer(f.read(), dtype = [np.dtype(np.int32), np.dtype(np.float32), np.dtype(np.float64)][dt]) #float or int #Need to look for byt
    fid = fid[0::2] + 1j*fid[1::2]
            
    DW2 = 1/SW2
    # td1 = int(len(fid)/td2)  ##TD1 from acqus seems to miss
    td2 = int(len(fid)/td1)  ##TD2 from acqus seems to miss
    fid = np.reshape(fid,(td1,td2))
    
    t2 = np.linspace(0, DW2*td2, num=td2)
    t1 = np.linspace(0, DW1*td1, num=td1)
    if plot == 'yes':
        plt.contour(t2,t1,np.abs(fid),40)
        plt.title('Magnitude 2D FID')
        plt.xlabel('t$_2$ (s)')
        plt.ylabel('t$_1$ (s)')
    return fid

def get_group(dic):
    """Return the # points from the Bruker digital filter.
    dic : dictionary of acqus
    """
    
    # g=open("acqus", mode='r')
    # lines=g.readlines()
    # for i in range(len(lines)):
    #     if lines[i].split()[0] == '##$SW_h=': #SW actual index
    #         SW = float(lines[i].split()[1])
    #     if lines[i].split()[0] == '##$DTYPA=': #TD actual index
    #         dt = int(lines[i].split()[1])
    # DW = 1/SW
    
    ##ssnake fcn
    dspfirm = dic["DSPFIRM"]
    digtyp = dic["DIGTYP"]
    decim = int(dic["DECIM"]) # decim can be float ?
    dspfvs = dic["DSPFVS"]
    digmod = dic["DIGMOD"]
    
    if digmod == 0:
        return None
    if dspfvs >= 20:
        return dic["GRPDLY"] #* 2 * np.pi
    if ((dspfvs == 10) | (dspfvs == 11) |
                         (dspfvs == 12) | (dspfvs == 13)):
        grpdly_table = {
            10:     {
                2: 44.7500,
                3: 33.5000,
                4: 66.6250,
                6: 59.0833,
                8: 68.5625,
                12: 60.3750,
                16: 69.5313,
                24: 61.0208,
                32: 70.0156,
                48: 61.3438,
                64: 70.2578,
                96: 61.5052,
                128: 70.3789,
                192: 61.5859,
                256: 70.4395,
                384: 61.6263,
                512: 70.4697,
                768: 61.6465,
                1024:    70.4849,
                1536: 61.6566,
                2048: 70.4924,
                },
            11:    {
                2: 46.0000,
                3: 36.5000,
                4: 48.0000,
                6: 50.1667,
                8: 53.2500,
                12: 69.5000,
                16: 72.2500,
                24: 70.1667,
                32: 72.7500,
                48: 70.5000,
                64: 73.0000,
                96: 70.6667,
                128: 72.5000,
                192: 71.3333,
                256: 72.2500,
                384: 71.6667,
                512: 72.1250,
                768: 71.8333,
                1024: 72.0625,
                1536: 71.9167,
                2048: 72.0313
                },
            12:     {
                2: 46.3110,
                3: 36.5300,
                4: 47.8700,
                6: 50.2290,
                8: 53.2890,
                12: 69.5510,
                16: 71.6000,
                24: 70.1840,
                32: 72.1380,
                48: 70.5280,
                64: 72.3480,
                96: 70.7000,
                128: 72.5240,
                192: 0.0000,
                256: 0.0000,
                384: 0.0000,
                512: 0.0000,
                768: 0.0000,
                1024:    0.0000,
                1536: 0.0000,
                2048: 0.0000
                },
            13:     {
                2: 2.75,
                3: 2.8333333333333333,
                4: 2.875,
                6: 2.9166666666666667,
                8: 2.9375,
                12: 2.9583333333333333,
                16: 2.96875,
                24: 2.9791666666666667,
                32: 2.984375,
                48: 2.9895833333333333,
                64: 2.9921875,
                96: 2.9947916666666667
                }
            }

#            # Take correction from database. Based on matNMR routine (Jacco van Beek), which is itself based
#            # on a text by W. M. Westler and F. Abildgaard.
        return grpdly_table[dspfvs][decim]#* 2 * np.pi
    if dspfvs == 0:
        return None
    if dspfvs == -1: # For FIDs gnereated by genser of genfid for example
        return None

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
        
        if lines[i].split()[0] == '##$SF=': #
            SF = float(lines[i].split()[1])

    off = ((OFFSET*SF)-SW/2)*1e-3
    freq = np.linspace(-SW/2e3+off, SW/2e3+off, num=zfi)
    if unit == 'ppm':
        freq = (freq*1e3)/SF
    os.chdir(cwd)
    return freq

def freqaxis1(spec,unit='kHz'):
    """Generate the referenced F1 axis for 2D data.
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
    
    off = ((OFFSET*SF)-SW1/2)*1e-3  #ref off in Hz
    freq = np.linspace(-SW1/2e3+off, SW1/2e3+off, num=zfi)
    
    # freq =  np.linspace(-(SW1)/2, (SW1)/2, zfi) ##F1 in Hz 
    # fiso =  ( (freq1+off) / (1 + abs(SH)) )/1e3 ##F1iso in kHz 
    #freq1 = np.linspace(-SW1/2,SW1/2,zfi)
    # off = ((OFFSET*SF)-SW1/2)*1e-3
    # freq = np.linspace(-SW/2e3+off, SW/2e3+off, num=zfi)
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

def gauss(fid,lb,c=0.5,ax=0):
    """Gaussian line broadening for whole or half echo fid
    Parameters
    ----------
    fid : ndarray
        1D or 2D NMR FID
    lb : int or float
        Amount of line-broadening
    c : float
        center of the gaussian curve, between 0 and 1; 0.5 is symmetric (default)
    ax : int
        axis to Gaussian broaden over for 2D (0 or 1)
    """
    td = len(fid)
    if lb != 0: 
        sd = 1e3/(lb)
        n = np.linspace(-int(c*td)/2,int((1-c)*td)/2,td)
        gauss = ((1/(2*np.pi*sd))*np.exp(-((n)**2)/(2*sd**2)))
        if fid.ndim == 1:
            gbfid = np.multiply(fid,gauss)
        elif fid.ndim == 2:
            [a,b] = fid.shape
            gbfid = np.zeros((a,b),dtype='complex64')
            if ax == 1:
                n = np.linspace(-int(c*b)/2,int((1-c)*b)/2,b)
                gauss = ((1/(2*np.pi*sd))*np.exp(-((n)**2)/(2*sd**2)))
                for i in range(a):
                    gbfid[i,:] = np.multiply(fid[i,:],gauss)    
            if ax == 0:
                n = np.linspace(-int(c*a)/2,int((1-c)*a)/2,a)
                gauss = ((1/(2*np.pi*sd))*np.exp(-((n)**2)/(2*sd**2)))
                for i in range(b):
                    gbfid[:,i] = np.multiply(fid[:,i],gauss)        
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

def wurst(fid,N):
    """Use a WURST ampltidue profile to line broaden"""
    td = len(fid)
    n = np.linspace(0,td,td)
    
    w = (1 - abs(np.cos((np.pi*n)/td))**N)
    lbfid = np.multiply(fid,w)
    
    return lbfid

def nearest(a, a0):
    "Element index in nd array 'a' closest to the scalar value 'a0'"
    idx = np.abs(a - a0).argmin()
    return idx

def contour(M):
    "Contour a matrix and exit; useful for debugging."
    plt.contour(abs(M),40)
    sys.exit()

def fft(fid,n=0):
    """1D fftshift and fft"""
    zfi = autozero(fid,n)
    spec = np.fft.fftshift(np.fft.fft(fid,n=zfi))
    return spec

def fft2(fid,zf1,zf2):
    spec = np.fft.fftshift(np.fft.fft2(fid,(zf1,zf2)))
    return spec

def fmc(fid,SW):
    """Calculates a FT and magnitude calculation of the FID"""
    
    zfi = autozero(fid)
    spec = np.fft.fftshift(np.fft.fft(fid,n=zfi)).abs()
    return spec

def peakp(f, spec):
    "Find peak width with same units as f"
    
    spec = np.real(spec)
    m = np.argmax(spec)
    idx = nearest(spec, np.max(spec)/2)
    delta = 2*( abs(f[m] - f[idx] ) )
    return delta

def mesh(matrix):
    """3D mesh plot of a 2D matrix just like Matlab's mesh function"""
    
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    # x, y = np.meshgrid(x, y)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf=ax.plot_surface(x.T, y.T, matrix, cmap='cool')
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    
    from mpl_toolkits import mplot3d
     
    # Creating figure
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
     
    # Creating plot
    ax.plot_surface(x.T, y.T, matrix)
     
    # show plot
    plt.show()
    
    
    sys.exit()
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
        spec= np.multiply( spec, np.outer(np.exp((phase*(np.pi*1j/180))),np.ones((1,n))) )
        if ax == 1:
            spec = np.transpose(spec)
    return spec

def mphase(data,fine=0,ax=1):
    """Manual phasing with matplotlib widget for 1D or 2D data
    Parameters
    ----------
    spec : numpy array
        Complex spectrum (or FID)
    fine : float
        scale the ph1 value range by some factor as (ph1/fine)
    ax : int
        For 2D phasing choose which axis to apply phases over (default = 1)
    """
    
    if data.ndim == 1:
        fig = plt.figure()
        plt.subplots_adjust(bottom=0.35)
        ax = fig.subplots()
        p, = ax.plot(np.real(data),'k')
        z, = ax.plot(np.zeros((len(data),1)),'m--')
        
        plt.ylim([-np.max(abs(data)), 1.2*np.max(abs(data))])
        
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
        b = Button(but, 'Print Phases')
        
        def update(val):
            y = phase(data,[ph0.val,ph1.val,ph2.val,off.val])
            p.set_ydata(np.real(y))
            z.set_ydata(np.zeros((len(data),1)))
            fig.canvas.draw()
            
        def press(val):
            print('[%d, %d, %d, %d]' % (ph0.val,ph1.val,ph2.val,off.val))
        
        ph0.on_changed(update); ph1.on_changed(update)
        ph2.on_changed(update); off.on_changed(update)
        b.on_clicked(press)
        # b.close_event(sys.exit()) #try
        fig.canvas.mpl_connect('close_event', sys.exit())
        plt.show(block=True)
        plt.close('all')
        
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
        b = Button(but, 'Print Phases')
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
            
        # Calling the function "update" when the value of the slider is changed
        ph0.on_changed(update)
        ph1.on_changed(update)
        ph2.on_changed(update)
        off.on_changed(update)
        b.on_clicked(press)
        plt.show(block=True)
        plt.close('all')
    return [ph0.val,ph1.val,ph2.val,off.val]

def autophase(spec,n,phase2='no'):
    """Automatically phases spectrum up to second order."""
    
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
    Iter1 = np.zeros((n,size)); Iter3 = np.zeros((n,size))
    M_ph1 = np.zeros(n); M_ph0 = np.zeros(n)
    ph1_1 = np.zeros(n)
    area1 = np.zeros(n)
    ph2_1 = np.zeros(n)
    if phase2 == 'yes':
        Iter2 = np.zeros((n,len(offsets),size))
        M_ph2 = np.zeros((n,len(offsets)))
        
    for k in range(n):
        #Find first-order phase 
        for r in range(len(i)):
            Iter1[k,r] = np.sum(np.real(phase(spec,[ph0_1[k],360*i[r],ph2_1[k],round(BestOffset[k])])))
        M_ph1[k] = np.max(np.abs(Iter1[k,:]))
        ph1_1[k] = i[np.argwhere(abs(Iter1[k,:]) == M_ph1[k])]
        
        #Find zero-order phase
        for r in range(len(j)):
            Iter3[k,r] = np.sum(np.real(phase(spec,[j[r],360*ph1_1[k],ph2_1[k],round(BestOffset[k])])))
        M_ph0[k] = np.max(np.real(Iter3[k,:]))
        a = np.argwhere(np.real(Iter3[k,:]) == M_ph0[k])
        if len(a) > 1:
            a = a[1]
        ph0_1[k] = j[a]
        
        if phase2 == 'yes':
        #Find second-order phase and offset
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
    # plt.plot(np.real(phase(spec,phases)) + 1*np.max(np.abs(spec)),'m',label='Phased')
    # plt.plot(np.abs(spec),'k',label='Magnitude')
    # plt.plot(np.abs(spec) - np.real(phase(spec,phases)) + 2*np.max(np.abs(spec)),'r--',label='Difference')
    # plt.gca().invert_xaxis()
    # plt.legend(loc='upper right')
    
    phases = np.round_(phases,0)
    print('[%d, %d, %d, %d]' % (phases[0],phases[1],phases[2],phases[3]))
    return phases

def coadd(fid,al='no',MAS='no',plot='no'):
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
    if al == 'yes':
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
                r = q - np.argmax(np.abs(np.abs(cpmg[:,i]))) #amount to roll by is difference of index of echo tops
                cpmg[:,i] = np.roll(cpmg[:,i],r)
        
    fidcoadd = np.sum(cpmg, axis=1) 
 
    if plot=='yes':
        plt.plot(np.real(fidcoadd),'m')
        plt.title('Real Coadded FID')
    return cpmg, fidcoadd

def coaddgen(fid,pw=11,dring=3,dwindow=6,loop=22,lrot = 15,MAS='no', al='no'):
    """Automatically coadd all spin echos. Attempt at generic coaddition.
    Assumes that the CPMG pulse sequence has the following structure:
        
    pulse - tau1 - [tau2 - pulse - tau2 - tau_echo]_N
    
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
    # l22 = 51 #may need to hardcode a re-write
    DW = 1/SW #(s)
    #par = [d3,d6,l22,l15,tp,decim,SW]
    
    # fid = fid[(2*decim-1):] #remove decimation points
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
    elif al=='yes':
        q = np.argmax(np.abs(np.real(cpmg[:,0])))
        for i in range(l22):
            r = q - np.argmax(np.abs(np.real(cpmg[:,i]))) #amount to roll by is difference of index of echo tops
            cpmg[:,i] = np.roll(cpmg[:,i],r)
        
    fidcoadd = np.sum(cpmg, axis=1) #note that it's more robust to do 2D FT and then coadd
    
    return cpmg, fidcoadd

def mqproc(fid, SH = -7/9, q = 0, zf1=0, zf2=0, lb1=0, lb2=0):
    """Process whole-echo MQMAS data with shearing.
    
    Parameters
    ----------
    fid : ndarray
        complex FID matrix
    SH : float
        Shearing constant; default = -7/9 for I = 3/2
    q : int
        Optional input to do Q-shearing with Fiso expansion, will use zf1 for freq. zero-fill
    zf1 : int
        0-fill F1
    zf2 : int
        0-fill F2
    lb1 : int
        Gaussian broaden t1 without mean (shift) 0
    lb2 : int
        Gaussian broaden t2
    """
    np1 = fid.shape[0]; np2 = fid.shape[1]
    if zf2 == 0:
        zf2 = autozero(fid[0,:])
    if zf1 == 0:
        zf1 = autozero(fid[:,0])
    for i in range(np2):
        fid[:,i] = gauss(fid[:,i],lb1,c=0)
    
    #FT t2
    spec1 = (np.fft.fftshift(np.fft.fft(fid,zf2,axis=1),axes=1))
    
    #Shearing t1-F2 matrix
    g=open("acqus", mode='r')
    lines=g.readlines()
    for i in range(len(lines)):
        if lines[i].split()[0] == '##$SW_h=': #SW actual index
            SW2 = float(lines[i].split()[1])
        if lines[i] == '##$IN= (0..63)\n': #Assumes in0 for DW1
            DW1 = float(lines[i+1].split()[0])
    
    freq2 = np.linspace(-SW2/2,SW2/2,zf2) #F2 freq. (Hz)
    t1 = np.linspace(0,np1*DW1,np1)             #t1 time vector (s)
    
    if q != 0:
        p = int((zf1-np1)/2)
        spec = np.zeros((zf1,zf2),dtype='complex')
        spec1 = np.multiply(spec1, np.exp(1j*q*2*np.pi*np.outer(t1,freq2)) ) #Q-shear
        #FT t1
        specq = np.zeros((np1,zf2),dtype='complex') ##rename spec
        for i in range(zf2):
            specq[:,i] = fft(spec1[:,i],np1) #dont ZF the FT here
        
        specq = np.roll(specq,6,axis = 0) #kill
        
        specq2 = np.pad(specq, [(p,p), (0,0)], mode='constant',constant_values=0) #0-filled
        
        l = (zf1/np1)
        for i in range(zf2): #IFT for reverse-Q-shear
            spec[:,i] = np.fft.ifft(specq2[:,i]) #dont ZF the FT here

        t1s = np.arange(0,zf1*(DW1/l),(DW1/l)) #faster t1 from 0-fill
        spec = np.multiply(spec, np.exp(-1j*q*2*np.pi*np.outer(t1s,freq2)) ) #reverse-Q-shear
        spec = np.multiply(spec, np.exp(1j*SH*2*np.pi*np.outer(t1s,freq2)) ) #isotropic shear
        
        for i in range(zf2): #FT t1 post shear
            spec[:,i] = np.fft.fft(spec[:,i]) ##rename spec #dont ZF the FT here
        
        z = np.unravel_index(spec.argmax(), spec.shape)
        c2 = np.argmax(np.abs(np.fft.fftshift((np.fft.ifft(spec[z[0],:]))))) / zf2
        for i in range(zf1):
            spec[i,:] = np.fft.fftshift( (np.fft.ifft(spec[i,:])) )
            spec[i,:] = (np.fft.fft( gauss(spec[i,:],lb2,c2)))

    else:
        spec1 = np.multiply(spec1, np.exp(1j*SH*2*np.pi*np.outer(t1,freq2)) ) #normal shear
        
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

def fiso(spec,SH = 7/9,q=3,unit='kHz',s=1):
    """Generate the referenced isotropic F1 axis for MQMAS.
    q: int
        MQC coherence order (default = 3)
    s: int
        If you used Q-shearing, scale Fiso by this factor (zf1/np1)
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
    
    freq1 =  np.linspace(-(SW1)/2, (SW1)/2, zfi)*s ##F1 in Hz 
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
    plt.plot(tau,tops,'c*')
    plt.xlabel('Time (ms)')
    def f(x,a,T2):
        return a*np.exp(-x/T2)
    
    popt, pcov = curve_fit(f, tau, tops, bounds= (0.0, [3., 1000.]) )
    print('a = %5.3f' % popt[0])
    print('T2 = %5.5f (s)' % popt[1])
    
    rmse = np.sqrt(np.mean(np.square(f(tau, popt[0], popt[1])-tops)))
    print('RMSE = %5.4f' % rmse)
    
    plt.plot(tau, f(tau, popt[0], popt[1]),'k',
             label='fit a*exp[-t/T2]: a=%5.3f, T2=%5.3f (s), RMSE = %5.3f' % (popt[0],popt[1],rmse))        
    plt.legend()
    
    print('sigma =',np.sqrt(np.diag(pcov)))
    return popt, np.sqrt(np.diag(pcov))
    
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
    # plt.yscale('log')
    plt.ylabel('Singular Value Magnitude')
    # plt.xlabel('Singular Value Entry')
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

def plot2(spec, freq1, base, s, units = 'kHz', xi = None, xj = None, yi = None, yj = None):
    """Nice 2D plotting with all axes controlled together.
    
    spec : spectrum
    freq1 : provide an axis for F1, since data could be pseudo 2D
    base : baseline contour %
    
    s : switch for skyline or max projection or both (0,1,2)
    
    units : 'kHz' or 'ppm'
    
    xij, yij = limits for axes
    """
    
    if units == 'JRES':
        freq2 = freqaxis(spec[0,:],unit='ppm')
    else:
        freq2 = freqaxis(spec[0,:],unit=units)
        
        
    # freq1 = fiso(spec[:,0],unit=units)
    
    h = np.max(np.real(spec))
    lvls = np.linspace((base*1e-2)*h,h,30)
    
    # Set up the axes with gridspec 
    fig = plt.figure(figsize=(12, 8)) # figure size w x h
    grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.6) #4x5 grid of subplots #spacings for h and w
    main_ax = fig.add_subplot(grid[1:, 1:4]) 
    
    yplot = fig.add_subplot(grid[1:, 0], sharey=main_ax)
    xplot = fig.add_subplot(grid[0, 1:4], yticklabels=[], sharex=main_ax)
    
    main_ax.contour(freq2,freq1,(np.real(spec)),lvls,cmap='jet')
    if units == 'ppm':
        main_ax.set_xlabel('F$_{2}$ (ppm)')
        main_ax.set_ylabel("F$_{1}$ (ppm)", labelpad=-429)
    if units == 'kHz':
        main_ax.set_xlabel('F$_{2}$ (kHz)')
        main_ax.set_ylabel("F$_{1}$ (kHz)", labelpad=-429)
    if units == 'JRES':
        main_ax.set_xlabel('F$_{2}$ (ppm)')
        main_ax.set_ylabel("F$_{1}$ (Hz)", labelpad=-429)
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
    
    # yplot.invert_xaxis()
    # yplot.axis('off')
    # xplot.axis('off')
    yplot.invert_xaxis()
    # s1.axis('off')
    yplot.axis('off')
    xplot.axis('off')
    main_ax.xaxis.grid(True, zorder=0)
    main_ax.yaxis.grid(True, zorder=0)
    
    
def corr_mls(data, n, ref):
    """
    Calculate the correlation between an input signal and an MLS signal of defined length.
    Normalizes the MLS and the input signal to vary between +/-1.
    
    data : nd array
    n : power for the MLS
    ref : function to multiply the MLS by
    
    """
    
    l = 2**n-1
    
    td = len(data) ##total length of sim data.
    m = int(td/l)
    
    s0 = sig.max_len_seq(n)[0]-0.5 #nmr data is centered on zero
    
    seq0 = []
    for i in range(m):
        seq0 = np.concatenate((seq0,s0), axis=None)
    
    seq0 = seq0*ref
    
    seq0 = seq0 / np.max(seq0)
    data = data/np.max(data)
    
    # plt.figure(4)
    # plt.plot(data)
    # # plt.plot(np.fft.fftshift(np.fft.fft(seq0)))
    # plt.plot((seq0))
    # sys.exit()
    
    out = sig.correlate(data, seq0, mode='same', method = 'fft')
    
    print('length = %d' %(l))
    
    return out


def region_spec(data, thresh=1):
    """
    Find regions in a spectrum that contain signal via a 2nd derivative.
    Gives an 8 percent buffer to to the found peak widths.
    """
    
    diff = np.diff(np.diff(data))
    # plt.plot(diff)
    # 1/0
    l0 = find_peaks(diff, prominence=max(diff)*(thresh*1e-2))[0]
    
    l=[]
    for i in range(int(len(l0)/2)):
        l+= list(range( int(l0[2*i]*0.92), int(l0[2*i+1]*1.08)) ) 
    return l

# def poly()

def bl_poly(data, order, g, plot='no'):
    """
    Perform polynomial baseline correction.
    data: real phased, positive-peak NMR data
    order: int
    region: binary list of regions where there are signals
    """
            
    # np.where((l==0)|(l==1), l^1, l)
    # l = np.ones((len(data),))
    # l[l0] = 0
    
    base = bl.polynomial.poly(data, None, order, weights = g)[0]
    
    if plot == 'yes':
        plt.plot(data)
        plt.plot(np.array(g)*max(base))
        plt.plot(base)
    # 1/0
    
    return data - base


def fit_poly(data, order, g, plot='no'):
    """
    Perform polynomial baseline correction.
    data: real phased, positive-peak NMR data
    order: int
    region: binary list of regions where there are signals
    """
    
    base = bl.polynomial.poly(data, None, order, weights = g)[0]
    
    if plot =='yes':
        plt.plot(data)
        plt.plot(np.array(g)*max(base))
        plt.plot(base)
    
    return base

def tsepsyche(data, dic):
    """2D -> 1D TSE-PSYCHE Processing."""
    
    SW1 = dic['acqu2s']['SW_h']
    dw2 = 1/( dic['acqus']['SW_h'] )
    conc = 1/(SW1*dw2)
    grp = get_group(dic['acqus'])
    
    # print('Grp. Delay = %.2f'%grp)

    data = data[:,grp:(grp+int(conc))]
    # contour(data.real)
    # 1/0

    fid = np.reshape(data, (data.shape[0]*data.shape[1],) )
    return fid

def simJ(M, J, sw, zf, r2):
    """"Simulate a weak J-coupled spectrum
    M: list of multiplicities
    J: list of J couplings
    sw: spectral window [Hz]
    zf: numper of spectral points    
    r2: T2 rate or line broadening
    """
    
    dw = 1/sw
    td = int(zf/4)
    t = np.linspace(0, td*dw,td)
    
    if len(M) != len(J):
        raise KeyError('Number of Spins is Inconsistent!')
    
    J1 = np.ones((td,))
    for k in range(len(M)):
        
        J1 = np.multiply( np.cos((J[k]/2)*2*np.pi*t)**(M[k]-1) , J1) 
    
    ref = np.exp(-r2*t)*np.exp(-1j*0*t)*J1
    spec = ( np.fft.fftshift(np.fft.fft(ref,zf)) ).real
    
    return spec / max(spec)


# print('Finished!')
