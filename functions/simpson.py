#Functions for processing SIMPSON data and some numerical tools
import numpy as np
import matplotlib.pyplot as plt
import pandas

def read(name,lb,plot):
    """Read SIMPSON file; preferrably the FID"""
    
    refid = np.array(pandas.read_csv(name, sep=' ',skiprows=[0,1,2,3], skipfooter=1,
                    engine='python',index_col=0))
    imfid = np.array(pandas.read_csv(name, sep=' ',skiprows=[0,1,2,3], skipfooter=1,
                    engine='python',index_col=1))
    fid = refid+1j*(imfid)
    
    g=open(name, mode='r')
    lines=g.readlines()
    g.close()
    td = int(lines[1].split('=')[1])
    SW = float(lines[2].split('=')[1])
    DW = 1/SW
    time = np.linspace(0, DW*float(td), num=td)
    
    fid = np.reshape(fid,(td,))
    fid = fid/np.max(fid)
    
    if lb!=0:
        fid = np.multiply( fid, np.exp(-time/lb) )
        
    if plot == 'yes':
        plt.subplot(211)
        plt.plot(time,np.real(fid),'b')
        plt.title('Real')
        plt.subplot(212)
        plt.plot(time,np.imag(fid),'r')
        plt.title('Imaginary')
        plt.xlabel('Time (s)')
    
    return fid, SW

def read2(name,plot='no'):
    """Read SIMPSON file; preferrably the FID"""
    
    refid = np.array(pandas.read_csv(name, sep=' ',skiprows=[0,1,2,3], skipfooter=1,
                    engine='python',index_col=0))
    imfid = np.array(pandas.read_csv(name, sep=' ',skiprows=[0,1,2,3], skipfooter=1,
                    engine='python',index_col=1))
    fid = refid+1j*(imfid)
    
    
    g=open(name, mode='r')
    lines=g.readlines()
    g.close()
    td = int(lines[1].split('=')[1])
    SW = float(lines[2].split('=')[1])
    DW = 1/SW
    time = np.linspace(0, DW*float(td), num=td)
    
    fid = np.reshape(fid,(td,))
        
    if plot == 'yes':
        plt.subplot(211)
        plt.plot(time,np.real(fid),'b')
        plt.title('Real')
        plt.subplot(212)
        plt.plot(time,np.imag(fid),'r')
        plt.title('Imaginary')
        plt.xlabel('Time (s)')
    
    return fid, SW


def read3(name,plot='no'):
    """Read SIMPSON file; preferrably the FID"""
    
    g=open(name, mode='r')
    lines=g.readlines()
    g.close()
    td = int(lines[1].split('=')[1])
    SW = float(lines[2].split('=')[1])
    DW = 1/SW
    time = np.linspace(0, DW*float(td), num=td)
    
    d = np.loadtxt(name, skiprows = 5, max_rows = td)
    
    fid = d[:,0]+1j*(d[:,1])
    
    fid = np.reshape(fid,(td,))
        
    if plot == 'yes':
        plt.subplot(211)
        plt.plot(time,np.real(fid),'b')
        plt.title('Real')
        plt.subplot(212)
        plt.plot(time,np.imag(fid),'r')
        plt.title('Imaginary')
        plt.xlabel('Time (s)')
    
    return fid, SW

def freqaxis(spec,SW,off = 0):
    """Frequency Axis in kHz"""
    freq = np.linspace(-SW/2e3+off, SW/2e3+off, len(spec))
    return freq

def noise(fid, th):
    """Add Guassian noise to simulated data. Fid is normalized and so is noise."""
    
    noise = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(len(fid),2)).view(np.complex128)
    noise = noise[:,0]
    fid = fid + th*noise
    
    return fid

def snrp(spec,i,j,th=0.1):
    """Peak-to-peak SNR. Need to know the indicies of the max [i] and min[j] peaks."""
    
    spec = np.real(spec)
    sn = ( spec[i] - spec[j] ) / np.std(spec[0:int(th*len(spec))])
    #print('SNRp = %.3f' %sn)
    return sn

def nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def ssim(specref, measure):
    X = np.real( measure )
    Y = np.real( specref )
    SSIM = (2*np.mean(X)*np.mean(Y) + 0)*( 2*np.cov(X,Y)[0][1] + 0) /( (np.mean(X)**2 + np.mean(Y)**2 +0)*(np.std(X)**2 +np.std(Y)**2 + 0))
    return SSIM

def coadd(fid,td,r):
    """Coadd simpson cpmg echo train with r # echoes for td # points"""
    
    fid = np.reshape(fid,(r,td))
    fidcoadd = np.sum(fid,axis=0)
    return fidcoadd
    
def residual(specref,specrecon):
    """compare source signal and reconstructed signal"""
    
    residual = np.sum( np.real( specref - specrecon ) ) / np.sum(np.real(specref))
    
    print('Residual = %5.3f' % residual)
    
    return residual

def rmse(specref,specrecon):
    """RMSE between two vectors"""
    ref = np.real(specref)/np.max(np.abs(np.real(specref)))
    result = np.real(specrecon)/np.max(np.abs(np.real(specrecon)))
    
    rmse = np.sqrt(np.mean((result-ref)**2))
    return rmse