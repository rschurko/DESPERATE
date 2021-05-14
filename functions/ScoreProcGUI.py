# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:08:15 2020

@author: mason42
"""

from tkinter import *
from tkinter.ttk import *
from tkinter.messagebox import askokcancel, showerror, showinfo
import tkinter.filedialog as fileD
import numpy as np, pylab as plt
from glob import glob
import Pmw
from os import path
from scipy.optimize import nnls, leastsq
from phasecorr import baseline_corr
from numpy import linalg as la
import nmrglue.fileio.bruker as br
import itertools
from scipy.optimize import leastsq, fmin_l_bfgs_b


global invrecovery, satrecovery, exp_decay, err, score, score_inner

def invrecovery(amp, t, T1):
    return amp*(1-2*np.exp(-(t/T1)))

def satrecovery(amp, t, T1):
    return amp*(1-np.exp(-t/T1))

def exp_decay(amp, t, T2):
    return amp*np.exp(-t/T2)

def res(v, t, y, opt):
    if opt == 1:
        return y - invrecovery(v[0], t, v[1])
    elif opt == 2:
        return y - satrecovery(v[0], t, v[1])
    else:
        return y - exp_decay(v[0], t, v[1])

def core(Dval, t2dat, intens, ncomp, opt):
    C= np.zeros((len(t2dat), ncomp))
    S = np.zeros((ncomp, len(intens)))
    for k in range(ncomp):
        if opt == 1:
            C[:,k] = invrecovery(1, t2dat, Dval[k])
        elif opt == 2:
            C[:,k] = satrecovery(1, t2dat, Dval[k])
        else:
            C[:,k] = exp_decay(1, t2dat, Dval[k])
    S = np.dot(intens, la.pinv(C.T)).T
    return C, S
    
def score_inner(Dval, t2dat, intens, ncomp, outscore, opt):
    C,S = core(Dval, t2dat, intens, ncomp, opt)
    if outscore == 1:
        for m in range(ncomp):
            S[m-1] = S[m-1]/np.sum(np.abs(S[m-1]))        
        CTalk = np.ones((1, len(intens)))
        CPairs = np.array(list(itertools.combinations(np.arange(ncomp), 2)))
        rows, columns = CPairs.shape
        for j in range(rows):
            CTalk = CTalk + S[CPairs[j,0]]*S[CPairs[j,1]]
            res = np.sum(CTalk)
    else:
        drec = np.dot(C, S).T
        res = np.sum((intens - drec)**2)
    return res

class Menu():
    def __init__(self, parent):
        self.master = parent
        self.menu_bar = Pmw.MenuBar(parent, 
                                    hull_relief = 'raised',
                                    hull_borderwidth = 1,
                                    hotkeys = True)
        self.menu_bar.pack(fill = 'x')
        self.menu_bar.addmenu('File', None) 
        self.menu_bar.addmenu('Save', None)
        self.menu_bar.addmenuitem('File', 'command', label = 'Open', command = self.file_read)
        self.menu_bar.addmenuitem('File', 'command', label = 'Quit', command = self.quit)
        self.menu_bar.addmenuitem('Save', 'command', label = 'Save graphics', command = self.save_grp)		
        self.menu_bar.addmenuitem('Save', 'command', label = 'Save XY data', command = self.save_xy)
    	
    def quit(self, event = None):
        ans = askokcancel('Verify exit', 'Really Quit?')
        if ans: self.master.destroy(); plt.close()
    
    def file_read(self):
        global x; global data; global vdlist; global folder
        folder = fileD.askdirectory()
        vdfile = folder +'//vdlist'
        vpfile = folder +'//vplist'
        if folder:
            dct, data = br.read_pdata(folder+'//pdata//1')
            if data.ndim == 2:
                if path.exists(vdfile):
                    vdlist = np.loadtxt(vdfile)
                elif path.exists(vpfile):
                    f = open(vpfile)
                    vdlist = f.read()
                    f.close()
                    vdlist = vdlist.replace('m', '').split('\n')
                    vdlist = np.array([float(i) for i in vdlist])
                off = dct['procs']['OFFSET']
                swp = dct['procs']['SW_p'] / dct['acqus']['SFO1']
                x =  np.linspace(off, off-swp, num=len(data.T))
                plt.clf()
                plt.title('First Spectrum', fontsize = 16)
                plt.plot(x, data[0])
                plt.show()
    
    def save_grp(self):
        fname = fileD.SaveAs(
            filetypes = [('PNG image files', '*.png')],
			initialfile = 'myfile.png',
			title = 'Save graphics as an image file').show()
			
        if fname:
            plt.savefig(fname)			


    def save_xy(self):
        fnameme = fileD.SaveAs(
            filetypes = [('XY files', '*.xy')],
			initialfile = 'myfile.xy',
			title = 'Save output as an XY file').show()
			
        if fname:
            np.savetxt(fname, outxy, fmt = "%1.6e")

class OUTSCORE(Frame):
    def __init__(self, parent):
        self.master = parent
        global p
        self.p = {}
        self.row_counter = 0
        self.p['sel'] = IntVar(); self.p['sel'].set(1) 
        self.p['bkgnd'] = IntVar(); self.p['bkgnd'].set(0)
        Checkbutton(root, 
                    text = 'background subtraction',
                    variable=self.p['bkgnd']).pack()
        Label(root,
              text = 'Number of Spectra to use').pack()
        self.Numb = IntVar(); self.Numb.set(26)
        Entry(root, 
              textvariable = self.Numb, 
              width = 3).pack(anchor = W)
        self.p['ncomp'] = IntVar(); self.p['ncomp'].set(2)
        Label(root, 
              text = 'Choose number of components',
              justify = LEFT).pack()
        Entry(root,
              textvariable = self.p['ncomp'],
              width = 3).pack(anchor = W)
        Label(root,
              text="Choose Fitting Method",
              justify = LEFT).pack()
        Radiobutton(root, 
                    text = "SCORE", 
                    variable = self.p['sel'],
                    value = 0).pack(anchor=W)
        Radiobutton(root, 
                    text = "OUTSCORE", 
                    variable = self.p['sel'],
                    value = 1).pack(anchor=W)
        Label(root,
              text='Choose Model',
              justify = LEFT).pack()
        self.p['mod'] = IntVar(); self.p['mod'].set(1)
        Radiobutton(root,
                    text="Inversion Recovery",
                    variable = self.p['mod'],
                    value = 1).pack(anchor=W)
        Radiobutton(root,
                    text="Saturation Recovery",
                    variable = self.p['mod'],
                    value = 2).pack(anchor=W)
        Radiobutton(root,
                    text='Exponential Decay',
                    variable = self.p['mod'],
                    value = 0).pack(anchor=W)
        Button(root,
               text = 'Preprocess',
               command = self.preprocess).pack()
        Button(root,
               text = 'Run SCORE',
               command = self.ScoreProc).pack()
        
        
    def preprocess(self):
        global data
        numb = self.Numb.get()
        data = data[:numb]
        if self.p['bkgnd'].get() == 1:
        #bcorr = np.zeros(data.shape)
            for i in range(len(data)):
                s0 = data[i]
                data[i] = baseline_corr(s0, buff = 2, filt = 4, smpts=100)
        plt.clf()
        plt.title('Pre-processed Spectra', fontsize = 16)
        plt.plot(x, data.T)
    
    def ScoreProc(self):
        opt = self.p['mod'].get(); sel = self.p['sel'].get()
        "use position of maximum intensity in final data set as initial guess" 
        init = np.sum(data, axis = 1)
        idx = np.where(data[-1] == np.max(data[-1]))[0][0]
        init = data[:,idx]
        v0 = np.array([1e9, .5])
        fit0, success = leastsq(res, v0, args = (vdlist, init, opt), maxfev = 2000)
        print(fit0)
        "generate inital guess for outscore processsing"
        ncomp = self.p['ncomp'].get()
        t2center = fit0[1]
        if ncomp % 2 == 0:
            dval = np.linspace(t2center/(ncomp/2*1.5), t2center*(ncomp/2*1.5), ncomp)
        elif ncomp > 2:
            dval = np.linspace(t2center/((ncomp-1)/2*1.5), t2center*((ncomp-1)/2*1.5), ncomp)
        else:
            dval = t2center

        "use a bounded minimization technique to find the T1 values"
        bounds = ncomp*([[0.1e-9,None]])
        t2fin = fmin_l_bfgs_b(score_inner, dval, args = (vdlist, data.T, ncomp, sel, opt),
                              bounds = bounds, approx_grad=True)
        C, S = core(t2fin[0], vdlist, data.T, ncomp, opt)
        np.savetxt(folder+'//spect.xy', np.vstack((x, S)).T)
        np.savetxt(folder+'//conc.xy', np.vstack((vdlist, C.T)).T)
        rel = np.sum(S[0])/np.sum(S)
        plt.figure()
        plt.title('Final Components', fontsize = 16)
        plt.subplot(121)
        if opt == 1:
            plt.plot(np.log10(vdlist), C)
            print('T1s are', t2fin[0]) #[0], 's and', t2fin[0][1], 's')
        else:
            plt.plot(vdlist, C)
            print('T1rs are', t2fin[0]) #[0], 'ms and', t2fin[0][1], 'ms')
        plt.subplot(122)
        plt.plot(x, S.T)
        plt.show()
        print('relative abundance =', rel, 'and', 1-rel)
        
            

root = Tk()
root.title('SCORE processing')
Menu(root)
OUTSCORE(root)
root.mainloop()
