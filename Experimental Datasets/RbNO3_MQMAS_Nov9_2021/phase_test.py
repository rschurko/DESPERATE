# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import simpson as simproc
import wavelet_denoise as wave
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib as mpl
import time
start_time = time.time()

cwd =  os.getcwd()
os.chdir(cwd + '\\' + '15')
fid = proc.loadfid2('ser',plot='no')

#Params:
nzf1 = 512
nzf2 = 4096
base = 2 #base contour %'age

##SH = +7/9 for this coherence selection
spec = proc.mqproc(fid, SH = -7/9, zf1=nzf1, zf2=nzf2, lb1=0, lb2=15) 

fid2 = np.fft.ifft(spec,axis = 0)
fid2 = np.roll(fid2,1,axis=0)   #shift 1 point in t1 to correct phase since t1 = 0 is not acq'd
spec = np.fft.fft(fid2,axis=0)
#Phase
#ph = [0,0, 0, 0]
ph = [361, 798382, 0, 0]
#ph = [362-25 - 10, 798157 - 150, 0, 0]
#ph = proc.autophase(spec[313,:],50,phase2='no')
#ph = proc.autophase(spec[:,1766],10,phase2='no')
spec = proc.phase(spec,ph,ax=1)
#sys.exit()


# import numpy as np
# from scipy.interpolate import UnivariateSpline
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# Initial x and y arrays

data = spec

# def mphase2(data,fine=0):
#     """Manual phasing with matplotlib widget"""
    
#     fig = plt.figure()
#     grid = plt.GridSpec(4, 3, hspace=0.3, wspace=0.3) #4x5 grid of subplots #spacings for h and w
#     main_ax = fig.add_subplot(grid[1:3, 1:3]) 
#     yplot = fig.add_subplot(grid[1:3, 0], yticklabels=[])
#     xplot = fig.add_subplot(grid[0, 1:3], yticklabels=[], sharex=main_ax)
    
#     main_ax.contour(np.real(data),12,cmap='jet')
#     xplot.plot(np.flipud(np.sum(np.real(data),0)),'k') #sum
#     yplot.plot(np.real(np.sum(data,1)),np.arange((len(np.sum(data,1)))),'k') #sum
#     #p, = ax.plot(x_spline, y_spline, 'g')
    
#     # Defining the Slider button
#     # xposition, yposition, width and height
#     but = plt.axes([0.25, 0.25, 0.65, 0.03])
#     off_slide = plt.axes([0.25, 0.2, 0.65, 0.03])
#     ph0_slide = plt.axes([0.25, 0.15, 0.65, 0.03])
#     ph1_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
#     ph2_slide = plt.axes([0.25, 0.05, 0.65, 0.03])
#     # Properties of the sliders
#     off = Slider(off_slide, 'Offset', valmin= 0, valmax = len(data), 
#                       valinit=0, valstep=10)
#     ph0 = Slider(ph0_slide, "Zero", valmin= -180, valmax = 180, 
#                       valinit=0, valstep=0.5)
#     if fine == 0:
#         ph1 = Slider(ph1_slide, 'First', valmin= -1e5, valmax = 1e5, 
#                           valinit=0, valstep=20)
#     else:
#         ph1 = Slider(ph1_slide, 'First', valmin= -1e5/fine, valmax = 1e5/fine, 
#                           valinit=0, valstep=20)
#     ph2 = Slider(ph2_slide, 'Second', valmin= -1e4, valmax = 1e4, 
#                       valinit=0, valstep=10)
#     b = Button(but, 'Print Phases')
#     # Updating the plot
#     def update(val):
#         y = proc.phase(data,[ph0.val,ph1.val,ph2.val,off.val])
#         main_ax.clear()
#         xplot.clear()
#         yplot.clear()
#         main_ax.contour(np.real(y),12,cmap='jet')
#         xplot.plot(np.flipud(np.sum(np.real(y),0)),'k') #sum
#         yplot.plot(np.real(np.sum(y,1)),np.arange((len(np.sum(y,1)))),'k') #sum
#         fig.canvas.draw()
        
#     def press(val):
#         print('[%d, %d, %d, %d]' % (ph0.val,ph1.val,ph2.val,off.val))
    
#     # Calling the function "update" when the value of the slider is changed
#     ph0.on_changed(update)
#     ph1.on_changed(update)
#     ph2.on_changed(update)
#     off.on_changed(update)
#     b.on_clicked(press)
#     plt.show(block=True)


proc.mphase(data,fine = 100)
#sys.exit()

#x = np.linspace(0, 10, 30)
#y = np.sin(0.5*x)*np.sin(x*np.random.randn(30))
# Spline interpolation
#spline = UnivariateSpline(x, y, s = 6)
#x_spline = np.linspace(0, 10, 1000)
#y_spline = spline(x_spline)
# Plotting
# fig = plt.figure()
# plt.subplots_adjust(bottom=0.25)
# ax = fig.subplots()
# p, = ax.plot(np.real(data),'k')
# #p, = ax.plot(x_spline, y_spline, 'g')

# # Defining the Slider button
# # xposition, yposition, width and height
# ph0_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
# ph1_slide = plt.axes([0.25, 0.05, 0.65, 0.03])
# # Properties of the slider
# ph0 = Slider(ph0_slide, 'ph0', valmin= 0, valmax = 360, 
#                   valinit=0, valstep=0.5)
# ph1 = Slider(ph1_slide, 'ph1', valmin= -1e6, valmax = 1e6, 
#                   valinit=0, valstep=10)
# # Updating the plot
# def update(val):
#     #current_v = s_factor.val
#     y = proc.phase(data,[ph0.val,ph1.val,0,0])
#     #spline = UnivariateSpline(x, y, s = current_v)
#     #p.set_ydata(spline(x_spline))
#     p.set_ydata(np.real(y))
#     #redrawing the figure
#     fig.canvas.draw()
    
# # Calling the function "update" when the value of the slider is changed
# ph0.on_changed(update)
# ph1.on_changed(update)
#plt.show()


os.chdir(cwd)
print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))