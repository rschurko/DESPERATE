# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'/Users/SRG/Documents/Adam/Python/SSNMR/functions')
import numpy as np
import functions as proc
import urQRd1 as proc2
import simpson as simproc
import matplotlib.pyplot as plt
import time
start_time = time.time()


T2 = 0.03 #fake T2 global, unitless
fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='yes')

ph = [558.67, 92160.09, 0, 0]

gb = 10 ##Amount of global gaussian broadening for coadd

fid, spec = simproc.coadd(fid, 512, 50)
fidref = proc.gauss( fid , lb = gb )


#specref = proc.phase(proc.fft(fidref),ph)[0,:]


fid,SW = simproc.read('Sn_CPMG_id.fid', lb=T2, plot='no') #dup for coadd
fid = simproc.noise(fid,0.2)
fid, spec = simproc.coadd(fid, 512, 50)                     #dup for coadd

spec = proc.phase(spec,ph)[0,:]
snrpin = simproc.snrp(spec,1446,3249)

#spec = proc.fft(fid)

#fidrecon = simproc.dwt(fid,SW,ph,thresh=0.0,lb=0)

fidrecon = proc2.urQRd(fid, 115)  ##Fast Cadzow
#fidrecon = simproc.cadzow(fid,ph,SW,lb = gb) ##Slow Cadzow
specrecon = proc.phase(proc.fft(fidrecon),ph)[0,:]
snrpout = simproc.snrp(specrecon,1446,3249)


simproc.deplot(fid,fidrecon,SW,ph)

#plt.figure(3)
#simproc.residual(fidref,fidrecon,SW,ph,plot='yes')

#specrecon = simproc.coadd(fidrecon,512,50)
#plt.plot(np.abs(specrecon))

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))