import sys
import os
sys.path.insert(0,'U:/Github/SSNMR/functions')
sys.path.insert(0,'C:\code\ARA')
import numpy as np
import functions as proc
import simpson as simp
# import urQRd1 as proc2
import time
import matplotlib.pyplot as plt
import subprocess
from subprocess import Popen
from tabulate import tabulate
start_time = time.time()

path = os.getcwd()

##NOTE: A good tensor model is from https://doi.org/10.1063/1.431418

name = '13C_DD.tcl'
# name = '13C_DD_ext.tcl'

exp = np.load('exp_data.npy') #exp field data

B0 = [1.16, 2.0, 2.9, 4, 5.48]
off = [-0.5e3, 0.6e3, 3.5e3, 1.9e3, 0] #compensate from field error?
# B0 = [1.16, 2.0, 2.9, 4, 5.2] #adjust field offset

diso0 = 0.0
diso = 0.0

aniso = -300
eta = 0.9 #griffin 0.67 ; eta = 1.0 yeilds Pake doublet! <1.0 yeilds asym.

DD = -2890

alpha = 8 #griffin 20; 6 looks good
beta = 90

lb = 17
zf = 2048
##
1/0
spec = []
k = 0
for j,i in enumerate(B0):

        g = open(name, mode='r')
        lines=g.readlines()
        s = lines[3].split()
        s1 = lines[4].split()
        d = lines[5].split()
        b = lines[15].split()
        o = lines[28].split()
        # sys.exit()
        
        s[2] = str(diso0)+'p'
        s[3] = str(aniso)+'p'
        s[4] = str(eta)
        s[5] = str(alpha)
        s[-1] = '0\n'
        
        s1[2] = str(diso)+'p'
        s1[3] = str(aniso)+'p'
        s1[4] = str(eta)
        s1[5] = str(-alpha)
        s1[-1] = '0\n'
        
        d[3] = str(DD)
        d[5] = str(beta)
        d[6] = str(beta)+'\n'
        
        b[1] = str(i*42.58)+'e6\n'
        
        o[1] = str(off[j])+'\n'
        
        lines[3] = ' '.join(s)
        lines[4] = ' '.join(s1)
        lines[5] = ' '.join(d)
        lines[15] = ' '.join(b)
        lines[28] = ' '.join(o)
        
        h = open(name,mode='w')
        h.writelines(lines)
        h.close()
        g.close()
        
        
        ##runs Simpson in powershell
        
        # p = Popen(f'python p00_GageController_v13.py {radarParamFile}' ##try this for simpson
        p = Popen('simpson 13C_DD.tcl' ##try this for simpson
                  #,shell=True
                  #,stdout=subprocess.PIPE
                  ,bufsize=1
                  #,stderr=subprocess.PIPE
                  ,universal_newlines=True)

        while p.poll() is None:
            time.sleep(0.4)

        
        # os.system(f'start cmd /K python p00_GageController_v13.py ./radarRecorderParams.json')
        
        # os.system(f'start cmd /K simpson p00_GageController_v13.py ./radarRecorderParams.json') ##write for simpson

        # d = subprocess.run( ["powershell", "-Command", "simpson "+str(name) ] , capture_output=True)
        
        # while p.poll is not None:
        #     time.sleep(0.2)
            
        # print(d.stderr)
        # print(d.stdout)
        # d.terminate() #try to kill
        # sys.exit()
        
        fid, SW = simp.read2('13C_DD.fid')
        fid = proc.gauss(fid,lb,0)
        
        # spec = proc.fft(fid, zf)
        spec = proc.fft(fid, zf).imag
        spec = spec / max(spec)
        # spec.append( proc.fft(fid, len(fid)*8) )
        # spec = np.flip(spec)

        # plt.subplot(3,3,k)
        k +=1
        print(k)
        f = np.linspace(-SW/2, SW/2, spec.shape[0])
        # plt.subplot(5,1,k)
        plt.plot(f*1e-3, spec+k*1, 'c')
        plt.plot(f*1e-3, exp[k-1,:].real*1.0 + k*1, 'k')
        plt.xlim([-40, 40])
        plt.xlabel('Frequency (kHz)')

os.chdir(path)

##Table SNR peak-peak
data= [["%.1f"%DD,  "%.1f"%diso, "%.1f"%aniso, "%.2f"%eta, "%d"%alpha ,"%d"%beta ]]
# create header
head = ['d (Hz)', '\u03B4iso (ppm)', '\u03B4 (ppm)', '\u03B7', '\u03B1', '\u03B2']
# display table
print(tabulate(data, headers=head, tablefmt="pretty", floatfmt="5.1f"))
#generate ns counter at end

print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))