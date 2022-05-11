import sys
import os
sys.path.insert(0,'/Users/SRG/Documents/GitHub/SSNMR/functions')
import numpy as np
import functions as proc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import nmrglue.fileio.bruker as br
start_time = time.time()

cwd =  os.getcwd()

fid = np.zeros((2048,), dtype='complex64')
for j in range(100,108):
    dic, data = br.read(cwd + '\\' + str(j))
    fid += data

# sys.exit()

#Save the spectrum
# spec = np.flip(spec)
# dic, data = br.read(cwd+ '\\' + '8') #load FID
br.write(cwd+ '\\' + '108', dic, fid, pdata_folder=1, overwrite=True)
br.write_pdata(cwd+ '\\' + '108', dic, proc.fft(fid), pdata_folder=1, write_procs=True, overwrite=True)

os.chdir( cwd )


print('Finished!')
print("-- %5.5f s Run Time --" % (time.time() - start_time))