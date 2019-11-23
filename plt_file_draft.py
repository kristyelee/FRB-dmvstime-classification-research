import copy
import filterbank
import imp
import numpy as np
from matplotlib import pyplot as plt

wt = imp.load_source('waterfaller', '/home/vgajjar/SpS/sps/src/waterfaller/waterfaller.py')
rawdatafile = filterbank.FilterbankFile("/datax/scratch/vgajjar/Test_pipeline/fake.fil")

spectra_data, bins, nbins, start_time = wt.waterfall(rawdatafile, 4.4, 0.3, 0, 1024, 128)
plt.imshow(spectra_data.data, aspect='auto')
plt.show()

print(spectra_data.data)
lodm, hidm, dmstep = 600, 800, 1
dmvstm_array = []
print(nbins)
datacopy = copy.deepcopy(spectra_data)
for ii in np.arange(lodm,hidm,dmstep):
	spectra_data.dedisperse(0,padval='rotate')
	spectra_data.dedisperse(ii,padval='rotate')
	Data = np.array(spectra_data.data[..., :nbins])
	Dedisp_ts = Data.sum(axis=0)
	dmvstm_array.append(Dedisp_ts)

dmvstm_array=np.array(dmvstm_array)
plt.imshow(dmvstm_array, aspect='auto')
plt.show()
plt.imshow(spectra_data.data, aspect='auto')
plt.show()
