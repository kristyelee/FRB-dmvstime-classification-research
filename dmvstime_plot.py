import numpy  as np
import sys
import copy
import matplotlib.pyplot as plt

def create_dmvstime_array(data):
    nbinlim=256
    dmvstm_array = []
    width = 1
    band = (data.freqs.max()-data.freqs.min())
    centFreq = (data.freqs.min()+band/2.0)/(10**3) # To get it in GHz
    dm = data.dm

    #This comes from Cordes and McLaughlin (2003) Equation 13.
    FWHM_DM = 506*float(width)*pow(centFreq,3)/band

    #The candidate DM might not be exact so using a longer range
    FWHM_DM = 3*FWHM_DM

    lodm = dm-FWHM_DM
    if lodm < 0:
        lodm = 0
        hidm = 2*dm # If low DM is zero then range should be 0 to 2*DM
    else:
        hidm= dm+FWHM_DM

    dmstep = (hidm-lodm)/48.0
    datacopy = copy.deepcopy(data)

    #Catches invalid data
    if (dmstep == 0.0 and hidm == 0 and lodm == 0):
        return dmvstm_array

    #Create dmvstime array
    for ii in np.arange(lodm,hidm,dmstep):
        #Without this, dispersion delay with smaller DM step does not produce delay close to bin width
        data.dedisperse(0,padval='rotate')
        data.dedisperse(ii,padval='rotate')
        Data = np.array(data.data[..., :nbinlim])
        Dedisp_ts = Data.sum(axis=0)
        dmvstm_array.append(Dedisp_ts)

    return np.array(dmvstm_array)

if __name__=='__main__':
    # Command Line Generation of Spectra Object
    spectras = np.load(sys.argv[1])
    collect_all_data = sys.argv[2] # 0 for no iteration through entire set, 1 for iteration
    index = sys.argv[3]
    data = spectras[index]
    dmvstime_array = create_dmvstime_array(data)
    print(dmvstime_array)
    plt.imshow(dmvstm_array, aspect='auto')
    plt.show()

    #Create dmvstime array for each spectra object
    if (collect_all_data == 1):
        dmvstime_array = []
        for data in spectras:
            #print(1)
            dmvstime_array.append(create_dmvstime_array(data))
