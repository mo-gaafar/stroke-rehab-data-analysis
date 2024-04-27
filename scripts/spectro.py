import pywt
import numpy as np
import mne


def cwt(X):
    '''Compute the continuous wavelet transform of the input data.'''
    
    # epochs, ch, t
    coefs, freqs = pywt.cwt(X, np.geomspace(
        1, 70, num=64), wavelet='morl')  # TODO : try morl or cmor1.5-1.0
    # freqs, epochs, ch, t
    xtw = np.transpose(coefs, [1, 2, 0, 3])
    # xtw[:,:] = np.abs(xtw[:,:,:-1, :-1])
    xtw = np.abs(xtw)
    return (xtw)

def plot_pseudospectrogram(epochs: mne.Epochs):
    X = epochs.get_data()
    xtw = cwt(X)

    axes = plt.subplot(111)
    #
    print(xwt.shape)
    axes.pcolormesh(np.arange(xwt.shape[-1]), np.geomspace(1, 70, num=64), xwt[5,10], shading='nearest')
    # Set yscale, ylim and labels
    plt.yscale('log')
    plt.ylim([1, 70])
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.show()


def playground(epochs: mne.Epochs):
    
    pass