import mne


def preprocess(epochs: mne.Epochs):
    
    # Bandpass Filtering
    # epochs.filter(1, 40)

    
    return epochs