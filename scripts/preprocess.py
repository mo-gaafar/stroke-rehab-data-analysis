import mne

from .artifact_removal import remove_artifacts


def filter_epoch(epochs: mne.Epochs):
    # Apply bandpass filter
    epochs.filter(l_freq=8, h_freq=50)
    return epochs


def preprocess(epochs: mne.Epochs):

    # Apply bandpass filter
    epochs = filter_epoch(epochs)
    # Remove artifacts
    epochs = remove_artifacts(epochs)

    # ?  Find bad epochs and remove them using mne.find_bads
    bad_epochs = []
    return epochs


def preprocess_all(dict_epochs: dict):
    '''
    Applies to test and non test data'''
    from .util import apply_to_dict

    dict_epochs = apply_to_dict(dict_epochs, preprocess)

    return dict_epochs
