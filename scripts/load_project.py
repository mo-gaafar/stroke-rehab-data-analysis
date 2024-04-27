import numpy as np
from scipy.io import loadmat
from mne.io import RawArray
from mne import create_info
import mne


def load_project():
    # Load the data
    data_dict = {
        "Patient_1": {
            "Pre": {
                "Train": loadmat('data/stroke/P1_pre_training.mat'),
                "Test": loadmat('data/stroke/P1_pre_test.mat')
            },
            "Post": {
                "Train": loadmat('data/stroke/P1_post_training.mat'),
                "Test": loadmat('data/stroke/P1_post_test.mat')
            }
        },
        "Patient_2": {
            "Pre": {
                "Train": loadmat('data/stroke/P2_pre_training.mat'),
                "Test": loadmat('data/stroke/P2_pre_test.mat')
            },
            "Post": {
                "Train": loadmat('data/stroke/P2_post_training.mat'),
                "Test": loadmat('data/stroke/P2_post_test.mat')
            }
        },
        "Patient_3": {
            "Pre": {
                "Train": loadmat('data/stroke/P3_pre_training.mat'),
                "Test": loadmat('data/stroke/P3_pre_test.mat')
            },
            "Post": {
                "Train": loadmat('data/stroke/P3_post_training.mat'),
                "Test": loadmat('data/stroke/P3_post_test.mat')
            }
        },
    }

    # Load into MNE format
    from .load_mne import mne_load_data
    mne_data_dict = mne_load_data(data_dict)

    return mne_data_dict
