import pickle as pkl
import pickle
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from scripts.load_project import load_project
from config import *



def init_cleaining():
    '''
    Initialize ICA cleaning logic
    '''

    # 1. Load the dataset
    mne_data = load_project()

    patient_epoch = mne_data["Patient_1"]["Pre"]["Test"]  # one patient
    epoch = patient_epoch["left"]  # one epoch
    epoch_ica = epoch.copy()

    # 3. Initialize and fit ICA
    ica = ICA(n_components=16, random_state=9)
    ica.fit(epoch_ica)  # TODO: should be only on training

    if VISUALIZATION:
        # 4. Identify artifact components (visualize and/or use detection methods)
        ica.plot_components()  # Visualize components
        # ica.plot_sources(epoch_ica, show_scrollbars=True)
        ica.plot_properties(epoch_ica, picks=range(0, 16), psd_args={
                            'fmax': 35.}, image_args={'sigma': 1.})
        plt.show(block=True)

    # 5. Exclude artifact components
    # ... Determine components to exclude (e.g., components 0 and 2)
    ica.exclude = [0, 2, 12]

    # 6. Reconstruct the EEG signal
    epoch = ica.apply(epoch_ica)

    # ica.plot_forward()
    # epoch.plot_sources()

    epoch_clean = epoch.copy()

    # this should contain the ica channels that have been excluded before,
    # pickle it in "data/ica_excluded.pkl to save processing time"

    with open("data/ica_excluded.pkl", "wb") as f:
        pkl.dump(ica.exclude, f)


# TODO: to generalize on different subjects we can try using template matching
def ica_cleaning_and_extraction(epoch, template_pkl_path=ICA_EXCLUDED):
    '''
    epoch: mne.Epochs
    Returns: cleaned epoch, ica object
    Using template matching to exclude ICA components
    '''
    template_matching = True
    if template_pkl_path:
        try:
            with open(template_pkl_path, "rb") as f:
                ica.exclude = pkl.load(f)
        except:
            template_matching = False
            print("Template file not found, skipping template matching")

    ica = ICA(n_components=16, random_state=9)


    if template_matching:
        # We now supposedly have the ica.exclude template, we can match it with the current epoch
        # and exclude the same components using corrmap
        input_ica = ica.fit(epoch)
        corr = ica.compute_scores(epoch)
        
        



    ica.fit(epoch)
    epoch_clean = epoch.copy()
    epoch_clean = ica.apply(epoch_clean)

    return epoch_clean, ica
