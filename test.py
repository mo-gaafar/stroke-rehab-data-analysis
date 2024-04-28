import pickle
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from scripts.load_project import load_project
# mne.viz.set_3d_backend("notebook")
mne_data = load_project()

patient_epoch = mne_data["Patient_1"]["Pre"]["Test"]  # one patient
epoch = patient_epoch["left"]  # one epoch
epoch_ica = epoch.copy()
# 3. Initialize and fit ICA
ica = ICA(n_components=16, random_state=9)
ica.fit(epoch_ica)  # TODO: should be only on training

# 4. Identify artifact components (visualize and/or use detection methods)
# ica.plot_components()  # Visualize components
ica.plot_sources(epoch_ica, show_scrollbars=True)
# ica.plot_properties(epoch_ica, picks=range(0, 16), psd_args={
#                     'fmax': 35.}, image_args={'sigma': 1.})


# 5. Exclude artifact components
# ... Determine components to exclude (e.g., components 0 and 2)
ica.exclude = [0, 12]
# TODO: to generalize on different subjects we can try using template matching
plt.show(block=True)

# 6. Reconstruct the EEG signal
epoch = ica.apply(epoch_ica)
'''
Selecting ICA components using template matching
When dealing with multiple subjects, it is also possible to manually select an IC for exclusion on one subject, 
and then use that component as a template for selecting which ICs to exclude from other subjects’ data, 
using mne.preprocessing.corrmap.
The idea behind corrmap is that the artifact patterns are similar enough across subjects that corresponding ICs can be identified 
by correlating the ICs from each ICA solution with a common template, 
and picking the ICs with the highest correlation strength. corrmap takes a list of ICA solutions, 
and a template parameter that specifies which ICA object and which component within it to use as a template.
'''

# select components using correlation with template
template = ica.exclude
ica.exclude = []


# ica.plot_forward()
# epoch.plot_sources()

epoch_clean = epoch.copy()

# this should contain the ica channels that have been excluded before, 
# pickle it in "data/ica_excluded.pkl to save processing time"

ICA_EXCLUDED = "data/ica_excluded.pkl"


def ica_cleaning_and_extraction(epoch, template=None):
    '''
    epoch: mne.Epochs


    Returns: cleaned epoch, ica object'''
    # 3. Initialize and fit ICA
    ica = ICA(n_components=16, random_state=9)
    ica.fit(epoch)  # ? TODO: should be only on training

    # 4. Identify artifact components (visualize and/or use detection methods)
    # ica.plot_components()  # Visualize components
    ica.plot_sources(epoch, show_scrollbars=True)
