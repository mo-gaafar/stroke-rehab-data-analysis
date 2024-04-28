
from scripts.load_project import load_project
from enum import Enum
from config import *


class ModelType(Enum):
    KOTEV_BOTEV = 1
    SAEID = 2


# Load data
mne_data = load_project()

# Preprocessing
if PREPROCESSING:
    from scripts.preprocess import preprocess_all
    mne_data = preprocess_all(mne_data)

# ICA cleaning
if ICA_CLEANING:
    from scripts.artifact_removal import ica_cleaning_and_extraction
    mne_data = ica_cleaning_and_extraction(mne_data)

# Feature extraction
if FEATURE_EXTRACTION:
    from scripts.feature_extraction import extract_features
    mne_data = extract_features(mne_data)

# Classification
if CLASSIFICATION:
    if ModelType.KOTEV_BOTEV:
        # from scripts.classification import classify_kotev_botev
        # mne_data = classify_kotev_botev(mne_data)
        pass
    # elif ModelType.SAEID:
        # from scripts.classification import classify
        # mne_data = classify(mne_data)

# Visualization
# if VISUALIZATION:
#     from scripts.visualize import visualize
#     visualize(mne_data)
