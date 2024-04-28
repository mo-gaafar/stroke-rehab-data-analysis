# ALL CONFIGURATION CONSTANTS ARE HERE import them in the modules

from enum import Enum


class ModelType(Enum):
    KOTEV_BOTEV = 1
    SAEID = 2


PREPROCESSING = True
ICA_CLEANING = False
FEATURE_EXTRACTION = False
CLASSIFICATION = True
VISUALIZATION = False
MODEL = ModelType.SAEID

DATA_LOADING_DIR = "data/stroke"
ICA_EXCLUDED_PATH = "data/ica_excluded.pkl"
