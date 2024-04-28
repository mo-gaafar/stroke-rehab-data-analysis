
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.callbacks import TensorBoard
import numpy as np
from scripts.load_project import load_project
from enum import Enum
from config import *

# CONFIGURATION in config.py


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
    if MODEL == ModelType.KOTEV_BOTEV:
        # from scripts.classification import classify_kotev_botev
        # mne_data = classify_kotev_botev(mne_data)
        pass
    elif MODEL == ModelType.SAEID:
        # Input data
        data = mne_data
        condition = "Pre"
        # TRAIN
        left = data["Patient_2"][condition]['Train']["left"].get_data()
        right = data["Patient_2"][condition]['Train']["right"].get_data()
        X_train = np.concatenate([left, right], axis=0)
        X_train3 = X_train[:, :, :, np.newaxis]
        y_train = np.array(([0]*left.shape[0])+([1]*right.shape[0]))

        # TESTING
        left = data["Patient_2"][condition]['Test']["left"].get_data()
        right = data["Patient_2"][condition]['Test']["right"].get_data()
        # X_test = np.concatenate([left, right], axis=0)
        # y_test = np.array(([0]*left.shape[0])+([1]*right.shape[0]))
        # labels = [to_categorical(y, 2) for y in [
        #    y_train, y_test]]
        labels = to_categorical(y_train, 2)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X_train3, labels, test_size=0.8, random_state=42)

        from cnn_stft import CNN_STFT
        cnn = CNN_STFT().create_model(X_train[0].shape, 2, 0.5, False)
        TensorBoardCallback = TensorBoard(log_dir="logs")
        history = cnn.fit(X_train2, y_train2, epochs=10,
                          batch_size=32, validation_data=(X_test2, y_test2), callbacks=[TensorBoardCallback])


# Visualization
if VISUALIZATION:
    # from scripts.visualize import visualize
    # visualize(mne_data)
    pass
