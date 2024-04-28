from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, Conv2D, MaxPooling1D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras._tf_keras.keras.losses import CategoricalCrossentropy
import tensorflow as tf


class CNN_STFT:

    def create_model(self, input_shape, classes, dropout, print_summary=False):

        # Basis of the CNN_STFT is a Sequential network
        model = Sequential()

        # Conv Block 1
        model.add(Conv2D(filters=4, kernel_size=3,
                  input_shape=(16, 1153, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Dropout(dropout))

        # Conv Block 2
        model.add(Conv2D(filters=4, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Dropout(dropout))

        # Conv Block 3
        model.add(Conv2D(filters=4, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Dropout(dropout))

        # Conv Block 4
        model.add(Conv2D(filters=4, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Dropout(dropout))

        # Conv Block 5
        # model.add(Conv2D(filters=16, kernel_size=5, padding='same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=2))
        # model.add(Dropout(dropout))

        # Conv Block 6
        model.add(Conv2D(filters=4, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Dropout(dropout))

        # Flatten layer
        model.add(Flatten())

        # Dense layer
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        # model.add(Dense(128))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout))

        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout))

        # model.add(Dense(32))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout))

        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout))

        # model.add(Dense(8))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout))

        # model.add(Dense(4))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        if print_summary:
            model.summary()

        # Compile the model
        model.compile(loss=CategoricalCrossentropy(),
                      optimizer='adam',
                      metrics=['accuracy'])

        # Assign model and return
        self.model = model
        return model
