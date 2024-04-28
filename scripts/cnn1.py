from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.losses import CategoricalCrossentropy
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.api.callbacks import TensorBoard  # Import TensorBoard callback

import os
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt


def train(X_train, y_train, X_val, y_val):
    '''
    Expects (batch, chan, time, freq) TODO freq, time?
    '''
    model_data = {'data': {}}
    model_data['data']['x_train'] = X_train
    model_data['data']['y_train'] = y_train
    model_data['data']['x_val'] = X_val
    model_data['data']['y_val'] = y_val

    model_trainer = ModelTrainer()
    model_class = CNN_STFT
    model_data['function'] = model_class
    model, history = model_trainer.train_model(model_data=model_data)
    return model


class ModelTrainer:
    def train_model(self, model_data):
        folder_path = model_data['path'] if 'path' in model_data else './'
        model_name = model_data['name'] if 'name' in model_data else (
            model_data['function'].__class__.__name__)
        model_function = model_data['function']
        all_dataset = model_data['data']
        num_epoch = model_data['num_epoch'] if 'num_epoch' in model_data else 100
        # num_sub=model_data['num_sub']
        dropout = model_data['dropout'] if 'dropout' in model_data else 0
        sys.path.append(folder_path)
        # module = __import__(model_name)  # Dynamically import the module
        # model_function = getattr(module, model_function)  # Get the model function/class

        # sub=num_sub

        data = all_dataset

        cnn_stft_instance = model_function()  # Instantiate the class

        # num_class =len(data['y_train'][0])
        num_class = 2
        data_shape = data['x_train'][0].shape

        model = cnn_stft_instance.create_model(
            data_shape, num_class, dropout, print_summary=True)
        logdir = 'logs'
        tensorboard_callback = TensorBoard(log_dir=logdir)

        history = model.fit(data['x_train'], data['y_train'], batch_size=64, epochs=num_epoch, validation_data=(
            data['x_val'], data['y_val']), callbacks=[tensorboard_callback])

        # Save the trained model
        # save_path = os.path.join(folder_path, f'imageclassifier_{sub}.h5')
        save_path = os.path.join(folder_path, 'imageclassifier.h5')

        model.save(save_path)

        # plot loss
        fig = plt.figure()
        plt.plot(history.history['loss'], color='teal', label='loss')
        plt.plot(history.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        # plot acc
        fig = plt.figure()
        plt.plot(history.history['accuracy'], color='teal', label='accuracy')
        plt.plot(history.history['val_accuracy'],
                 color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        self.model = model
        return model, history


class CNN_STFT:

    def create_model(self, input_shape, classes, dropout, print_summary=False):

        # Basis of the CNN_STFT is a Sequential network
        model = Sequential()

        # Conv Block 1
        model.add(Conv2D(filters=128, kernel_size=(3, 3),
                  input_shape=input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Conv Block 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Conv Block 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Conv Block 4
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Conv Block 5
        # model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Conv Block 6
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Flatten layer
        model.add(Flatten())

        # Dense layer
        model.add(Dense(128))
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
