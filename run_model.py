import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard  # Import TensorBoard callback
import sys
import matplotlib.pyplot as plt


class MODEL:
    def train_model(self, model_data):
        folder_path = model_data['path']
        model_name = model_data['name']
        model_function = model_data['function']
        all_dataset = model_data['data']
        num_epoch = model_data['num_epoch']
        dropout = model_data['dropout']
        sys.path.append(folder_path)
        module = __import__(model_name)  # Dynamically import the module
        # Get the model function/class
        model_function = getattr(module, model_function)

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
