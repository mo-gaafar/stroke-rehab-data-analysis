from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.callbacks import TensorBoard  # Import TensorBoard callback
import tensorflow as tf
import numpy as np
import os
mohamed
__mohamed
Online
🏠G33 Stroke Analysis

KotěvBotěv — Yesterday at 10: 47 PM
good night 🙂
hans — Yesterday at 10: 48 PM
i can hear you but i cant talk back
very good
i will restart my machine - maybe we can actually talk after that...
KotěvBotěv — Yesterday at 10: 53 PM
Crud
do you have access to the github repo?
Fatemeh — Yesterday at 11: 04 PM
Yes
Could you send me your email?
KotěvBotěv — Yesterday at 11: 06 PM
I sent you my email to your email 😄
Fatemeh — Yesterday at 11: 06 PM
Thank you😅
Would it be helpful if I send you whatever I wrote part to part so you can modify it for our data?
If it's not helpful please let me know 😅
KotěvBotěv — Yesterday at 11: 12 PM
It depends on what it is . For example it might not make sense to actually implement ICA and try to port it to python, as it is already in the MNE library: https: // mne.tools/stable/generated/mne.preprocessing.ICA.html  # mne.preprocessing.ICA
it can do all the plot things as well
what would make sense is if you coded up using this function and the plots you think are interesting
the docs page has refrences to exaples with the plots as well
but for that you would have to be able to run the code locally
but I have no idea how to use python on anything other than Linux.
Fatemeh — Yesterday at 11: 21 PM
Yeah that makes more sense. But I'm so lost right now😅 my mind went to sleep an hour ago. I will try it again tomorrow 😅✌️
Good night everyone 😊👋
KotěvBotěv — Yesterday at 11: 21 PM
alright 🙂 gn
hans — Yesterday at 11: 22 PM
gn
hans — Yesterday at 11: 31 PM
i am here
trying to get this stuff into google colab
so that i can mess around with pretrained models etc... but running into probs
a short search into best models for time-series is also on the todo list...
thats a question to answer AFTER i have colab running ^ ^
hans — Yesterday at 11: 47 PM
yeah
that was quick...
np
hans — Today at 12: 08 AM
finally runs in colab
yep
hans — Today at 12: 35 AM
i think i am out of energy
since i dont contribute in any significant way, i will go to sleep.
see you tomorrow
mohamed — Today at 5: 39 AM
hey guys
amr.mohamed — Today at 6: 39 AM
heeey ya moe
mohamed — Today at 6: 40 AM
khaleehom yeb2o yedkholo lamma yes7o
amr.mohamed — Today at 6: 40 AM
tamam
mohamed — Today at 6: 42 AM
https: // prod.liveshare.vsengsaas.visualstudio.com/join?D142A70F7954BA6CAFB43D0BB30BD1E43BE3
Visual Studio Code for the Web
Build with Visual Studio Code, anywhere, anytime, entirely in your browser.
amr.mohamed — Today at 6: 43 AM
mesh 3aref akhosh 3aleih
mohamed — Today at 6: 44 AM
leh keda
saeid alipour — Today at 6: 44 AM
hey guys🤚
amr.mohamed — Today at 6: 44 AM
hey saeid 🙂
mohamed — Today at 6: 45 AM
restarting session
amr.mohamed — Today at 6: 45 AM
sanya hagarab haga
amr.mohamed — Today at 6: 45 AM
tamam
mohamed — Today at 6: 45 AM
https: // prod.liveshare.vsengsaas.visualstudio.com/join?51E990A39D3BE65ABC91FF205BDE04542678
Visual Studio Code for the Web
Build with Visual Studio Code, anywhere, anytime, entirely in your browser.
Fatemeh — Today at 6: 46 AM
Hey guys
mohamed — Today at 6: 46 AM
saeid you'll wake them from their sleep 😂
hello
saeid alipour — Today at 6: 46 AM
😁
amr.mohamed — Today at 6: 47 AM
kolo tamam
mohamed — Today at 6: 48 AM
Image
nice spectrogram
saeid alipour — Today at 6: 49 AM
nice, we can see filtering, how is the frequency more than 50?
mohamed — Today at 6: 50 AM
I think its aliasing?
not sure
maybe its not filtered yet
saeid alipour — Today at 6: 51 AM
below than 10 hz, is almost zero i think in the figure
according to stroke rehab, if i'm true we must have energy between 8 to 30 hz
mohamed — Today at 6: 54 AM
ok ill try to filter
saeid alipour — Today at 6: 54 AM
no need to filter
they filtered before
mohamed — Today at 6: 55 AM
Image
8, 50 bandpass
saeid alipour — Today at 6: 56 AM
yeah it's according to what they said 👌
last night henzo was working on classifying
saeid alipour — Today at 6: 57 AM
do we have its results?
mohamed — Today at 6: 58 AM
i tried running what he pushed
there are issues with the pooling layer
reducing the dimensions too much
saeid alipour — Today at 6: 59 AM
i think we must reduce the filter, did you try that?
mohamed — Today at 7: 00 AM
Image
mohamed — Today at 7: 00 AM
no
what do you mean exactly
saeid alipour — Today at 7: 01 AM
filter in our cnn model
mohamed — Today at 7: 02 AM
Image
saeid alipour — Today at 7: 03 AM
yeah these filters
did you reduce these?
mohamed — Today at 7: 03 AM
which filter do we reduce
saeid alipour — Today at 7: 04 AM
they are multiple of 2
we must reduce all of them until we get results
256 become 128
mohamed — Today at 7: 04 AM
ok
saeid alipour — Today at 7: 05 AM
can you share screen Mohammed?
mohamed — Today at 7: 08 AM
any idea?
saeid alipour — Today at 7: 08 AM
its dimension error not filter
mohamed — Today at 7: 08 AM
yes
saeid alipour — Today at 7: 08 AM
let me see our input
mohamed — Today at 7: 12 AM
https: // prod.liveshare.vsengsaas.visualstudio.com/join?51E990A39D3BE65ABC91FF205BDE04542678
Visual Studio Code for the Web
Build with Visual Studio Code, anywhere, anytime, entirely in your browser.
saeid alipour — Today at 7: 21 AM
Image
Mohammed do you know why i can't see the files?
mohamed — Today at 7: 22 AM
notebooks dont work on browser
you would have to install vscode
saeid alipour — Today at 7: 23 AM
aha ok
Fatemeh — Today at 7: 23 AM
It worked for me last night but when you went for sleep I   got kicked out too
saeid alipour — Today at 7: 23 AM


class MODEL:
    def train_model(self, model_data):
        folder_path = model_data['path']
        model_name = model_data['name']
        model_function = model_data['function']
        all_dataset = model_data['data']
        num_epoch = model_data['num_epoch']
        # num_sub=model_data['num_sub']
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


Collapse
run_model.py
3 KB


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
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Conv Block 4
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
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
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
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


... (1 line left)
Collapse
cnn_stft.py
4 KB
mohamed — Today at 7: 24 AM
sorry 😂
you can rejoin now
no kicking out this time
Fatemeh — Today at 7: 25 AM
No it's okay😂 I'm just saying that it works on browser too. Probably you just need to give Saeid access or sth
Fatemeh — Today at 7: 26 AM
😂😂
﻿


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
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(dropout))

        # Conv Block 4
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
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
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
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


