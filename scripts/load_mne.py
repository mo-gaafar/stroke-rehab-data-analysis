import numpy as np
from mne.io import RawArray
from mne import create_info
import mne

def mne_load_data(data_dict):
    # Load into MNE format

    # for every patient, pre and post, train and test
    for patient in data_dict.keys():
        for session in data_dict[patient].keys():
            for task in data_dict[patient][session].keys():
                eeg = data_dict[patient][session][task]['y'].T
                sfreq = data_dict[patient][session][task]['fs'].squeeze().item()
                triggers = data_dict[patient][session][task]['trig']

                print(f"Patient: {patient}, Session: {session}, Task: {task}")
                print(
                    f"EEG data shape (channels, samples): {eeg.shape}")
                print(
                    f"Trigger data shape: {data_dict[patient][session][task]['trig'].shape}")

                # Create info object
                info = create_info(ch_names=[f"EEG {i}" for i in range(eeg.shape[0])],
                                sfreq=sfreq)

                # Create RawArray object
                raw = RawArray(eeg, info)

                # Create events array
                # events_raw = data_dict[patient][session][task]['trig'].reshape(
                #     -1, 1)
                # events = np.array([[i, 0, int(events_raw[i])]
                #                 for i in range(events_raw.shape[0])])
                triggerd = np.zeros_like(triggers, dtype=int)
                triggerd[1:] = ((triggers[1:]-triggers[:-1])!=0) * (triggers[1:]!=0)
                triggerd[triggerd!=0] = triggers[triggerd!=0]
                events = np.column_stack((np.argwhere(triggerd)[:,0], np.zeros(sum(triggerd!=0), dtype=int),triggerd[triggerd!=0]))

                # Mapping of events to ids
                event_dict = {
                    "left" : 1,
                    "right" : -1,
                }

                # divide the eeg into epochs, each epoch is 8 seconds long (trigger is at 2 seconds, at 3.5 seconds )
                # One session was composed by 240 MI repetitions on both hands, divided in 3 runs of 80 trials each.

                # Create epochs
                print("Creating epochs...")
                print(events.shape)
                epochs = mne.Epochs(raw, events, tmin=0, tmax=8,
                                    baseline=None, preload=True, event_id=event_dict)

                # Separate epochs for left and right hand MI
                # left_epochs = epochs[data_dict[patient]
                #                     [session][task]['trig'] == 1]
                # right_epochs = epochs[data_dict[patient]
                #                     [session][task]['trig'] == -1]
                left_epochs = epochs["left"]
                right_epochs = epochs["right"]

                print(f"Left hand MI epochs: {left_epochs.get_data().shape}")
                print(f"Right hand MI epochs: {right_epochs.get_data().shape}")

                # Save epochs
                left_epochs.save(
                    f"data/stroke/{patient}_{session}_{task}_left-epo.fif")
                right_epochs.save(
                    f"data/stroke/{patient}_{session}_{task}_right-epo.fif")

                print("\n")

    return 
