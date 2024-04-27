import numpy as np
from mne.io import RawArray
from mne import create_info
import mne


def mne_load_data(data_dict):
    # Load into MNE format

    mne_data_dict = {}
    # for every patient, pre and post, train and test
    for patient in data_dict.keys():
        mne_data_dict[patient] = {
            "Pre": {},
            "Post": {},
        }
        for session in data_dict[patient].keys():
            mne_data_dict[patient][session] = {
                "Train": {},
                "Test": {},
            }
            for task in data_dict[patient][session].keys():
                mne_data_dict[patient][session][task] = {
                    "left": None,
                    "right": None,
                }
                eeg = data_dict[patient][session][task]['y'].T
                sfreq = data_dict[patient][session][task]['fs'].squeeze(
                ).item()
                triggers = data_dict[patient][session][task]['trig']

                print(f"Patient: {patient}, Session: {session}, Task: {task}")
                print(
                    f"EEG data shape (channels, samples): {eeg.shape}")
                print(
                    f"Trigger data shape: {data_dict[patient][session][task]['trig'].shape}")

                # Create info object
                channel_map = {
                    1: "FC3", 2: "FCz", 3: "FC4",
                    4: "C5", 5: "C3", 6: "C1", 7: "Cz", 8: "C2", 9: "C4", 10: "C6",
                    11: "CP3", 12: "CP1", 13: "CPz", 14: "CP2", 15: "CP4",
                    16: "Pz",
                }
                info = create_info(ch_names=[channel_map[i+1] for i in range(eeg.shape[0])],
                                   sfreq=sfreq, ch_types=['eeg']*eeg.shape[0])

                # Create RawArray object
                raw = RawArray(eeg, info)
                raw.set_montage(
                    mne.channels.make_standard_montage('standard_1020'))

                # Create events array
                # events_raw = data_dict[patient][session][task]['trig'].reshape(
                #     -1, 1)
                # events = np.array([[i, 0, int(events_raw[i])]
                #                 for i in range(events_raw.shape[0])])
                triggerd = np.zeros_like(triggers, dtype=int)
                triggerd[1:] = ((triggers[1:]-triggers[:-1])
                                != 0) * (triggers[1:] != 0)
                triggerd[triggerd != 0] = triggers[triggerd != 0]
                events = np.column_stack((np.argwhere(triggerd)[:, 0], np.zeros(
                    sum(triggerd != 0), dtype=int), triggerd[triggerd != 0]))

                # Mapping of events to ids
                event_dict = {
                    "left": 1,
                    "right": -1,
                }

                # divide the eeg into epochs, each epoch is 8 seconds long (trigger is at 2 seconds, at 3.5 seconds )
                # One session was composed by 240 MI repetitions on both hands, divided in 3 runs of 80 trials each.

                # Create epochs
                print("Creating epochs...")
                print(events.shape)
                # Note: zero of the epoch is the trigger time
                epochs = mne.Epochs(raw, events, tmin=-0.5, tmax=6,
                                    baseline=(-0.5, -0.2), preload=True, event_id=event_dict)

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
                    f"data/stroke/{patient}_{session}_{task}_left-epo.fif", overwrite=True)
                right_epochs.save(
                    f"data/stroke/{patient}_{session}_{task}_right-epo.fif", overwrite=True)

                print("\n")

                mne_data_dict[patient][session][task] = {
                    "left": left_epochs,
                    "right": right_epochs,
                }

    return mne_data_dict
