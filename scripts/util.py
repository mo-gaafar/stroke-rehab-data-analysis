

def apply_to_dict(data_dict: dict, func: callable, test_disabled=False):
    '''Applies a function to all elements of a dictionary
    the func should take a single epoch and return a single epoch
    Example input and return schema:
        dict_epochs = {'Patient_1': {
            'Pre': {
                    'Train':{
                    'left': mne.Epochs,
                    'right': mne.Epochs
                    },
                    'Test':{
                    'left': mne.Epochs,
                    'right': mne.Epochs
                    }
                },
            'Post': {
                    'Train':{
                    'left': mne.Epochs,
                    'right': mne.Epochs
                    },
                    'Test':{
                    'left': mne.Epochs,
                    'right': mne.Epochs
                }
            }
        }
    '''
    for patient in data_dict.keys():
        for session in data_dict[patient].keys():
            for task in data_dict[patient][session].keys():
                for side in data_dict[patient][session][task].keys():
                    if test_disabled and task == 'Test':
                        continue  # Skip test data
                    data_dict[patient][session][task][side] = func(
                        data_dict[patient][session][task][side])

    return data_dict
