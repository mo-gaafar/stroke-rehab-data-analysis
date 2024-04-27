import numpy as np
# plot eeg and triggers for the first 10000 samples
import matplotlib.pyplot as plt

eeg_trigs = data_dict["Patient_1"]["Pre"]["Train"]["trig"][0, :]
out_trigs = np.zeros_like(eeg_trigs)
# find the triggers
for i in range(len(eeg_trigs)):
    if (eeg_trigs[i] != eeg_trigs[i-1]):
        out_trigs.append(eeg_trigs[i])


plt.figure(figsize=(10, 5))
plt.plot(data_dict["Patient_1"]["Pre"]["Train"]["y"][0, :100000].T)
plt.plot(data_dict["Patient_1"]["Pre"]["Train"]["trig"][0, :100000])

plt.show()

np.min(argwhere(triggers!=0))

eeg_trig[1:] = (eeg_trig[1:]-eeg_trigs[:1])!=0 * (eeg_trigs[:1]==0)


00011 1 0  0 -1-1-1 0
00111 0 0 -1 -1-1 0 0
00100-1 0 -1  0 0 1 0
