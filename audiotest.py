import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

# sns.set_theme(style="white", palette=None)
# color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


filepath = 'music_synthesis/audio/violin.wav'

y, sr = librosa.load(filepath)
N = len(y)

ft = np.fft.fft(y)
magnitude = np.abs(ft)
frequency = np.arange(N//2) * sr / N

# print(f'shape y: {y.shape}')
# print(f'sr: {sr}')

# print(len(y) // 3)
# print(sr)

plt.plot(frequency)
plt.plot(np.arange(N//2))

plt.show()
