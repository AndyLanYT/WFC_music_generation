import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from glob import glob

import librosa
import librosa.display


lan_file = 'music_synthesis/audio/220Hz_Lan.wav'
vat_file = 'music_synthesis/audio/220Hz_Vat.wav'
a4 = 'music_synthesis/audio/A4.wav'

lan, sr = librosa.load(lan_file)
vat, sr = librosa.load(vat_file)
a4, sr = librosa.load(a4)

N = len(lan)
print(N, sr)

ft = np.fft.fft(lan)
magnitude = np.abs(ft)
frequency = np.arange(N//2+1) * sr / N
# frequency = np.linspace(0, sr, N//2 + 1)

print(np.arange(N//2 + 1) * sr / N)

print(np.max(magnitude), np.argmax(magnitude))

plt.plot(frequency, magnitude[:N//2+1])

plt.show()
