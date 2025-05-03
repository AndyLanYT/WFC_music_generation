# import pandas as pd
# import numpy as np
# import matplotlib.pylab as plt
# import seaborn as sns

# from glob import glob

# import librosa
# import librosa.display
# import IPython.display as ipd

# from itertools import cycle

# sns.set_theme(style="white", palette=None)
# color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# y, sr = librosa.load('music_synthesis/audio/220Hz_Lan.wav')

# # amplitude = librosa.stft(y)

# # # power = np.abs(amplitude) ** 2

# # amplitude_to_db = librosa.amplitude_to_db(amplitude)
# # power_to_db = librosa.power_to_db(power)
# # db = 10 * np.log10(np.maximum(power, 1e-3))

# N = 2 ** 14
# filter_banks = librosa.filters.mel(n_fft=N, sr=sr, n_mels=5)

# # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
# # log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

# # fig, ax = plt.subplots(figsize=(10, 5))
# # img = librosa.display.specshow(log_mel_spectrogram,
# #                               x_axis='time',
# #                               y_axis='mel',
# #                               ax=ax)
# # fig.colorbar(img, ax=ax, format=f'%0.2f')
# # plt.show()

# plt.xlabel('Sample')
# plt.ylabel('Weight')

# plt.grid(True, linestyle="--", alpha=0.7)

# plt.xlim(0, len(filter_banks[0]))
# plt.ylim(-0.003, 1)

# # plt.text(5, 1.2, "Frequency (mel)", fontsize=12, ha="center", va="center", fontweight="bold")
# # plt.text(5, -1.3, "Frequency (Hz)", fontsize=12, ha="center", va="center", fontweight="bold")


# for idx, fb in enumerate(filter_banks):
#     plt.plot(fb / np.max(fb))

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import librosa


def mel_scale(f):
    return 2595 * np.log10(1 + f / 700)  # Convert Hz to Mel

def inv_mel_scale(m):
    return 700 * (10**(m / 2595) - 1)  # Convert Mel to Hz

sr = 16000
n_filters = 6
n_fft = 1024
low_freq = 0
high_freq = sr / 2

mel_points = np.linspace(mel_scale(low_freq), mel_scale(high_freq), n_filters + 2)
hz_points = inv_mel_scale(mel_points)
bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

filters = np.zeros((n_filters, int(n_fft // 2 + 1)))
for i in range(1, n_filters + 1):
    left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
    filters[i - 1, left:center] = np.linspace(0, 1, center - left)
    filters[i - 1, center:right] = np.linspace(1, 0, right - center)


plt.figure(figsize=(10, 5))

plt.xlim(0, sr//2)
plt.ylim(-0.003, 1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weight")
plt.grid(True, linestyle="--", alpha=0.5)

for i in range(n_filters):
    plt.plot(np.linspace(0, sr / 2, len(filters[i])), filters[i], label=f"Filter {i+1}")


hz_tick_positions = np.linspace(low_freq, high_freq, sr // 2000 + 1)
mel_tick_positions = mel_scale(np.linspace(low_freq, high_freq, n_filters + 2))
plt.xticks(hz_tick_positions, [f"{int(h)}" for h in hz_tick_positions])

plt.text(sr//4, 1.067, 'Frequency (Mel)', ha="center", fontsize=10)
for i, mel_pos in enumerate(mel_tick_positions):
    plt.text(hz_points[i], 1.015, f"{int(mel_points[i])}", ha="center", fontsize=10)


# Show plot
plt.legend()
plt.show()
