from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np
from glob import glob


PATH = 'music_synthesis/results/vis/*.wav'
SAMPLE_RATE = 22050

audio_files = glob(PATH)


# Load audio file
audio_files_data = list(map(librosa.load, audio_files))


# 1. Waveform
def plot_waveform(ax, y, sr):
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

# 2. Frequency Spectrum (FFT)
def plot_fft(ax, y, sr):
    n = len(y)
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(n, 1/sr)
    ax.plot(freqs[:n // 2], np.abs(Y[:n // 2]))
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

# 3. Spectrogram
def plot_spectrogram(y, sr):
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    # ax.set_title("Spectrogram (dB)")
    plt.colorbar(img, format="%+2.0f dB")

# 4. Mel Spectrogram
def plot_mel_spectrogram(ax, y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set_title("Mel Spectrogram")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")


for y, sr in audio_files_data:
    # Create 4 vertically stacked plots
    # fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    # plot_waveform(axs[0], y, sr)
    # plot_fft(axs[0], y, sr)
    plot_spectrogram(y, sr)
    # plot_mel_spectrogram(axs[2], y, sr)

    plt.tight_layout()
    plt.show()
