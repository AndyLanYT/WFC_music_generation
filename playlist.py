from constants import *

import numpy as np
import pygame
import librosa
from matplotlib import pyplot as plt
from io import BytesIO
from glob import glob


class Playlist:
    def __init__(self, path):
        try:
            self.__audio_files = glob(path)
        except:
            raise SystemExit(f'Path {path} is not valid')

        self.__audio_files_data = list(map(librosa.load, self.__audio_files))
        self.__fts = []
        self.__magnitudes = []
        self.__stfts = []
        self.__spectrograms = []
        self.__mel_spectrograms = []
        self.__wave_imgs = []
        self.__freq_imgs = []
        self.__spectrogram_imgs = []
        self.__mel_imgs = []
        for y, sr in self.audio_files_data:
            # print(sr)
            ft = np.fft.fft(y)
            magnitude = np.abs(ft)[:len(y) // 2 + 1]
            stft = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
            spectrogram = librosa.power_to_db(np.abs(stft) ** 2)
            mel_spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=FILTERS_COUNT))
            
            self.__fts.append(ft)
            self.__magnitudes.append(magnitude)
            self.__stfts.append(librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
            self.__spectrograms.append(spectrogram)
            self.__mel_spectrograms.append(mel_spectrogram)
            
            self.__wave_imgs.append(self.__plot_wave_img(y))
            self.__freq_imgs.append(self.__plot_freq_img(magnitude))
            self.__spectrogram_imgs.append(self.__plot_spectrogram_img(spectrogram, sr))
            self.__mel_imgs.append(self.__plot_mel_img(mel_spectrogram, sr))

    def __plot_wave_img(self, samples):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        librosa.display.waveshow(samples, alpha=0.5)
        # ax.plot(y[2000:5000])
        ax.axis('off')

        buf = BytesIO()
        fig.savefig(buf, format='raw', dpi=100)
        buf.seek(0)

        image = pygame.image.fromstring(buf.getvalue(), fig.canvas.get_width_height(), 'RGBA')

        buf.close()
        plt.close(fig)

        return image
    
    def __plot_freq_img(self, magnitude):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.plot(magnitude, linewidth=2.5)
        ax.axis('off')

        buf = BytesIO()
        fig.savefig(buf, format='raw', dpi=100)
        buf.seek(0)

        image = pygame.image.fromstring(buf.getvalue(), fig.canvas.get_width_height(), 'RGBA')

        buf.close()
        plt.close(fig)

        return image
    
    def __plot_spectrogram_img(self, spectrogram, sr, hop_length=HOP_SIZE):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        librosa.display.specshow(spectrogram,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis='time',
                                 y_axis='log')
        ax.axis('off')

        buf = BytesIO()
        fig.savefig(buf, format='raw', dpi=100)
        buf.seek(0)

        image = pygame.image.fromstring(buf.getvalue(), fig.canvas.get_width_height(), 'RGBA')

        buf.close()
        plt.close(fig)

        return image
    
    def __plot_mel_img(self, mel_spectrogram, sr, hop_length=HOP_SIZE):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        librosa.display.specshow(mel_spectrogram,
                                 sr=sr,
                                 hop_length=hop_length,
                                 x_axis='time',
                                 y_axis='mel')

        ax.axis('off')

        buf = BytesIO()
        fig.savefig(buf, format='raw', dpi=100)
        buf.seek(0)

        image = pygame.image.fromstring(buf.getvalue(), fig.canvas.get_width_height(), 'RGBA')

        buf.close()
        plt.close(fig)

        return image
    
    @property
    def audio_files(self):
        return self.__audio_files

    @property
    def audio_files_data(self):
        return self.__audio_files_data
    
    @property
    def fts(self):
        return self.__fts
    
    @property
    def magnitudes(self):
        return self.__magnitudes
    
    @property
    def stfts(self):
        return self.__stfts
    
    @property
    def spectrograms(self):
        return self.__spectrograms
    
    @property
    def mel_spectrograms(self):
        return self.__mel_spectrograms

    @property
    def wave_imgs(self):
        return self.__wave_imgs
    
    @property
    def freq_imgs(self):
        return self.__freq_imgs
    
    @property
    def spectrogram_imgs(self):
        return self.__spectrogram_imgs
    
    @property
    def mel_imgs(self):
        return self.__mel_imgs
