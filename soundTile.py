from constants import TILES_COUNT
from InterfaceRender import InterfaceRender

import pygame
import numpy as np


class SoundTile(InterfaceRender):
    COUNT = 0

    def __init__(self, filepath, samples, sr, fourier_transform, magnitude, stft, spectrogram, mel_spectrogram, wave_img, freq_img, spectrogram_img, mel_img, size=(100, 100)):
        self.__samples = samples
        self.__sample_rate = sr

        self.__ft = fourier_transform
        self.__magnitude = magnitude
        self.__stft = stft
        self.__spectrogram = spectrogram
        self.__mel_spectrogram = mel_spectrogram
    
        self.__wave_img = wave_img
        self.__freq_img = freq_img
        self.__spectrogram_img = spectrogram_img
        self.__mel_img = mel_img
        
        self.__sound = pygame.mixer.Sound(filepath)

        self.__idx = SoundTile.COUNT % TILES_COUNT
        SoundTile.COUNT += 1

    def fundamental_frequency(self):
        idx = np.argmax(self.__magnitude)
        frequencies = np.arange(len(self.__samples) // 2 + 1) * self.__sample_rate / len(self.__samples)
        
        return frequencies[idx]
    
    # render includes sound playing? it's weird, but it works well
    def render(self, screen, *args, **kwargs):
        x, y, size, render_mode = args

        if   render_mode == 0:  # waveform
            image = pygame.transform.scale(self.__wave_img, size)
        elif render_mode == 1:  # frequency
            image = pygame.transform.scale(self.__freq_img, size)
        elif render_mode == 2:  # spectrogram
            image = pygame.transform.scale(self.__spectrogram_img, size)
        elif render_mode == 3:  # mel_spectrogram
            image = pygame.transform.scale(self.__mel_img, size)

        rect = image.get_rect(x=x, y=y)
        mouse_pos = pygame.mouse.get_pos()

        if rect.collidepoint(mouse_pos):
            self.__sound.play()
            # print(self.fundamental_frequency())
            
            if not kwargs['is_single']:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill((0, 0, 0, 128))
                
                image.blit(surf, (0, 0))
        
        else:
            self.__sound.stop()
        
        screen.blit(image, (x, y))

    def __eq__(self, val):
        if isinstance(val, SoundTile):
            return self.idx == val.idx
        elif isinstance(val, int):
            return self.idx == val

    @property
    def samples(self):
        return self.__samples
    
    @property
    def fourier_transform(self):
        return self.__ft
    
    @property
    def magnitude(self):
        return self.__magnitude
    
    @property
    def idx(self):
        return self.__idx
