
from IRenderable import IRenderable
import numpy as np


class Tile(IRenderable):
    COUNT = 0

    def __init__(self, samples, sr):
        self.__samples = samples
        self.__sample_rate = sr
        
        self.__ft = np.fft.fft(self.__samples)
        self.__magnitude = np.abs(self.__ft)

        self.__idx = Tile.COUNT % 5     # numer of tiles

    
    def plot(self):
        pass

    def render(self, screen, size):
        pass


    @property
    def samples(self):
        return self.__samples
    
    @property
    def fourier_transform(self):
        return self.__ft
    
    @property
    def magnitude(self):
        return self.__magnitude
