from constants import TILES_COUNT, BLOCK_WIDTH, BLOCK_GAP, TOP_PAD, SIDE_PAD
from soundTile import SoundTile
from soundBlock import SoundBlock
from rules import Rules
from InterfaceRender import InterfaceRender

from math import log2
from random import choices, uniform


class Wave(InterfaceRender):
    def __init__(self, playlist, length):
        self.__length = length
        
        self.__coeffs = []
        for i in range(length):
            tileset = []
            for idx in range(TILES_COUNT):
                filepath = playlist.audio_files[idx]
                samples, sr = playlist.audio_files_data[idx]
                ft = playlist.fts[idx]
                magnitude = playlist.magnitudes[idx]
                stft = playlist.stfts[idx]
                spectrogram = playlist.spectrograms[idx]
                mel_spectrogram = playlist.mel_spectrograms[idx]
                
                wave_img = playlist.wave_imgs[idx]
                freq_img = playlist.freq_imgs[idx]
                spectrogram_img = playlist.spectrogram_imgs[idx]
                mel_img = playlist.mel_imgs[idx]
                
                tile = SoundTile(filepath, samples, sr, ft, magnitude, stft, spectrogram, mel_spectrogram, wave_img, freq_img, spectrogram_img, mel_img)
                tileset.append(tile)

            x = i * (BLOCK_WIDTH + BLOCK_GAP) + SIDE_PAD
            y = TOP_PAD

            soundblock = SoundBlock(tileset, x, y)
            self.__coeffs.append(soundblock)

        self.__tiles = tileset
        self.probabilities = {tile.idx: 1 / len(tileset) for tile in tileset}
        self.rules = Rules(tileset)
        
        self.__stack = []

    def __entropy(self, soundblock_idx):
        soundblock = self.__coeffs[soundblock_idx]
        if len(soundblock) == 1:
            return 0
        
        return -sum([self.probabilities[tile.idx] * log2(self.probabilities[tile.idx]) for tile in soundblock]) - uniform(0, 0.1)

    def __min_entropy_idx(self):
        min_entropy = None
        soundblock_idx = None

        for i in range(self.__length):
            entropy = self.__entropy(i)
            if entropy != 0 and (min_entropy is None or min_entropy > entropy):
                min_entropy = entropy
                soundblock_idx = i
        
        return soundblock_idx
    
    def __valid_directions(self, soundblock_idx):
        directions = []

        if soundblock_idx != 0:
            directions.append(-1)
        
        if soundblock_idx != self.length - 1:
            directions.append(1)
        
        return directions

    def collapse_soundblock(self, soundblock_idx, tile_idx=None):
        soundblock = self.__coeffs[soundblock_idx]
        if tile_idx is not None:
            soundblock.tiles = soundblock[tile_idx]
        else:
            soundblock.tiles = choices(soundblock.tiles, [self.probabilities[tile.idx] for tile in soundblock])
        
        self.__stack.append(soundblock_idx)

    def observe(self):
        soundblock_idx = self.__min_entropy_idx()
        if soundblock_idx is None:
            return
        
        self.collapse_soundblock(soundblock_idx)

    def propagate(self):
        while len(self.__stack) != 0:
            soundblock_idx = self.__stack.pop()
            soundblock = self.__coeffs[soundblock_idx]

            for direction in self.__valid_directions(soundblock_idx):
                neighbor_idx = soundblock_idx + direction
                neighbor_soundblock = self.__coeffs[neighbor_idx]

                is_changed = False
                for neighbor_tile in neighbor_soundblock.tiles[:]:
                    if len(neighbor_soundblock) == 1:
                        break

                    if not any([self.rules.is_possible_neighbor(tile, neighbor_tile, direction) for tile in soundblock]):
                        neighbor_soundblock.remove(neighbor_tile)
                        is_changed = True

                        yield
                
                if is_changed:
                    self.__stack.append(neighbor_idx)

    def is_collapsed(self):
        for soundblock in self.__coeffs:
            if len(soundblock) > 1:
                return False
        
        return True

    def collapse(self):
        while not self.is_collapsed():
            propagate_gen = self.propagate()
            propagation = True

            while propagation:
                try:
                    next(propagate_gen)
                    yield
                except StopIteration:
                    propagation = False

            self.observe()
            yield
    
    def renovate(self, playlist):
        self.__coeffs = []
        for i in range(self.__length):
            tileset = []
            for idx in range(TILES_COUNT):
                filepath = playlist.audio_files[idx]
                samples, sr = playlist.audio_files_data[idx]
                ft = playlist.fts[idx]
                magnitude = playlist.magnitudes[idx]
                stft = playlist.stfts[idx]
                spectrogram = playlist.spectrograms[idx]
                mel_spectrogram = playlist.mel_spectrograms[idx]
                
                wave_img = playlist.wave_imgs[idx]
                freq_img = playlist.freq_imgs[idx]
                spectrogram_img = playlist.spectrogram_imgs[idx]
                mel_img = playlist.mel_imgs[idx]
                
                tile = SoundTile(filepath, samples, sr, ft, magnitude, stft, spectrogram, mel_spectrogram, wave_img, freq_img, spectrogram_img, mel_img)
                tileset.append(tile)

            x = i * (BLOCK_WIDTH + BLOCK_GAP) + SIDE_PAD
            y = TOP_PAD

            soundblock = SoundBlock(tileset, x, y)
            self.__coeffs.append(soundblock)
            self.__stack = []

    def render(self, screen, *args, **kwargs):
        for soundblock in self.__coeffs:
            soundblock.render(screen, kwargs['render_mode'])

    @property
    def length(self):
        return self.__length
    
    @property
    def coeffs(self):
        return self.__coeffs
