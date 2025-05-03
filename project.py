from abc import ABC, abstractmethod
import numpy as np
import pygame
import time
import librosa
from matplotlib import pyplot as plt
from io import BytesIO

from math import sqrt, ceil, log2
from random import uniform, choices
from glob import glob
import wave


PATH = 'music_synthesis/audio/*.wav'
SAMPLE_RATE = 22050

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800

LEFT_MARGIN = 100
RIGHT_MARGIN = LEFT_MARGIN
TOP_MARGIN = 30
BOTTOM_MARGIN = TOP_MARGIN

MAX_FIELD_WIDTH = SCREEN_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
MAX_FIELD_HEIGHT = 500

OUTPUT_LENGTH = 15
TILES_COUNT = 12

TILE_GAP = 1
BLOCK_GAP = min(max(70 // OUTPUT_LENGTH, 1), 10)

BLOCK_WIDTH = MAX_FIELD_WIDTH // OUTPUT_LENGTH
FIELD_WIDTH = BLOCK_WIDTH * OUTPUT_LENGTH

TILES_COUNT_IN_ROW = ceil(sqrt(TILES_COUNT))

TILE_WIDTH = int(((MAX_FIELD_WIDTH - BLOCK_GAP * (OUTPUT_LENGTH - 1)) / OUTPUT_LENGTH + TILE_GAP) / TILES_COUNT_IN_ROW - TILE_GAP)
TILE_HEIGHT = TILE_WIDTH

BLOCK_WIDTH = TILES_COUNT_IN_ROW * (TILE_WIDTH + TILE_GAP) - TILE_GAP
BLOCK_HEIGHT = TILES_COUNT_IN_ROW * (TILE_HEIGHT + TILE_GAP) - TILE_GAP

FIELD_WIDTH = BLOCK_WIDTH * OUTPUT_LENGTH + BLOCK_GAP * (OUTPUT_LENGTH - 1)
FIELD_HEIGHT = BLOCK_HEIGHT

SIDE_PAD = 30
TOP_PAD = 30

BLACK = 0, 0, 0
WHITE = 255, 255, 255
DARK_GREY = 64, 64, 64
DARK_BLUE = 48, 32, 128
DARK_RED = 128, 0, 0
DARK_GREEN = 0, 128, 0
BUTTON_COLORS = ((200, 200, 200), (170, 170, 170), (150, 150, 150))

pygame.font.init()
FONT = pygame.font.SysFont(None, 30)

BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40

COLLAPSE_BUTTON_POS = ((SCREEN_WIDTH-BUTTON_WIDTH) // 2 - BUTTON_WIDTH - 15, SCREEN_HEIGHT-BUTTON_HEIGHT-25)
RENOVATE_BUTTON_POS = ((SCREEN_WIDTH-BUTTON_WIDTH) // 2, SCREEN_HEIGHT-BUTTON_HEIGHT-25)
SAVE_BUTTON_POS = ((SCREEN_WIDTH-BUTTON_WIDTH) // 2 + BUTTON_WIDTH + 15, SCREEN_HEIGHT-BUTTON_HEIGHT-25)
MODE_BUTTON_POS = ((SCREEN_WIDTH-BUTTON_WIDTH) // 2 + 2 * BUTTON_WIDTH + 50, SCREEN_HEIGHT-BUTTON_HEIGHT-25)

LEFT = 1

FPS = 60

FRAME_SIZE = 2048
HOP_SIZE = 512
FILTERS_COUNT = 100


class InterfaceRender(ABC):
    @abstractmethod
    def render(self, screen, *args, **kwargs):
        pass


class InterfaceClick(ABC):
    @abstractmethod
    def check_click(self):
        pass


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

    def dominant_frequency(self):
        idx = np.argmax(self.__magnitude)
        frequencies = np.arange(len(self.__samples) // 2 + 1) * self.__sample_rate / len(self.__samples)
        
        return frequencies[idx]
    
    # render includes sound playing? it's weird, but it works well
    def render(self, screen, *args, **kwargs):
        x, y, size, mode = args

        if   mode == 0:  # waveform
            image = pygame.transform.scale(self.__wave_img, size)
        elif mode == 1:  # frequency
            image = pygame.transform.scale(self.__freq_img, size)
        elif mode == 2:  # spectrogram
            image = pygame.transform.scale(self.__spectrogram_img, size)
        elif mode == 3:  # mel_spectrogram
            image = pygame.transform.scale(self.__mel_img, size)

        rect = image.get_rect(x=x, y=y)
        mouse_pos = pygame.mouse.get_pos()

        if rect.collidepoint(mouse_pos):
            self.__sound.play()
            print(self.dominant_frequency())
            
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
    

class SoundBlock(InterfaceRender):
    def __init__(self, tileset, x, y):
        self.__tiles = tileset
        
        self.__x = x
        self.__y = y

    @property
    def tiles(self):
        return self.__tiles
    
    @tiles.setter
    def tiles(self, val):
        if isinstance(val, SoundTile):
            self.__tiles = [val]
        elif isinstance(val, list):
            self.__tiles = val

    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y

    def render(self, screen, mode):
        if len(self.__tiles) == 1:
            x = self.__x
            y = SCREEN_HEIGHT - BLOCK_HEIGHT - 200
            size = BLOCK_WIDTH, BLOCK_HEIGHT

            tile = self.__tiles[0]
            tile.render(screen, x, y, size, mode, is_single=True)
        
        else:
            for tile in self.__tiles:
                x = self.__x + tile.idx %  TILES_COUNT_IN_ROW * (TILE_WIDTH + TILE_GAP)
                y = self.__y + tile.idx // TILES_COUNT_IN_ROW * (TILE_HEIGHT + TILE_GAP)
                size = TILE_WIDTH, TILE_HEIGHT

                tile.render(screen, x, y, size, mode, is_single=False)

    def remove(self, tile):
        self.__tiles.remove(tile)

    def __getitem__(self, key):
        for tile in self.__tiles:
            if tile.idx == key:
                return tile
    
    def __setitem__(self, key, val):
        self.__tiles[key] = val
    
    def __len__(self):
        return len(self.__tiles)
    
    def __contains__(self, key):
        if isinstance(key, SoundTile):
            return key in self.__tiles
        elif isinstance(key, int):
            return key in map(lambda tile: tile.idx, self.__tiles)
    
    def __eq__(self, val):
        return self.__tiles == val.tiles
    
    def __iter__(self):
        return iter(self.__tiles)

    def __repr__(self):
        return f'SoundBlock{self.__tiles}'
        return str(self.__tiles)


class Rules:
    def __init__(self, tileset):
        self.__rules = {}
        for tile in tileset:
            self.__rules[tile.idx] = {}
            
            for direction in [-1, 1]:
                self.__rules[tile.idx][direction] = []
                
                for neighbor in tileset:
                    if self.__is_similar(tile, neighbor, direction):
                        self.__rules[tile.idx][direction].append(neighbor.idx)
    
# ---- HERE ----
    # ---- HERE ----
    # ---- HERE ----
    # ---- HERE ----
    # ---- HERE ----
    # ---- HERE ----
# ---- HERE ----

# START AND FINISH WITH THE SAME NOTE (TONIC) 

    def __is_similar(self, tile, neighbor, direction, k=0.8):
        # a = min([2**(i/12) / (tile.dominant_frequency() / neighbor.dominant_frequency()) for i in [-12, -9, -8, -7, -5, -4, -3, 0, 3, 4, 5, 7, 8, 9, 12]]) <= 2 ** (1/24)
        
        n = 12 * log2(tile.dominant_frequency() / neighbor.dominant_frequency())
        for step in [-12, -9, -8, -7, -5, -4, -3, 0, 3, 4, 5, 7, 8, 9, 12]:
            if abs(n - step) <= 0.25:
                return True

    def is_possible_neighbor(self, tile, neighbor, direction):
        return neighbor.idx in self.__rules[tile.idx][direction]


class Button(InterfaceRender, InterfaceClick):
    def __init__(self, text, pos, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, colors=BUTTON_COLORS, font=FONT):
        self.rect = pygame.Rect(*pos, width, height)
        self.text = text
        self.font = font
        self.colors = colors
        self.current_color = colors[0]
        self.clicked = False

    def render(self, screen, *args, **kwargs):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=8)

        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.colors[1]  # Hover color

            if pygame.mouse.get_pressed()[2]:
                self.clicked = True
                self.current_color = self.colors[2]  # Clicked color
                return True
            elif self.clicked:
                self.clicked = False
        else:
            self.current_color = self.colors[0]  # Normal color


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
                filepath    = playlist.audio_files[idx]
                samples, sr = playlist.audio_files_data[idx]
                image       = playlist.tile_images[idx]
                
                tile = SoundTile(filepath, samples, sr, image)
                tileset.append(tile)

            x = i * (BLOCK_WIDTH + BLOCK_GAP) + SIDE_PAD
            y = TOP_PAD

            soundblock = SoundBlock(tileset, x, y)
            self.__coeffs.append(soundblock)


    def render(self, screen, *args, **kwargs):
        for soundblock in self.__coeffs:
            soundblock.render(screen, kwargs['mode'])

    @property
    def length(self):
        return self.__length
    
    @property
    def coeffs(self):
        return self.__coeffs


class MusicGenerator:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

        self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Music generator')

        self.__clock = pygame.time.Clock()

        self.__playlist = Playlist(PATH)
        self.__sample_rate = SAMPLE_RATE
        self.__wave_function = Wave(self.__playlist, OUTPUT_LENGTH)

        self.__mode = 0

        self.__collapse_button = Button('Collapse', COLLAPSE_BUTTON_POS)
        self.__renovate_button = Button('Renovate', RENOVATE_BUTTON_POS)
        self.__save_button     = Button('Save', SAVE_BUTTON_POS)
        self.__mode_button     = Button('Change mode', MODE_BUTTON_POS)

        self.__start = False
        self.__runner = True

    def get_soundblock_tile_idx(self, mouse_x, mouse_y):
        soundblock_idx = None
        tile_idx = None 

        if SIDE_PAD < mouse_x < SIDE_PAD + FIELD_WIDTH and TOP_PAD < mouse_y < TOP_PAD + FIELD_HEIGHT:
            soundblock_idx = (mouse_x - SIDE_PAD) // (BLOCK_WIDTH + BLOCK_GAP)
            soundblock = self.__wave_function.coeffs[soundblock_idx]

            if soundblock.x < mouse_x < soundblock.x + BLOCK_WIDTH and soundblock.y < mouse_y < soundblock.y + BLOCK_HEIGHT:
                x = (mouse_x - SIDE_PAD) % (BLOCK_WIDTH + BLOCK_GAP) // (TILE_WIDTH + TILE_GAP)
                y = (mouse_y - TOP_PAD) % (BLOCK_HEIGHT + BLOCK_GAP) // (TILE_HEIGHT + TILE_GAP)

                tile_idx = y * TILES_COUNT_IN_ROW + x

        return soundblock_idx, tile_idx

    def __check_clicked_tile(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()

        soundblock_idx, tile_idx = self.get_soundblock_tile_idx(mouse_x, mouse_y)
        if tile_idx is not None:
            soundblock = self.__wave_function.coeffs[soundblock_idx]

            if len(soundblock) != 1 and tile_idx in soundblock:
                self.__wave_function.collapse_soundblock(soundblock_idx, tile_idx)
                
                return True

    def __audio(self):
        return np.concatenate([soundblock.tiles[0].samples for soundblock in self.__wave_function.coeffs])

    def save_audio(self, filename):
        audio = self.__audio()
        print(type(audio[0]))

        with wave.open(f'music_synthesis/results/{filename}.wav', mode='wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(4)
            wav_file.setframerate(self.__sample_rate)
            wav_file.writeframes(bytes(audio))


    def __process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__runner = False
            
            elif event.type == pygame.KEYDOWN:
                if not self.__start:
                    if event.key == pygame.K_c:
                        self.__collapse_gen = self.__wave_function.collapse()
                        self.__start = True
                
                if event.key == pygame.K_r:
                    self.__wave_function.renovate(self.__playlist)
                    self.__start = False

                elif event.key == pygame.K_n:
                    try:
                        self.__wave_function.update()
                    except StopIteration:
                        self.__start = False
                    
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT:
                if self.__check_clicked_tile():
                    self.__collapse_gen = self.__wave_function.propagate()
                    self.__start = True

            elif self.__collapse_button.check_click():
                self.start_time = time.time()
                self.__collapse_gen = self.__wave_function.collapse()
                self.__start = True
                print('collapse btn')

            elif self.__renovate_button.check_click():
                # self.__wave_function.renovate(self.__playlist)
                self.__start = False
                print('renovate btn')

            elif self.__mode_button.check_click():
                self.__mode += 1
                if self.__mode == 4:
                    self.__mode = 0
                print('mode btn')
            elif self.__wave_function.is_collapsed() and self.__save_button.check_click():
                self.save_audio(input('filename: '))
                print('save btn')
                pass

    def __update(self):
        if self.__start:
            try:
                next(self.__collapse_gen)
                time.sleep(0.03)
            except StopIteration:
                self.__start = False

    def __render(self):
        self.__screen.fill((128, 128, 128))

        self.__wave_function.render(self.__screen, mode=self.__mode)
        
        self.__collapse_button.render(self.__screen)
        self.__renovate_button.render(self.__screen)
        self.__save_button.render(self.__screen)
        self.__mode_button.render(self.__screen)
        
        pygame.display.flip()

    def run(self):
        while self.__runner:
            self.__process_input()
            self.__update()
            self.__render()
        
        pygame.quit()


MusicGenerator().run()
