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

# from waveFunction import WaveFunction
# from tile import Tile
# from block import Block
# from button import Button
# from toggle import Toggle
# from IRenderable import IRenderable


SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800

LEFT_MARGIN = 100
RIGHT_MARGIN = LEFT_MARGIN
TOP_MARGIN = 30
BOTTOM_MARGIN = TOP_MARGIN

MAX_FIELD_WIDTH = SCREEN_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
MAX_FIELD_HEIGHT = 500

OUTPUT_LENGTH = 15
TILES_COUNT = 13
    
TILE_GAP = 1
BLOCK_GAP = min(max(70 // OUTPUT_LENGTH, 1), 10)

BLOCK_WIDTH = MAX_FIELD_WIDTH // OUTPUT_LENGTH
FIELD_WIDTH = BLOCK_WIDTH * OUTPUT_LENGTH

# ROW_WIDTH = BLOCK_WIDTH - 4
# TILE_COLUMNS_COUNT = 1
# while MAX_FIELD_HEIGHT // (ROW_WIDTH + TILE_GAP) > TILES_COUNT:
#     TILE_COLUMNS_COUNT += 1




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

pygame.font.init()
FONT = pygame.font.SysFont(None, 30)

BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40

COLLAPSE_BUTTON_POS = ((SCREEN_WIDTH-BUTTON_WIDTH) // 2 - BUTTON_WIDTH - 15, SCREEN_HEIGHT-BUTTON_HEIGHT-25)
RENOVATE_BUTTON_POS = ((SCREEN_WIDTH-BUTTON_WIDTH) // 2, SCREEN_HEIGHT-BUTTON_HEIGHT-25)
SAVE_BUTTON_POS = ((SCREEN_WIDTH-BUTTON_WIDTH) // 2 + BUTTON_WIDTH + 15, SCREEN_HEIGHT-BUTTON_HEIGHT-25)

LEFT = 1

FPS = 60



FRAME_SIZE = 2048
HOP_SIZE = 512


class IRenderable(ABC):
    @abstractmethod
    def render(self, screen, *args, **kwargs):
        pass


class IClickable(ABC):
    @abstractmethod
    def check_click(self):
        pass


class Tilesheet:
    def __init__(self, path):
        try:
            self.__audio_files = glob(path)
            # [print(file) for file in self.__audio_files]
            # map(print, self.__audio_files)
        except:
            raise SystemExit(f'Path {path} is not valid')

        self.__audio_files_data = list(map(librosa.load, self.__audio_files))
        self.__tile_images = [self.__plot(y) for y, _ in self.__audio_files_data]

    def __plot(self, y):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        librosa.display.waveshow(y, alpha=0.5)
        # ax.plot(y[2000:5000])
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
    def tile_images(self):
        return self.__tile_images

    @property
    def tile_sounds(self):
        return self.__tile_sounds
    

class Tile(IRenderable):
    COUNT = 0

    def __init__(self, filepath, samples, sr, image, size=(100, 100)):
        self.__samples = samples
        self.__sample_rate = sr

        self.__ft = np.fft.fft(samples)
        self.__magnitude = np.abs(self.__ft)
        self.__stft = librosa.stft(samples, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        Y = np.abs(self.__stft) ** 2
    
        self.__image = image
        self.__sound = pygame.mixer.Sound(filepath)

        self.__idx = Tile.COUNT % TILES_COUNT
        Tile.COUNT += 1
    
    # render includes sound playing? it's weird, but it works well
    def render(self, screen, *args, **kwargs):
        x, y, size = args

        image = pygame.transform.scale(self.__image, size)

        rect = image.get_rect(x=x, y=y)
        mouse_pos = pygame.mouse.get_pos()

        if rect.collidepoint(mouse_pos):
            self.__sound.play()
            
            if not kwargs['is_single']:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill((0, 0, 0, 128))
                
                image.blit(surf, (0, 0))
        else:
            self.__sound.stop()
        
        screen.blit(image, (x, y))

    def __eq__(self, val):
        if isinstance(val, Tile):
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
    

class Block(IRenderable):
    def __init__(self, tileset, x, y):
        self.__tiles = tileset
        
        self.__x = x
        self.__y = y

    @property
    def tiles(self):
        return self.__tiles
    
    @tiles.setter
    def tiles(self, val):
        if isinstance(val, Tile):
            self.__tiles = [val]
        elif isinstance(val, list):
            self.__tiles = val

    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y

    def render(self, screen):
        if len(self.__tiles) == 1:
            x = self.__x
            y = SCREEN_HEIGHT - BLOCK_HEIGHT - 200
            size = BLOCK_WIDTH, BLOCK_HEIGHT

            tile = self.__tiles[0]
            tile.render(screen, x, y, size, is_single=True)
        
        else:
            for tile in self.__tiles:
                x = self.__x + tile.idx %  TILES_COUNT_IN_ROW * (TILE_WIDTH + TILE_GAP)
                y = self.__y + tile.idx // TILES_COUNT_IN_ROW * (TILE_HEIGHT + TILE_GAP)
                size = TILE_WIDTH, TILE_HEIGHT

                tile.render(screen, x, y, size, is_single=False)

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
        if isinstance(key, Tile):
            return key in self.__tiles
        elif isinstance(key, int):
            return key in map(lambda tile: tile.idx, self.__tiles)
    
    def __eq__(self, val):
        return self.__tiles == val.tiles
    
    def __iter__(self):
        return iter(self.__tiles)

    def __repr__(self):
        return str(self.__tiles)


class Index:
    def __init__(self, tileset):
        self.__rules = {}
        for tile in tileset:
            self.__rules[tile.idx] = {}
            
            for direction in [-1, 1]:
                self.__rules[tile.idx][direction] = []
                
                for neighbor in tileset:
                    if self.__is_similar(tile, neighbor, direction):
                        self.__rules[tile.idx][direction].append(neighbor.idx)
    
    def __is_similar(self, tile, neighbor, direction, k=0.8):
        return False
    
    def is_possible_neighbor(self, tile, neighbor, direction):
        return neighbor.idx in self.__rules[tile.idx][direction]


class Button(IRenderable, IClickable):
    def __init__(self, x, y, width, height, text, func, font, colors):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.func = func
        self.font = font
        self.colors = colors
        self.current_color = colors[0]
        self.clicked = False

    def render(self, screen, *args, **kwargs):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=8)

        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def update(self, event):
        mouse_pos = pygame.mouse.get_pos()

        # Check for hover
        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.colors[1]  # Hover color

            # Check for click
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.current_color = self.colors[2]  # Clicked color
                self.clicked = True

            elif event.type == pygame.MOUSEBUTTONUP and self.clicked:
                self.clicked = False
                self.func()  # Call the button's function
        else:
            self.current_color = self.colors[0]  # Normal color

# Dummy functions for buttons
def play_sound():
    print("Play sound button clicked!")

def merge_audio():
    print("Merge audio button clicked!")

button_colors = ((200, 200, 200), (170, 170, 170), (150, 150, 150))  # Normal, Hover, Clicked


class Button(IRenderable, IClickable):
    def __init__(self, text, pos):
        self.__is_pressed = False

        elevation = int(0.14 * BUTTON_HEIGHT)
        self.__elevation = elevation
        self.__dynamic_elevation = elevation
        self.__y = pos[1]

        self.__bottom_rect = pygame.Rect(pos, (BUTTON_WIDTH, BUTTON_HEIGHT + elevation))
        self.__bottom_color = BLACK

        self.__top_rect = pygame.Rect(pos, (BUTTON_WIDTH, BUTTON_HEIGHT))
        self.__top_color = DARK_BLUE

        self.__text_surf = FONT.render(text, True, WHITE)
        self.__text_rect = self.__text_surf.get_rect(center=self.__top_rect.center)

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()

        if self.__top_rect.collidepoint(mouse_pos):
            self.__top_color = DARK_RED
            
            if pygame.mouse.get_pressed()[0]:
                self.__is_pressed = True
                self.__dynamic_elevation = 0
            else:
                if self.__is_pressed:
                    self.__is_pressed = False
                    self.__dynamic_elevation = self.__elevation
                    
                    return True
        else:
            self.__top_color = DARK_BLUE
            self.__dynamic_elevation = self.__elevation

    def render(self, screen, *args, **kwargs):
        self.__top_rect.y = self.__y - self.__dynamic_elevation
        self.__text_rect.center = self.__top_rect.center

        self.__bottom_rect.y = self.__top_rect.y
        self.__bottom_rect.height = self.__top_rect.height + self.__dynamic_elevation

        pygame.draw.rect(screen, self.__bottom_color, self.__bottom_rect, border_radius=12)
        pygame.draw.rect(screen, self.__top_color, self.__top_rect, border_radius=12)
        screen.blit(self.__text_surf, self.__text_rect)


class WaveFunction(IRenderable):
    def __init__(self, tilesheet, length):
        self.__length = length
        
        self.__coeffs = []
        for i in range(length):
            tileset = []
            for idx in range(TILES_COUNT):
                filepath    = tilesheet.audio_files[idx]
                samples, sr = tilesheet.audio_files_data[idx]
                image       = tilesheet.tile_images[idx]
                
                tile = Tile(filepath, samples, sr, image)
                tileset.append(tile)

            x = i * (BLOCK_WIDTH + BLOCK_GAP) + SIDE_PAD
            y = TOP_PAD

            block = Block(tileset, x, y)
            self.__coeffs.append(block)

        self.__tileset = tileset
        self.probabilities = {tile.idx: 1 / len(tileset) for tile in tileset}
        self.index = Index(tileset)
        
        self.__stack = []

    def __entropy(self, block_idx):
        block = self.__coeffs[block_idx]
        if len(block) == 1:
            return 0
        
        return -sum([self.probabilities[tile.idx] * log2(self.probabilities[tile.idx]) for tile in block]) - uniform(0, 0.1)

    def __min_entropy_idx(self):
        min_entropy = None
        block_idx = None

        for i in range(self.__length):
            entropy = self.__entropy(i)
            if entropy != 0 and (min_entropy is None or min_entropy > entropy):
                min_entropy = entropy
                block_idx = i
        
        return block_idx
    
    def __valid_directions(self, block_idx):
        directions = []

        if block_idx != 0:
            directions.append(-1)
        
        if block_idx != self.length - 1:
            directions.append(1)
        
        return directions

    def collapse_block(self, block_idx, tile_idx=None):
        block = self.__coeffs[block_idx]
        if tile_idx is not None:
            block.tiles = block[tile_idx]
        else:
            block.tiles = choices(self.__tiles, [self.probabilities[tile.idx] for tile in block])
        
        self.__stack.append(block_idx)

    def observe(self):
        block_idx = self.__min_entropy_idx()
        if block_idx is None:
            return
        
        self.collapse_block(block_idx)

    def propagate(self):
        while len(self.__stack) != 0:
            block_idx = self.__stack.pop()
            block = self.__coeffs[block_idx]

            for direction in self.__valid_directions(block_idx):
                neighbor_idx = block_idx + direction
                neighbor_block = self.__coeffs[neighbor_idx]

                is_changed = False
                for neighbor_tile in neighbor_block.tiles[:]:
                    if len(neighbor_block) == 1:
                        break

                    if not any([self.index.is_possible_neighbor(tile, neighbor_tile, direction) for tile in block]):
                        neighbor_block.remove(neighbor_tile)
                        is_changed = True

                        yield
                
                if is_changed:
                    self.__stack.append(neighbor_idx)

    def is_collapsed(self):
        for block in self.__coeffs:
            if len(block) > 1:
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
    
    def renovate(self, tilesheet):
        self.__coeffs = []
        for i in range(self.__length):
            tileset = []
            for idx in range(TILES_COUNT):
                filepath    = tilesheet.audio_files[idx]
                samples, sr = tilesheet.audio_files_data[idx]
                image       = tilesheet.tile_images[idx]
                
                tile = Tile(filepath, samples, sr, image)
                tileset.append(tile)

            x = i * (BLOCK_WIDTH + BLOCK_GAP) + SIDE_PAD
            y = TOP_PAD

            block = Block(tileset, x, y)
            self.__coeffs.append(block)


    def render(self, screen, *args, **kwargs):
        for block in self.__coeffs:
            block.render(screen)

    @property
    def length(self):
        return self.__length
    
    @property
    def coeffs(self):
        return self.__coeffs


class Visualizer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

        self.__clock = pygame.time.Clock()

        self.__tilesheet = Tilesheet('music_synthesis/audio/*.wav')
        self.__wave_function = WaveFunction(self.__tilesheet, OUTPUT_LENGTH)

        self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('WFC visualizer')

        self.__collapse_button = Button('Collapse', COLLAPSE_BUTTON_POS)
        self.__renovate_button = Button('Renovate', RENOVATE_BUTTON_POS)
        self.__save_button = Button('Save', SAVE_BUTTON_POS)

        self.__start = False
        self.__runner = True


    # -----------------------------------

    def get_block_idx(self, mouse_x, mouse_y):
        block_idx = None 

        if SIDE_PAD < mouse_x < SIDE_PAD + FIELD_WIDTH and TOP_PAD < mouse_y < TOP_PAD + FIELD_HEIGHT:
            block_idx = (mouse_x - SIDE_PAD) // (BLOCK_WIDTH + BLOCK_GAP)
            block = self.__wave_function.coeffs[block_idx]

            if block.x < mouse_x < block.x + BLOCK_WIDTH and block.y < mouse_y < block.y + BLOCK_HEIGHT:
                x = (mouse_x - SIDE_PAD) % (BLOCK_WIDTH + BLOCK_GAP) // (TILE_WIDTH + TILE_GAP)
                y = (mouse_y - TOP_PAD) % (BLOCK_HEIGHT + BLOCK_GAP) // (TILE_HEIGHT + TILE_GAP)

                tile_idx = y * TILES_COUNT_IN_ROW + x

        return block_idx, tile_idx
    
    def get_tile_idx(self, mouse_x, mouse_y):
        tile_idx = None

        block_idx = self.get_block_idx(mouse_x, mouse_y)
        if block_idx is not None:
            block = self.__wave_function.coeffs[block_idx]
            if block.x < mouse_x < block.x + BLOCK_WIDTH and block.y < mouse_y < block.y + BLOCK_HEIGHT:
                x = (mouse_x - SIDE_PAD) % (BLOCK_WIDTH + BLOCK_GAP) // (TILE_WIDTH + TILE_GAP)
                y = (mouse_y - TOP_PAD) % (BLOCK_HEIGHT + BLOCK_GAP) // (TILE_HEIGHT + TILE_GAP)

                tile_idx = y * TILES_COUNT_IN_ROW + x
        
        return tile_idx
    
    # -----------------------------------
    

    def get_block_tile_idx(self, mouse_x, mouse_y):
        block_idx = None
        tile_idx = None 

        if SIDE_PAD < mouse_x < SIDE_PAD + FIELD_WIDTH and TOP_PAD < mouse_y < TOP_PAD + FIELD_HEIGHT:
            block_idx = (mouse_x - SIDE_PAD) // (BLOCK_WIDTH + BLOCK_GAP)
            block = self.__wave_function.coeffs[block_idx]

            if block.x < mouse_x < block.x + BLOCK_WIDTH and block.y < mouse_y < block.y + BLOCK_HEIGHT:
                x = (mouse_x - SIDE_PAD) % (BLOCK_WIDTH + BLOCK_GAP) // (TILE_WIDTH + TILE_GAP)
                y = (mouse_y - TOP_PAD) % (BLOCK_HEIGHT + BLOCK_GAP) // (TILE_HEIGHT + TILE_GAP)

                tile_idx = y * TILES_COUNT_IN_ROW + x

        return block_idx, tile_idx

    def __check_clicked_tile(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()

        block_idx, tile_idx = self.get_block_tile_idx(mouse_x, mouse_y)
        if tile_idx is not None:
            block = self.__wave_function.coeffs[block_idx]

            if len(block) != 1 and tile_idx in block:
                self.__wave_function.collapse_block(block_idx, tile_idx)
                
                return True

    def save_audio(self):
        pass

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
                    self.__wave_function.renovate(self.__tilesheet)
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
                # self.start_time = time.time()
                self.__wave_function.collapse()
                self.__start = True

            elif self.__renovate_button.check_click():
                # self.__wave_function.renovate(self.__tilesheet)
                self.__start = False

            elif self.__wave_function.is_collapsed() and self.__save_button.check_click():
                # self.save_image(input('filename: '))
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

        self.__wave_function.render(self.__screen)
        
        self.__collapse_button.render(self.__screen)
        self.__renovate_button.render(self.__screen)
        self.__save_button.render(self.__screen)

        # self.__propagation_toggle.render(self.__screen)
        # self.__update_toggle.render(self.__screen)
        
        pygame.display.flip()

    def run(self):
        while self.__runner:
            self.__process_input()
            self.__update()
            self.__render()
        
        pygame.quit()


Visualizer().run()
