from constants import SCREEN_WIDTH, SCREEN_HEIGHT, PATH, OUTPUT_LENGTH, COLLAPSE_BUTTON_POS, RENOVATE_BUTTON_POS, SAVE_BUTTON_POS, RENDER_MODE_BUTTON_POS, COLLAPSE_MODE_BUTTON_POS, SAMPLE_RATE, TOP_PAD, SIDE_PAD, FIELD_WIDTH, FIELD_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_GAP, TILE_WIDTH, TILE_HEIGHT, TILE_GAP, TILES_COUNT_IN_ROW, LEFT
from playlist import Playlist
from waveFunction import Wave
from button import Button

import numpy as np
import pygame
import wave
import time


class MusicGenerator:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()

        self.__screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Music generator')

        self.__clock = pygame.time.Clock()

        self.__playlist = Playlist(PATH)
        self.__sample_rate = SAMPLE_RATE
        length = int(input('length: '))
        self.__wave_function = Wave(self.__playlist, length)

        self.__collapse_button      = Button('Collapse', COLLAPSE_BUTTON_POS)
        self.__renovate_button      = Button('Renovate', RENOVATE_BUTTON_POS)
        self.__save_button          = Button('Save', SAVE_BUTTON_POS)
        self.__render_mode_button   = Button('Render mode', RENDER_MODE_BUTTON_POS)
        self.__collapse_mode_button = Button('Collapse mode', COLLAPSE_MODE_BUTTON_POS)

        self.__collapse_mode = 0
        self.__render_mode = 0

        self.__start = False
        self.__runner = True

        self.start_time = None

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
        audio = (self.__audio() * 32767).astype(np.int16)
        # print(np.min(audio), np.max(audio))

        with wave.open(f'music_synthesis/results/{filename}.wav', mode='wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.__sample_rate)
            wav_file.writeframes(audio.tobytes())

    def __process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__runner = False
            
            elif event.type == pygame.KEYDOWN:
                if not self.__start:
                    if event.key == pygame.K_c:
                        self.__collapse_gen = self.__wave_function.collapse()
                        self.__start = True
                        self.start_time = time.time()
                
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
                    self.start_time = time.time()

            elif self.__collapse_button.check_click():
                self.start_time = time.time()
                self.__collapse_gen = self.__wave_function.collapse()
                self.__start = True

            elif self.__renovate_button.check_click():
                self.__wave_function.renovate(self.__playlist)
                self.__start = False

            elif self.__render_mode_button.check_click():
                self.__render_mode += 1
                if self.__render_mode == 4:
                    self.__render_mode = 0
            elif self.__wave_function.is_collapsed() and self.__save_button.check_click():
                self.save_audio(input('filename: '))

    def __update(self):
        if self.__start:
            try:
                next(self.__collapse_gen)
                time.sleep(0.03)
            except StopIteration:
                self.__start = False
                print(time.time() - self.start_time)

    def __render(self):
        self.__screen.fill((128, 128, 128))

        self.__wave_function.render(self.__screen, render_mode=self.__render_mode)
        
        self.__collapse_button.render(self.__screen)
        self.__renovate_button.render(self.__screen)
        self.__save_button.render(self.__screen)
        self.__render_mode_button.render(self.__screen)
        
        pygame.display.flip()

    def run(self):
        while self.__runner:
            self.__process_input()
            self.__update()
            self.__render()
        
        pygame.quit()


MusicGenerator().run()
