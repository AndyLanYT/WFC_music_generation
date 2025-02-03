import pygame

from waveFunction import WaveFunction
# from button import Button
# from toggle import Toggle


class Visualizer:
    def __init__(self):
        self.wave_function = WaveFunction()

        self.__screen = pygame.display.set_mode((self.__render_cfg.screen_width, self.__render_cfg.screen_height))
        pygame.display.set_caption('WFC visualizer')
        
        self.__clock = pygame.time.Clock()

        # self.__collapse_button = Button('Collapse', (595, 150))
        # self.__renovate_button = Button('Renovate', (595, 210))
        # self.__save_button = Button('Save', (595, 270))

        self.__runner = True
        self.__start = False

    def get_block_idx(self, mouse_x, mouse_y):
        idx = None

        return idx

    def save_audio(self):
        pass

    def __process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__runner = False
            
            elif event.type == pygame.KEYDOWN:
                if not self.__start:
                    if event.key == pygame.K_c:
                        self.__wave_function.collapse()
                        self.__start = True
                    
                    elif event.key == pygame.K_r:
                        self.__wave_function.renovate(self.__tilesheet_cfg, self.__render_cfg)
                        self.__start = False
                    
                    elif event.key == pygame.K_p:
                        self.__to_propagate = not self.__to_propagate
                
                if event.key == pygame.K_n:
                    try:
                        self.__wave_function.update()
                    except StopIteration:
                        self.__start = False
            
            
            if self.check_clicked_tile():
                self.__wave_function.propagate()
                self.__start = True

            elif self.__collapse_button.check_click():
                # self.start_time = time.time()
                self.__wave_function.collapse()
                self.__start = True

            elif self.__renovate_button.check_click():
                self.__wave_function.renovate(self.__tilesheet_cfg, self.__render_cfg)
                self.__start = False

            elif self.__wave_function.is_collapsed() and self.__save_button.check_click():
                self.save_image(input('filename: '))
            
            elif self.__propagation_toggle.check_click():
                self.__to_propagate = not self.__to_propagate
            
            elif self.__update_toggle.check_click():
                self.__to_update = not self.__to_update

    def __update(self):
        pass

    def __render(self):
        self.__screen.fill((128, 128, 128))

        self.__wave_function.render(self.__screen)
        
        # self.__collapse_button.render(self.__screen)
        # self.__renovate_button.render(self.__screen)
        # self.__save_button.render(self.__screen)

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
