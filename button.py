from constants import BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLORS, FONT
from InterfaceRender import InterfaceRender
from InterfaceClick import InterfaceClick

import pygame


class Button(InterfaceRender, InterfaceClick):
    def __init__(self, text, pos, width=BUTTON_WIDTH, height=BUTTON_HEIGHT, colors=BUTTON_COLORS, font=FONT):
        self.__rect = pygame.Rect(*pos, width, height)
        self.__text = text
        self.__font = font
        self.__colors = colors
        self.__current_color = colors[0]
        self.__clicked = False

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()

        if self.__rect.collidepoint(mouse_pos):
            self.__current_color = self.__colors[1]  # Hover color

            if pygame.mouse.get_pressed()[2]:
                self.__clicked = True
                self.__current_color = self.__colors[2]  # Clicked color
                return True
            elif self.__clicked:
                self.__clicked = False
        else:
            self.__current_color = self.__colors[0]  # Normal color

    def render(self, screen, *args, **kwargs):
        pygame.draw.rect(screen, self.__current_color, self.__rect, border_radius=8)

        text_surface = self.__font.render(self.__text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.__rect.center)
        screen.blit(text_surface, text_rect)
