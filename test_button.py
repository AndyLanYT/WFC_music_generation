import pygame
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Animated Buttons")

# Button class
class Button:
    def __init__(self, x, y, width, height, text, func, font, colors):
        """
        x, y, width, height: Position and size of the button.
        text: The label on the button.
        func: The function to call when clicked.
        font: Pygame font object.
        colors: Tuple (normal, hover, clicked).
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.func = func
        self.font = font
        self.colors = colors
        self.current_color = colors[0]
        self.clicked = False

    def draw(self, surface):
        # Draw button rectangle
        pygame.draw.rect(surface, self.current_color, self.rect, border_radius=8)

        # Draw text
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

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

# Fonts and Colors
font = pygame.font.Font(None, 36)
button_colors = ((200, 200, 200), (170, 170, 170), (150, 150, 150))  # Normal, Hover, Clicked

# Create buttons
buttons = [
    Button(100, 200, 200, 50, "Play Sound", play_sound, font, button_colors),
    Button(100, 300, 200, 50, "Merge Audio", merge_audio, font, button_colors),
]

# Main loop
running = True
while running:
    screen.fill((255, 255, 255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Update buttons based on events
        for button in buttons:
            button.update(event)

    # Draw buttons
    for button in buttons:
        button.draw(screen)

    pygame.display.flip()

pygame.quit()
