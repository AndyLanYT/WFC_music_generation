import matplotlib.pyplot as plt
import pygame
import io
import numpy as np
import librosa

y, sr = librosa.load('music_synthesis/audio/violin.wav')

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))

def create_graph_surface():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    plt.subplots_adjust(left=-0.03, right=1.03, top=1, bottom=0)
    
    x = np.linspace(0, 10, 100)
    # y = np.sin(x)
    
    librosa.display.waveshow(y, alpha=0.5)
    # ax.plot(x, y)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='raw', dpi=100)
    buf.seek(0)

    image = pygame.image.fromstring(buf.getvalue(), fig.canvas.get_width_height(), 'RGBA')

    buf.close()
    plt.close(fig)

    return image


runner = True
while runner:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            runner = False

    screen.fill((128, 128, 128))

    graph_surface = create_graph_surface()
    screen.blit(graph_surface, (50, 50))

    pygame.display.flip()

pygame.quit()
