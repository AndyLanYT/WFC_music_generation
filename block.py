from random import choices
from tile import Tile
from IRenderable import IRenderable


class Block(IRenderable):
    def __init__(self, tileset, x, y):
        self.__tiles = tileset
        
        self.__x = x
        self.__y = y

    def set_random_tile(self, probabilities):
        self.tiles = choices(self.__tiles, [probabilities[tile.idx] for tile in self.__tiles])

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
            for tile in self.__tiles:
                x = self.__x
                y = self.__y
                size = 100, 100   # block size

                tile.render(screen, x, y, size, is_single=True)
        
        else:
            for tile in self.__tiles:
                x = self.__x
                y = self.__y
                size = 100, 100   # block size

                tile.render(screen, x, y, size, is_single=False)


    def __len__(self):
        return len(self.tiles)
    