from constants import SCREEN_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT, TILES_COUNT_IN_ROW, TILE_WIDTH, TILE_HEIGHT, TILE_GAP
from soundTile import SoundTile
from InterfaceRender import InterfaceRender


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

    def render(self, screen, render_mode):
        if len(self.__tiles) == 1:
            x = self.__x
            y = SCREEN_HEIGHT - BLOCK_HEIGHT - 200
            size = BLOCK_WIDTH, BLOCK_HEIGHT

            tile = self.__tiles[0]
            tile.render(screen, x, y, size, render_mode, is_single=True)
        
        else:
            for tile in self.__tiles:
                x = self.__x + tile.idx %  TILES_COUNT_IN_ROW * (TILE_WIDTH + TILE_GAP)
                y = self.__y + tile.idx // TILES_COUNT_IN_ROW * (TILE_HEIGHT + TILE_GAP)
                size = TILE_WIDTH, TILE_HEIGHT

                tile.render(screen, x, y, size, render_mode, is_single=False)

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
