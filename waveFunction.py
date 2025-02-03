from block import Block
from tile import Tile
from math import log2
from random import uniform


class WaveFunction:
    def __init__(self, tileset, length):
        self.__length = length
        self.__coeffs = [Block(tileset, 200 * i, 100) for i in range(length)]
        self.probabilities = {tile.idx: 1 / len(tileset) for tile in tileset}
        self.__stack = []


    def __entropy(self, idx):
        block = self.__coeffs[idx]
        if len(block) == 1:
            return 0
        
        return -sum([self.probabilities[tile.idx] * log2(self.probabilities[tile.idx]) for tile in block]) - uniform(0, 0.1)

    def __min_entropy_idx(self):
        min_entropy = None
        idx = None

        for i in range(self.__length):
            entropy = self.__entropy(i)
            if entropy != 0 and (min_entropy is None or min_entropy > entropy):
                min_entropy = entropy
                idx = i
        
        return idx

    def observe(self):
        idx = self.__min_entropy_idx()
        if idx is None:
            return
        
        self.__coeffs[idx].set_random_tile(self.probabilities)

    # FIX IT...
    def propagate(self):
        while len(self.__stack) != 0:
            idx = self.__stack.pop()
            block = self.__coeffs[idx]

            for direction in [-1, 1]:
                adjacent_idx = idx + direction
                adjacent_block = self.__coeffs[adjacent_idx]

                is_changed = False
                for neighbor in adjacent_block.tiles[:]:
                    if len(adjacent_block) == 1:
                        break

                    if True not in [self.index.is_possible_neighbor(tile, neighbor, direction) for tile in block]:
                        adjacent_block.tiles.remove(neighbor)
                        is_changed = True

                        yield
                
                if is_changed:
                    self.__stack.append(adjacent_idx)

    def is_collapsed(self):
        for block in self.__coeffs:
            if len(block) > 1:
                return False
        
        return True

    def collapse(self):
        while not self.is_collapsed():
            propagation = True

            while propagation:
                pass
        
    def render(self, screen):
        for block in self.__coeffs:
            block.render(screen)


    @property
    def length(self):
        return self.__length

