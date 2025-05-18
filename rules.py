from math import log2


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
        
        # self.__major_rules = {}
        # self.__minor_rules = {}

    def __is_similar(self, tile, neighbor, direction, k=0.25):
        n = 12 * log2(tile.fundamental_frequency() / neighbor.fundamental_frequency())
        for step in [-12, -9, -8, -7, -5, -4, -3, 0, 3, 4, 5, 7, 8, 9, 12]:
            if abs(n - step) <= k:
                return True

    # def __is_major(self, tile, key, direction, k=0.25):   # MAJOR
    #     n = 12 * log2(tile.fundamental_frequency() / key.fundamental_frequency())
    #     for step in [-12, -10, -8, -7, -5, -3, -1, 0, 2, 4, 5, 7, 9, 11, 12]:
    #         if abs(n - step) <= k:
    #             return True
    
    # def __is_minor(self, tile, key, direction, k=0.25):   # MINOR
    #     n = 12 * log2(tile.fundamental_frequency() / key.fundamental_frequency())
    #     for step in [-12, -10, -9, -7, -5, -4, -2, -0, 2, 3, 5, 7, 8, 10, 12]:
    #         if abs(n - step) <= k:
    #             return True

    def is_possible_neighbor(self, tile, neighbor, direction):
        return neighbor.idx in self.__rules[tile.idx][direction]
