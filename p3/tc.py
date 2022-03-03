from math import ceil
from copy import deepcopy
import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method
        self.tiles = [[] for i in range(num_tilings)]
        row_min = state_low[0]
        row_max = state_high[0]
        col_min = state_low[1]
        col_max = state_high[1]
        for tiling_index in range(len(self.tiles)):
            row_val = row_min - (tiling_index/num_tilings) * tile_width[0]
            col_val = col_min - (tiling_index/num_tilings) * tile_width[1]
            for row_idx in range(ceil((row_max - row_min)/tile_width[0]) + 1):
                self.tiles[tiling_index].append([])
                for _ in range(ceil((col_max - col_min)/tile_width[1]) + 1):
                    self.tiles[tiling_index][row_idx].append((row_val, col_val, row_val+tile_width[0], col_val + tile_width[1])) #bounds bottom left and upper right
                    col_val += tile_width[1]
                
                row_val += tile_width[0]
                col_val = col_min - (tiling_index/num_tilings) * tile_width[1]
                
        # print(self.tiles)
        print("tiling rows " + str(len(self.tiles[0])))
        print("tiling cols " + str(len(self.tiles[0][0])))
        print(state_low)
        print(state_high)
        # weights should have a value per tile in each tiling
        self.weights = [[[0 for _ in range(len(self.tiles[0][0]))] for _ in range(len(self.tiles[0]))] for _ in range(num_tilings)]


    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        sanity_checks = False
        
        tile_indices = []
        # find the tiles where this state is (feature vector)
        for tiling_idx in range(len(self.tiles)):
            found_in_tiling = False
            for row_idx in range(len(self.tiles[0])):
                for col_idx in range(len(self.tiles[0][0])):
                    row_min, col_min, row_max, col_max = self.tiles[tiling_idx][row_idx][col_idx]
                    # check if s is in this tile (exclusve upper, inclusive lower)
                    srow, scol = s
                    if srow >= row_min and scol >= col_min and srow < row_max and scol < col_max:
                        assert found_in_tiling == False # should never have overlapping in the same tile layer
                        found_in_tiling = True
                        tile_indices.append((row_idx, col_idx))

                    if found_in_tiling and not sanity_checks:
                        break
                if found_in_tiling and not sanity_checks: # found match for this tile layer, move on to next layer
                    break
            assert found_in_tiling, "tiling idx : " + str(tiling_idx) + " state : " + str(s) + "\ntile layer " + str(self.tiles[tiling_idx])
        assert len(tile_indices) == len(self.tiles), "matches " + str(len(tile_indices)) + " tile layers " + str(len(self.tiles)) # should have a match for each layer in our tilemap (or we messed up the tilemap :D)
        weighted_sum_of_features = 0
        for layer, tile in enumerate(tile_indices):
            weighted_sum_of_features += self.weights[layer][tile[0]][tile[1]]
        return weighted_sum_of_features

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        editable_weights = deepcopy(self.weights)
        for tiling_idx in range(len(self.tiles)):
            for row_idx in range(len(self.tiles[0])):
                for col_idx in range(len(self.tiles[0][0])):
                    row_min, col_min, row_max, col_max = self.tiles[tiling_idx][row_idx][col_idx]
                    # check if s is in this tile (exclusve upper, inclusive lower)
                    srow, scol = s_tau
                    # only adjust weight of tile if s_tau is in it
                    if srow >= row_min and scol >= col_min and srow < row_max and scol < col_max:
                        editable_weights[tiling_idx][row_idx][col_idx] += alpha * (G - self(s_tau)) 
                        
        self.weights = editable_weights
        return None
