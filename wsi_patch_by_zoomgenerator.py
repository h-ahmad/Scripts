# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:40:07 2023

@author: Hussain Ahmad Madni
"""

import openslide
import os
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    folder = 'patches'
    os.makedirs(folder, exist_ok=True)
    slide_folder = 'slides'
    slide_name = '2bd704_0'
    slide_path = os.path.join(slide_folder, slide_name+'.tif')
    slide = openslide.open_slide(slide_path)
    tiles = DeepZoomGenerator(slide, tile_size=128, overlap=0, limit_bounds=False) # tile size = 256
    cols, rows = tiles.level_tiles[10]
    
    for row in range(rows):
        for col in range(cols):
            temp_tile = tiles.get_tile(10, (col, row))
            temp_tile_RGB = temp_tile.convert('RGB')
            temp_tile_np = np.array(temp_tile_RGB)
            plt.imsave(os.path.join(folder, slide_name + '_'+str(row) + '_'+str(col) + ".png"), temp_tile_np)
            # plt.imshow(temp_tile_np)
            # plt.show()