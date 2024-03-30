import os, time

import cupy as cp
import numpy as np
import skimage.transform
from scipy.ndimage import gaussian_filter
from numba import jit
from numba import cuda, float64
import math

from chrislib.data_util import load_image
from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan

import utils

#@jit(nopython=True)
@cuda.jit
def calculate_screen_space_shadows_cuda(light_direction, depth_map, composite_mask, depth_cutoff, shaded_mask, min_depth, max_depth):
    x, y = cuda.grid(2)  # Get the x, y position of the thread.
    height, width = depth_map.shape

    if x >= width or y >= height:
        return  # Check boundaries

    MAX_STEPS = 256
    light_direction_norm = cuda.local.array(3, dtype=float64)
    # Normalizing the light_direction vector
    norm = (light_direction[0]**2 + light_direction[1]**2 + light_direction[2]**2)**0.5
    for i in range(3):
        light_direction_norm[i] = light_direction[i] / norm
        
    step_vector = cuda.local.array(3, dtype=float64)    
    for i in range(3):
        step_vector[i] = -light_direction_norm[i] / light_direction_norm[2] * (min_depth - max_depth) / MAX_STEPS

    z = depth_map[y, x]
    camera_relative_coord = cuda.local.array(3, dtype=float64)
    camera_relative_coord[0] = x
    camera_relative_coord[1] = y
    camera_relative_coord[2] = z

    for step in range(MAX_STEPS):
        for i in range(3):
            camera_relative_coord[i] += step_vector[i]

        approx_x = int(camera_relative_coord[0])
        approx_y = int(camera_relative_coord[1])

        # Boundary check
        if approx_x < 0 or approx_x >= width or approx_y < 0 or approx_y >= height:
            break

        current_depth_loc = camera_relative_coord[2]
        depth_map_value = depth_map[approx_y, approx_x]

        if current_depth_loc < depth_map_value and composite_mask[approx_y, approx_x] > 0.0 and current_depth_loc > depth_cutoff:
            shaded_mask[y, x] = 1
              
    
def combine_depth(
    bg_depth: np.ndarray[np.float32], 
    fg_full_mask: np.ndarray[np.float32], 
    fg_full_depth: np.ndarray[np.float32], 
    
    bg_depth_multiplier=1.0,
    fg_squish=0.2,
    fg_depth_pad=0.2,
    fg_distance=0.6, # can be negative
) -> tuple[np.ndarray[np.float32], float]:
    closest_masked_z = np.max(fg_full_mask[fg_full_mask > 0.0] * bg_depth_multiplier * bg_depth[fg_full_mask > 0.0])
    depth_cutoff = closest_masked_z + fg_distance

    combined_depth = bg_depth_multiplier * bg_depth.copy()
    combined_depth[fg_full_mask > 0.0] = depth_cutoff + fg_depth_pad + fg_full_depth[fg_full_mask > 0.0] * fg_squish

    return combined_depth, depth_cutoff


if __name__ == "__main__":

    '''
    FOLDER_NAME = "cone_chair"
    COMBINED_NAME = "cone_chair"
    
    BG_DEPTH_PATH = f"examples/{FOLDER_NAME}/bg_depth.png"

    FG_FULL_MASK_PATH = f"examples/{FOLDER_NAME}/mask.png"
    FG_FULL_DEPTH_PATH = f"examples/{FOLDER_NAME}/composite_depth.png"
    '''
    
    FOLDER_NAME = "lotus-door"
    COMBINED_NAME = "lotus-door"

    BG_DEPTH_PATH = f"output/{FOLDER_NAME}/door-8453898_depth.png"

    FG_FULL_MASK_PATH = f"output/{FOLDER_NAME}/lotus-3192656_full_mask.png"
    FG_FULL_DEPTH_PATH = f"output/{FOLDER_NAME}/lotus-3192656_full_depth.png"
    
    # --------------------------------------------------- #

    '''
    FOLDER_NAME = "dresser-music"
    COMBINED_NAME = "dresser-music"

    BG_DEPTH_PATH = f"output/{FOLDER_NAME}/sheet-music-8463988_depth.png"

    FG_FULL_MASK_PATH = f"output/{FOLDER_NAME}/dressing-table-947429_full_mask.png"
    FG_FULL_DEPTH_PATH = f"output/{FOLDER_NAME}/dressing-table-947429_full_depth.png"
    '''

    # --------------------------------------------------- #

    '''
    FOLDER_NAME = "shampoo-cycling"
    COMBINED_NAME = "cycling-8215973_shampoo-1860642"

    BG_DEPTH_PATH = f"output/{FOLDER_NAME}/cycling-8215973_depth.png"

    FG_FULL_MASK_PATH = f"output/{FOLDER_NAME}/shampoo-1860642_full_mask.png"
    FG_FULL_DEPTH_PATH = f"output/{FOLDER_NAME}/shampoo-1860642_full_depth.png"
    '''
    
    # --------------------------------------------------- #
    '''
    FOLDER_NAME = "trolley-cycling"
    COMBINED_NAME = "cycling-8215973_trolley-2582492"

    BG_DEPTH_PATH = f"output/{FOLDER_NAME}/cycling-8215973_depth.png"

    FG_FULL_MASK_PATH = f"output/{FOLDER_NAME}/trolley-2582492_full_mask.png"
    FG_FULL_DEPTH_PATH = f"output/{FOLDER_NAME}/trolley-2582492_full_depth.png"
    ''' 
    # --------------------------------------------------- #

    SHADOW_OPACITY = 0.45
    SHADOW_BLUR_PX = 6

    # --------------------------------------------------- #
    
    print("\n1.1 load our images")
    bg_depth = load_image(BG_DEPTH_PATH)

    # get "full image mask" from the selected area mask 
    fg_full_mask = load_image(FG_FULL_MASK_PATH)
    fg_full_depth = load_image(FG_FULL_DEPTH_PATH)

    print(f"\tbg_min_depth before:{np.min(bg_depth)}")
    print(f"\tbg_max_depth before:{np.max(bg_depth)}")

    print("\n1.2 combine depth maps")
    combined_depth, depth_cutoff = combine_depth(
        bg_depth,
        fg_full_mask,
        fg_full_depth,
        
        bg_depth_multiplier=64.0,
        fg_squish=20.0,
        fg_depth_pad=0.0,
        fg_distance=0.2,
    )

    print(f"\tbg_min_depth combined:{np.min(combined_depth)}")
    print(f"\tbg_max_depth combined:{np.max(combined_depth)}")

    start = time.time()
    print("\n2. generate shaded maps")
    light_direction = np.asarray([-40, -40, 1], dtype=np.float32)
    # Kernel invocation
    shaded_mask_gpu = np.zeros_like(fg_full_mask)
    min_depth = combined_depth.min()
    max_depth = combined_depth.max()

    d_light_direction = cuda.to_device(light_direction)
    d_depth_map = cuda.to_device(combined_depth)
    d_composite_mask = cuda.to_device(fg_full_mask)
    shaded_mask_gpu = cuda.to_device(shaded_mask_gpu)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(fg_full_mask.shape[1] / threadsperblock[0])
    blockspergrid_y = math.ceil(fg_full_mask.shape[0] / threadsperblock[1])
    
    print(blockspergrid_x, blockspergrid_y)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    calculate_screen_space_shadows_cuda[blockspergrid, threadsperblock](d_light_direction, d_depth_map, d_composite_mask, depth_cutoff, shaded_mask_gpu, min_depth, max_depth)
    end = time.time()
    print(f"elapsed: {end-start}s")
    shaded_mask = shaded_mask_gpu.copy_to_host()
    shaded_mask[fg_full_mask > 0.0] = 0

    print("\n3. combine shaded mask with image to produce a shadow & final result")

    print("\t3.1. apply blur & intensity to shadow")

    # get to-composite shadow
    blurred_shadow_mask = np.zeros((bg_depth.shape[0], bg_depth.shape[1], 4))
    blurred_shadow_mask[:, :, 3] = shaded_mask * SHADOW_OPACITY
    blurred_shadow_mask[:, :, 3] = gaussian_filter(blurred_shadow_mask[:, :, 3], sigma=SHADOW_BLUR_PX)

    # get albedo
    # get shading
    # get harmonized albedo of composite
    # get reshading of composite

    print("\n4. save output")

    np_to_pil(combined_depth / np.max(combined_depth)).save(f"examples/{COMBINED_NAME}_combined_depth.png")
    np_to_pil(shaded_mask).save(f"examples/{COMBINED_NAME}_shaded_mask.png")
    #np_to_pil(self_shading_shaded_mask).save(f"examples/{COMBINED_NAME}_shaded_mask_self_shading.png")
    np_to_pil(blurred_shadow_mask).save(f"examples/{COMBINED_NAME}_blurred_shadow_mask.png")
