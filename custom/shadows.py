import os

import numpy as np
import skimage.transform

from chrislib.data_util import load_image
from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan

import utils

# TODO: reimplement as shader, or with numba for gpu/cpu (but fast)
def calculate_screen_space_shadows(
    light_direction: np.ndarray[np.float64], # [x,y,z]
    depth_map, # closest values will be positive, while furthest values will be 0.0 (or negative!)
    composite_mask, # NOTE: if this mask is set to always be one, then the mask will display all shadows
    #working_area, # TODO: add support for working area when using the GUI (performance boost)
    depth_cutoff:float,
):
    """
    params:
    - light_direction is a vector which represents the ray direction of the light. Shadows will step in the opposite direction.
    - `composite_mask` is the mask which is the size of the original image & contains only the composite image. If a pixel is behind this one, then we should include the shadow.
    - `depth_multiplier` refers to the conversion rate between the depth values and the actual metric depth of the scene. Amaller than 1.0 makes the scene more shallow, while larger than 1.0 makes the scene deeper.

    returns:
    - a colored mask representing which pixels were shaded by the new composited image
    """

    MAX_STEPS = 256
    height, width = depth_map.shape[0], depth_map.shape[1]

    shaded_mask = np.zeros_like(composite_mask)

    # a step should be the correct size so that the last step ends up higher than the max depth in the image.
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    print(f"\tmin_depth: {min_depth}")
    print(f"\tmax_depth: {max_depth}")

    light_direction = light_direction / np.linalg.norm(light_direction)
    step_vector = -light_direction / light_direction[2] * (min_depth - max_depth) / MAX_STEPS
    print(f"\tstep_vector: {step_vector} (in px)")

    for y in range(height):
        if y % 50 == 0: 
            print(f"\ty:{y}")
        
        # TODO: implement this using raycasting instead of raymarching
        # Intersect ray with the quad generated from the corners of the pixel 
        for x in range(width):
            z = depth_map[y, x]
            camera_relative_coord = np.asarray([x, y, z])
        
            for step in range(MAX_STEPS):
                camera_relative_coord += step_vector

                approx_x = int(camera_relative_coord[0])
                approx_y = int(camera_relative_coord[1])

                # check boundary conditions
                if approx_x < 0 or approx_x > depth_map.shape[1]:
                    break
                elif approx_y < 0 or approx_y > depth_map.shape[0]:
                    break
                
                # take a step, and check the texture again
                current_depth_loc = camera_relative_coord[2]
                depth_map_value = depth_map[approx_y, approx_x]
                if current_depth_loc < depth_map_value:
                    if composite_mask[approx_y, approx_x] > 0.0:
                        # let the light ray go behind our object if there is space
                        # if current_depth_loc > (depth_map_value - composite_mask_depth):
                        if current_depth_loc > depth_cutoff:
                            # set shadow iff the shadow was being casted by our object specifically
                            shaded_mask[y, x] = 1
                    
    return shaded_mask

# combine depth maps
# NOTE: this may generate depth maps that cannot immediately be converted into rgb8
def combine_depth(
    bg_depth: np.ndarray[np.float64], 
    fg_full_mask: np.ndarray[np.float64], 
    fg_full_depth: np.ndarray[np.float64], 
    
    bg_depth_multiplier=1.0,
    fg_squish=0.2, 
    fg_distance=0.6, # can be negative
) -> tuple[np.ndarray[np.float64], float]:
    closest_masked_z = np.max(fg_full_mask[fg_full_mask > 0.0] * bg_depth_multiplier * bg_depth[fg_full_mask > 0.0])
    depth_cutoff = closest_masked_z + fg_distance

    combined_depth = bg_depth_multiplier * bg_depth.copy()
    combined_depth[fg_full_mask > 0.0] = depth_cutoff + fg_full_depth[fg_full_mask > 0.0] * fg_squish

    return combined_depth, depth_cutoff

if __name__ == "__main__":
    FOLDER_NAME = "shampoo-cycling"

    COMBINED_DEPTH_MAP_PATH = f"output/{FOLDER_NAME}/cycling-8215973_shampoo-1860642_depth.png"
    COMBINED_NAME = os.path.basename(COMBINED_DEPTH_MAP_PATH).split(".")[0]
    #COMBINED_NAME = "cycling-8215973_shampoo-1860642"

    BG_DEPTH_PATH = f"output/{FOLDER_NAME}/cycling-8215973_depth.png"

    FG_FULL_MASK_PATH = f"output/{FOLDER_NAME}/shampoo-1860642_full_mask.png"
    FG_FULL_DEPTH_PATH = f"output/{FOLDER_NAME}/shampoo-1860642_full_depth.png"
    
    COMPOSITE_MASK = f"output/shampoo-cycling/shampoo-1860642_mask.png"
    #COMPOSITE_NAME = "shampoo-1860642_mask"
    COMPOSITE_NAME = os.path.basename(COMPOSITE_MASK).split(".")[0]
    
    # --------------------------------------------------- #
    
    print("\n1.1 load our images")
    #combined_depth = load_image(COMBINED_DEPTH_MAP_PATH)
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
        fg_squish=8.0,
        fg_distance=4.2,
    )

    print("\n2. generate shaded maps")
    light_direction = np.asarray([16, 16, -1])
    shaded_mask = calculate_screen_space_shadows(
        light_direction,
        combined_depth,
        fg_full_mask,
        #working_area=[(30, 450), (350, 700)], # (xmin, xmax), (ymin, ymax)
        depth_cutoff=depth_cutoff,
    )

    # TODO: xor the screen-space shadows with ... ?

    print("\n3. combine shaded mask with image to produce a shadow")

    print("\n4. save output")

    np_to_pil(combined_depth / np.max(combined_depth)).save(f"output/{FOLDER_NAME}/{COMBINED_NAME}_combined_depth.png")
    np_to_pil(shaded_mask).save(f"output/{FOLDER_NAME}/{COMBINED_NAME}_shaded_mask.png")

