import os

import numpy as np
import skimage.transform

from chrislib.data_util import load_image
from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan

def calculate_screen_space_shadows(
    light_direction:np.ndarray[np.float64], # [x,y,z]
    depth_map, # closest values will be 1.0, while furthest values will be 0.0
    composite_mask, # NOTE: if this mask is set to always be one, then the mask will display all shadows
    working_area,
    composite_mask_depth:float,
    depth_multiplier:float=1.0,
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
    max_depth = depth_multiplier * np.max(depth_map)
    min_depth = depth_multiplier * np.min(depth_map)
    print(f"\tmax_depth:{max_depth}")
    print(f"\tmin_depth:{min_depth}")

    light_direction = light_direction / np.linalg.norm(light_direction)
    step_vector = -light_direction / light_direction[2] * (min_depth - max_depth) / MAX_STEPS
    print(f"\tstep_vector (in px): {step_vector}")

    for y in range(height):
        if y % 50 == 0: print(f"\ty:{y}")
        
        # TODO: implement this using raycasting instead of raymarching
        # Intersect ray with the quad generated from the corners of the pixel 
        for x in range(width):
            z = depth_multiplier * depth_map[y, x]
            camera_relative_coord = np.asarray([x, y, z])
        
            for step in range(MAX_STEPS):
                camera_relative_coord += step_vector

                approx_x = int(camera_relative_coord[0])
                approx_y = int(camera_relative_coord[1])

                # DEBUG:
                #if(x == 500 and y == 500):
                #    print(f"\t{camera_relative_coord}")
                #    print(f"\t{depth_map[approx_y, approx_x]}")

                # check boundary conditions
                if approx_x < 0 or approx_x > depth_map.shape[1]:
                    break
                elif approx_y < 0 or approx_y > depth_map.shape[0]:
                    break
                
                # take a step, and check the texture again
                current_depth_loc = camera_relative_coord[2]
                depth_map_value = depth_multiplier * depth_map[approx_y, approx_x]
                if current_depth_loc < depth_map_value:
                    if composite_mask[approx_y, approx_x] > 0.0:
                        # let the light ray go behind our object if there is space
                        #if current_depth_loc > (depth_map_value - composite_mask_depth):
                        if current_depth_loc > composite_mask_depth:
                            # set shadow iff the shadow was being casted by our object specifically
                            shaded_mask[y, x] = 1
                    

    return shaded_mask

if __name__ == "__main__":
    
    COMBINED_DEPTH_MAP_PATH = "./output/cycling-8215973_shampoo-1860642_depth.png"
    COMBINED_NAME = "cycling-8215973_shampoo-1860642" # os.path.basename(COMBINED_DEPTH_MAP_PATH).split(".")[0]

    #ORIGINAL_DEPTH_MAP = ""
    
    COMPOSITE_MASK = "./output/shampoo-1860642_mask.png"
    COMPOSITE_NAME = "shampoo-1860642_mask"
    COMPOSITE_SCALE = 0.25 # how large the fg image should be when compared to the bg
    COMPOSITE_LOCATION = [0.55, 0.04]
    
    # --------------------------------------------------- #
    
    print("1. load & generate our images")
    combined_depth_map = load_image(COMBINED_DEPTH_MAP_PATH)

    # get "full image mask" from the selected area mask 
    fg_mask_crop = load_image(COMPOSITE_MASK)
    fg_mask_rescaled = skimage.transform.resize(fg_mask_crop, (
        fg_mask_crop.shape[0] * COMPOSITE_SCALE, 
        fg_mask_crop.shape[1] * COMPOSITE_SCALE
    ), anti_aliasing=True)
    full_size_mask = np.zeros_like(combined_depth_map, dtype=np.float64)
    top = int(COMPOSITE_LOCATION[0] * combined_depth_map.shape[0])
    left = int(COMPOSITE_LOCATION[1] * combined_depth_map.shape[1])
    full_size_mask[
        top : top + fg_mask_rescaled.shape[0], 
        left : left + fg_mask_rescaled.shape[1]
    ] = fg_mask_rescaled

    #comp_depth = utils.composite_depth(
    #    im_depth,
    #    (top, left),
    #    fg_depth_rescaled,
    #    fg_mask_rescaled,
    #)

    # everything_mask = np.ones_like(combined_depth_map, dtype=np.uint8)

    print("2. generate shaded maps")
    light_direction = np.asarray([16, 16, -1])
    shaded_mask = calculate_screen_space_shadows(
        light_direction,
        combined_depth_map,
        full_size_mask,
        working_area=[(30, 450), (350, 700)], # (xmin, xmax), (ymin, ymax)
        composite_mask_depth=64.0 * 210.0 / 255.0, # TODO: make the lower bound the min + this depth
        depth_multiplier=64.0,
    )

    # TODO: xor the screen-space shadows with 

    print("4. save output")
    np_to_pil(full_size_mask).save(f"output/{COMPOSITE_NAME}_full_mask.png")
    np_to_pil(full_size_mask).save(f"full_mask.png")

    np_to_pil(shaded_mask).save(f"output/{COMBINED_NAME}_shaded_mask.png")
    np_to_pil(shaded_mask).save(f"shaded_mask.png")
