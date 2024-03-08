import os

import numpy as np

from chrislib.data_util import load_image
from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan

def calculate_screen_space_shadows(
    light_direction:np.ndarray[np.float64], # [x,y,z]
    depth_map,
    composite_mask, # NOTE: if this mask is set to always be one, then the mask will display all shadows
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

    MAX_STEPS = 8 #16
    height, width = depth_map.shape[0], depth_map.shape[1]

    shaded_mask = np.zeros_like(composite_mask)

    # a step should be the correct size so that the last step ends up higher than the max depth in the image.
    max_depth = -depth_multiplier * np.max(depth_map)
    min_depth = -depth_multiplier * np.min(depth_map)
    light_direction = light_direction / np.linalg.norm(light_direction)
    step_vector = light_direction / light_direction[2] * (min_depth - max_depth) / MAX_STEPS # TODO: is this backwards?

    print(step_vector)

    for y in range(height):
        
        if y % 50 == 0: 
            print(f"\ty:{y}")
        
        # TODO: run the following using a compute shader or something
        for x in range(width):
            z = depth_multiplier * depth_map[y, x]
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
                if camera_relative_coord[2] < depth_map[approx_y, approx_x]:
                    # TODO: figure out why the shadows aren't...
                    # set shadow iff the shadow was being casted by our object specifically
                    shaded_mask[y, x] = composite_mask[approx_y, approx_x]

    return shaded_mask

if __name__ == "__main__":
    
    COMBINED_DEPTH_MAP_PATH = "./output/cycling-8215973_shampoo-1860642_depth.png"
    COMBINED_NAME = "cycling-8215973_shampoo-1860642" # os.path.basename(COMBINED_DEPTH_MAP_PATH).split(".")[0]
    #ORIGINAL_DEPTH_MAP = ""

    print("1. load & generate our images")
    combined_depth_map = load_image(COMBINED_DEPTH_MAP_PATH)
    everything_mask = np.ones_like(combined_depth_map, dtype=np.uint8)

    print("2. generate shaded maps")
    light_direction = np.asarray([1, 1, -1])
    shaded_mask = calculate_screen_space_shadows(light_direction, combined_depth_map, everything_mask, depth_multiplier=0.8)

    # TODO: xor the screen-space shadows with 

    print("4. save output")
    np_to_pil(shaded_mask).save(f"output/{COMBINED_NAME}_shaded_mask.png")
