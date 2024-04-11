import os, time

import numpy as np
import skimage.transform
from scipy.ndimage import gaussian_filter
import numba

from chrislib.data_util import load_image
from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan

import intrinsic_compositing.shading.pipeline
from intrinsic_compositing.shading.pipeline import (
    compute_reshading,
    load_reshading_model,
    generate_shd,
    get_light_coeffs
)

import intrinsic_compositing.albedo.pipeline

from omnidata_tools.model_util import load_omni_model
from chrislib.normal_util import get_omni_normals

import utils

@numba.njit()
def calculate_screen_space_shadows(
    light_direction: np.ndarray[np.float64], # [x,y,z]
    depth_map, # closest values will be positive, while furthest values will be 0.0 (or negative!)
    composite_mask, # NOTE: if this mask is set to always be one, then the mask will display all shadows
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

    MAX_STEPS = 512
    height, width = depth_map.shape[0], depth_map.shape[1]

    shaded_mask = np.zeros_like(composite_mask)

    # a step should be the correct size so that the last step ends up higher than the max depth in the image.
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)

    light_direction = light_direction / np.linalg.norm(light_direction)
    step_vector = -light_direction / light_direction[2] * (min_depth - max_depth) / MAX_STEPS
    #print(f"\tstep_vector: {step_vector} (in px)")

    for y in range(height):
        # TODO: implement this using raycasting instead of raymarching
        # Intersect ray with the quad generated from the corners of the pixel 
        for x in range(width):
            z = depth_map[y, x]
            camera_relative_coord = np.asarray([x, y, z], dtype=np.float64)
        
            for step in range(MAX_STEPS):
                camera_relative_coord += step_vector

                approx_x = int(camera_relative_coord[0])
                approx_y = int(camera_relative_coord[1])

                # check boundary conditions
                if approx_x < 0 or approx_x >= depth_map.shape[1]:
                    break
                elif approx_y < 0 or approx_y >= depth_map.shape[0]:
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
    fg_depth_pad=0.2,
    fg_distance=0.6, # can be negative
) -> tuple[np.ndarray[np.float64], float]:
    closest_masked_z = np.max(fg_full_mask[fg_full_mask > 0.0] * bg_depth_multiplier * bg_depth[fg_full_mask > 0.0])
    depth_cutoff = closest_masked_z + fg_distance

    combined_depth = bg_depth_multiplier * bg_depth.copy()
    combined_depth[fg_full_mask > 0.0] = depth_cutoff + fg_depth_pad + fg_full_depth[fg_full_mask > 0.0] * fg_squish

    return combined_depth, depth_cutoff

if __name__ == "__main__":

    FOLDER_NAME = "lotus-door"
    COMBINED_NAME = "lotus-door"

    BG_IM_PATH = f"output/{FOLDER_NAME}/door-8453898.png"
    BG_DEPTH_PATH = f"output/{FOLDER_NAME}/door-8453898_depth.png"
    BG_INV_SHADING_PATH = f"output/{FOLDER_NAME}/door-8453898_inv_shading.png"

    COMP_IM_PATH = f"output/{FOLDER_NAME}/door-8453898_lotus-3192656.png"
    COMP_INV_SHADING_PATH = f"output/{FOLDER_NAME}/door-8453898_lotus-3192656_inv_shading.png"
    COMP_NORMALS_PATH = f"output/{FOLDER_NAME}/door-8453898_lotus-3192656_normals.png"
    COMP_DEPTH_PATH = f"output/{FOLDER_NAME}/lotus-door_combined_depth.png"

    FG_FULL_MASK_PATH = f"output/{FOLDER_NAME}/lotus-3192656_full_mask.png"
    FG_FULL_DEPTH_PATH = f"output/{FOLDER_NAME}/lotus-3192656_full_depth.png"

    # --------------------------------------------------- #

    FOLDER_NAME = "lotus-door"
    COMBINED_NAME = "lotus-door"

    BG_IM_PATH = f"output/{FOLDER_NAME}/door-8453898.png"
    BG_DEPTH_PATH = f"output/{FOLDER_NAME}/door-8453898_depth.png"
    BG_INV_SHADING_PATH = f"output/{FOLDER_NAME}/door-8453898_inv_shading.png"

    COMP_IM_PATH = f"output/{FOLDER_NAME}/door-8453898_lotus-3192656.png"
    COMP_INV_SHADING_PATH = f"output/{FOLDER_NAME}/door-8453898_lotus-3192656_inv_shading.png"
    COMP_NORMALS_PATH = f"output/{FOLDER_NAME}/door-8453898_lotus-3192656_normals.png"
    COMP_DEPTH_PATH = f"output/{FOLDER_NAME}/lotus-door_combined_depth.png"

    FG_FULL_MASK_PATH = f"output/{FOLDER_NAME}/lotus-3192656_full_mask.png"
    FG_FULL_DEPTH_PATH = f"output/{FOLDER_NAME}/lotus-3192656_full_depth.png"

    # --------------------------------------------------- #

    SHADOW_OPACITY = 0.75
    SHADOW_BLUR_PX = 5

    # --------------------------------------------------- #
    
    print("\n1.1 load our images")
    bg_im          = load_image(BG_IM_PATH)
    bg_depth       = load_image(BG_DEPTH_PATH)
    bg_inv_shading = load_image(BG_INV_SHADING_PATH)

    comp             = load_image(COMP_IM_PATH)
    comp_inv_shading = load_image(COMP_INV_SHADING_PATH)[:, :, np.newaxis]
    comp_normals     = load_image(COMP_NORMALS_PATH) 
    comp_depth_og    = load_image(COMP_DEPTH_PATH)

    # get "full image mask" from the selected area mask 
    fg_full_mask  = load_image(FG_FULL_MASK_PATH)
    fg_full_depth = load_image(FG_FULL_DEPTH_PATH)

    bg_height = bg_im.shape[0]
    bg_width  = bg_im.shape[1]

    print("\n2.1 loading models")
    normals_model = load_omni_model()
    #print("\n5.2 loading albedo model")
    albedo_model = intrinsic_compositing.albedo.pipeline.load_albedo_harmonizer()
    #print("\n5.3 loading reshading model")
    reshading_model = load_reshading_model('further_trained')

    print("\n2.2 get shading coefficients")
    # to ensure that normals are globally accurate we compute them at
    # a resolution of 512 pixels, so resize our shading and image to compute 
    # rescaled normals, then run the lighting model optimization
    max_dim = max(bg_height, bg_width)
    small_height = int(bg_height * (512.0 / max_dim))
    small_width = int(bg_width * (512.0 / max_dim))
    small_bg_im = skimage.transform.resize(bg_im, (small_height, small_width), anti_aliasing=True)
    small_bg_normals = get_omni_normals(normals_model, small_bg_im)
    small_bg_inv_shading = skimage.transform.resize(bg_inv_shading, (small_height, small_width), anti_aliasing=True)

    coeffs, _ = intrinsic_compositing.shading.pipeline.get_light_coeffs(
        small_bg_inv_shading[:, :], 
        small_bg_normals,
        small_bg_im,
    )
    #light_direction = np.asarray([4, 10, -1], dtype=np.float64)
    light_direction = np.asarray([coeffs[0], coeffs[1], -coeffs[2]], dtype=np.float64)
    
    print("[x, y, z], c")
    print(coeffs)

    print("\n3. combine depth maps")
    combined_depth, depth_cutoff = combine_depth(
        bg_depth,
        fg_full_mask,
        fg_full_depth,
        
        bg_depth_multiplier=128.0,
        fg_squish=20.0,
        fg_depth_pad=0.0,
        fg_distance=0.2,
    )

    print(f"\tbg_min_depth before:\t{np.min(bg_depth)}")
    print(f"\tbg_max_depth before:\t{np.max(bg_depth)}")
    print(f"\tbg_min_depth combined:\t{np.min(combined_depth)}")
    print(f"\tbg_max_depth combined:\t{np.max(combined_depth)}")

    # NOTE: ensure the MAX_STEPS is correct
    step_vector = -light_direction / light_direction[2] * (np.min(combined_depth) - np.max(combined_depth)) / 512
    print(f"\tstep_vector: {step_vector} (in px)")

    start = time.time()
    print("\n4. generate shaded maps")
    self_shading_shaded_mask = calculate_screen_space_shadows(
        light_direction,
        combined_depth,
        fg_full_mask,
        depth_cutoff=depth_cutoff,
    )
    end = time.time()
    print(f"elapsed: {end-start}s")

    # remove internal shadows
    shaded_mask = self_shading_shaded_mask.copy()
    #shaded_mask[fg_full_mask > 0.0] = 0
    shaded_mask[fg_full_mask > 0.0] *= (1 - fg_full_mask[fg_full_mask > 0.0])

    print("\n5. combine shaded mask with image to produce a shadow & final result")

    print("\t5.1. apply blur & intensity to shadow")

    # get to-composite shadow
    blurred_shadow_mask = np.zeros((bg_depth.shape[0], bg_depth.shape[1], 4))
    blurred_shadow_mask[:, :, 3] = shaded_mask * SHADOW_OPACITY
    blurred_shadow_mask[:, :, 3] = gaussian_filter(blurred_shadow_mask[:, :, 3], sigma=SHADOW_BLUR_PX)

    # remove any blured amount that would bleed over the fg image
    BLUR_BLEED = 0.25
    blurred_shadow_mask[:, :, 3][fg_full_mask > 0.0] *= (1 - fg_full_mask[fg_full_mask > 0.0]) * BLUR_BLEED + BLUR_BLEED

    print("\n5.2 do albedo harmonization")

    # the albedo comes out gamma corrected so make it linear
    comp_albedo_harmonized = intrinsic_compositing.albedo.pipeline.harmonize_albedo(
        comp,
        fg_full_mask[:, :, np.newaxis],
        comp_inv_shading, 
        albedo_model,
        reproduce_paper=False,
    ) ** 2.2

    # Q: why do these look so weird? they just do
    #original_albedo = (comp ** 2.2) / uninvert(comp_inv_shading)
    comp_harmonized = comp_albedo_harmonized * uninvert(comp_inv_shading)

    print("\n*. run reshading model")

    # run the reshading model using the various composited components,
    # and our lighting coefficients from the user interface
    final_result = utils.compute_reshading_with_shadow(
        comp_harmonized,
        fg_full_mask,
        blurred_shadow_mask[:, :, 3:],
        comp_inv_shading,
        comp_depth_og,
        # TODO: figure out what's wrong with combined depth...
        #combined_depth, # TODO: test if this is confusing to the network (it shouldn't be....)
        comp_normals,
        comp_albedo_harmonized,
        coeffs, # TODO: do coefficients need to be reshaded?
        reshading_model,
    )

    print("\n6. save output")

    np_to_pil(combined_depth / np.max(combined_depth)).save(
        f"output/{FOLDER_NAME}/{COMBINED_NAME}_combined_depth.png")
    np_to_pil(shaded_mask).save(
        f"output/{FOLDER_NAME}/{COMBINED_NAME}_shaded_mask.png")
    np_to_pil(self_shading_shaded_mask).save(
        f"output/{FOLDER_NAME}/{COMBINED_NAME}_shaded_mask_self_shading.png")
    np_to_pil(blurred_shadow_mask).save(
        f"output/{FOLDER_NAME}/{COMBINED_NAME}_blurred_shadow_mask.png")
    np_to_pil(final_result['composite']).save(
        f"output/{FOLDER_NAME}/0_{COMBINED_NAME}_final_result.png")
    np_to_pil(comp_harmonized).save(f"output/{FOLDER_NAME}/{COMBINED_NAME}_TMP.png")
    
