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

# use the bg_depth as a starting point
@numba.njit()
def calculate_screen_space_shadows_v2(
    light_direction: np.ndarray[np.float64], # [x,y,z]
    bg_depth_scaled,
    comp_depth,
    composite_mask, # NOTE: if this mask is set to always be one, then the mask will display all shadows
    depth_cutoff:float,
    ignore_fg=True, # set to False if there's a wall in the way or something.
):
    """
    params:
    - light_direction is a vector which represents the ray direction of the light. Shadows will step in the opposite direction.
    - `composite_mask` is the mask which is the size of the original image & contains only the composite image. If a pixel is behind this one, then we should include the shadow.
    - `depth_multiplier` refers to the conversion rate between the depth values and the actual metric depth of the scene. Amaller than 1.0 makes the scene more shallow, while larger than 1.0 makes the scene deeper.

    returns:
    - a colored mask representing which pixels were shaded by the new composited image
    """

    height, width = comp_depth.shape[0], comp_depth.shape[1]

    shaded_mask = np.zeros_like(composite_mask)

    # a step should be the correct size so that the last step ends up higher than the max depth in the image.
    min_depth = np.min(comp_depth)
    max_depth = np.max(comp_depth)

    step_vector = -light_direction / light_direction[2] * (min_depth - max_depth) 
    # NOTE: 512 is faster, but less accurate. For most images it's practically equivalent.
    #MAX_STEPS = 512
    MAX_STEPS = np.maximum(np.abs(step_vector[0]), np.abs(step_vector[1]))
    step_vector /= MAX_STEPS

    MAX_STEPS_INT = int(np.maximum(np.iinfo(np.int32).max, int(MAX_STEPS)+1))

    for y in range(height):
        # TODO: implement this using raycasting instead of raymarching
        # Intersect ray with the quad generated from the corners of the pixel 
        for x in range(width):
            z = bg_depth_scaled[y, x]
            camera_relative_coord = np.asarray([x, y, z], dtype=np.float64)
        
            for step in range(MAX_STEPS_INT):
                camera_relative_coord += step_vector

                approx_x = int(camera_relative_coord[0])
                approx_y = int(camera_relative_coord[1])

                # check boundary conditions
                if approx_x < 0 or approx_x >= comp_depth.shape[1]:
                    break
                elif approx_y < 0 or approx_y >= comp_depth.shape[0]:
                    break
                
                # take a step, and check the texture again
                current_depth_loc = camera_relative_coord[2]
                depth_map_value = comp_depth[approx_y, approx_x]

                # let the light ray go behind our object if there is space
                # set shadow iff the shadow was being casted by our object specifically
                if current_depth_loc < depth_map_value and composite_mask[approx_y, approx_x] > 0.5 and current_depth_loc > depth_cutoff:
                    shaded_mask[y, x] = 1
                elif (not ignore_fg) and current_depth_loc < depth_map_value and composite_mask[approx_y, approx_x] <= 0.0 and step > int(MAX_STEPS * 0.01):
                    break
                    
    return shaded_mask

# combine depth maps
# NOTE: this will very likely generate depth maps that cannot immediately be converted into rgb8
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

    bg_depth_scaled = bg_depth_multiplier * bg_depth.copy()

    return combined_depth, bg_depth_scaled, depth_cutoff

def run_shadow_pipeline(
    folder_name: str, 
    shadow_opacity: float, 
    shadow_blur_px: int,

    bg_depth_multiplier: float,
    fg_squish: float,
    fg_depth_pad: float,
    fg_distance: float,

    ignore_fg:bool,
):  
    print("\n1.1 load our images")
    bg_im          = load_image(f"output/{folder_name}/bg_im.png")
    bg_depth       = load_image(f"output/{folder_name}/bg_depth.png")
    bg_inv_shading = load_image(f"output/{folder_name}/bg_inv_shading.png")

    comp             = load_image(f"output/{folder_name}/comp.png")
    comp_inv_shading = load_image(f"output/{folder_name}/comp_inv_shading.png")[:, :, np.newaxis]
    comp_normals     = load_image(f"output/{folder_name}/comp_normals.png") 
    comp_depth_og    = load_image(f"output/{folder_name}/comp_depth.png")

    # get "full image mask" from the selected area mask 
    fg_full_mask  = load_image(f"output/{folder_name}/fg_full_mask.png")
    fg_full_depth = load_image(f"output/{folder_name}/fg_full_depth.png")

    bg_height = bg_im.shape[0]
    bg_width  = bg_im.shape[1]

    print("\n2.1 loading models")
    normals_model = load_omni_model()
    albedo_model = intrinsic_compositing.albedo.pipeline.load_albedo_harmonizer()
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
    light_direction = np.asarray([coeffs[0], coeffs[1], coeffs[2]], dtype=np.float64)
    light_direction = light_direction / np.linalg.norm(light_direction)
    light_direction[2] = -abs(light_direction[2])
    #light_direction[2] = np.sqrt(1 - light_direction[0]**2 - light_direction[1]**2)
    
    print("[x, y, -z], c")
    print(coeffs)

    print("\n3. combine depth maps")
    combined_depth, bg_depth_scaled, depth_cutoff = combine_depth(
        bg_depth,
        fg_full_mask,
        fg_full_depth,
        
        bg_depth_multiplier,
        fg_squish,
        fg_depth_pad,
        fg_distance,
    )

    print(f"\tbg_min_depth before:\t{np.min(bg_depth)}")
    print(f"\tbg_max_depth before:\t{np.max(bg_depth)}")
    print(f"\tbg_min_depth combined:\t{np.min(combined_depth)}")
    print(f"\tbg_max_depth combined:\t{np.max(combined_depth)}")

    print(f"\n\t depth_cutoff: {depth_cutoff}")

    # NOTE: ensure the TMP_MAX_STEPS is correct
    step_vector = -light_direction / light_direction[2] * (np.min(combined_depth) - np.max(combined_depth)) 
    TMP_MAX_STEPS = np.maximum(np.abs(step_vector[0]), np.abs(step_vector[1]))
    step_vector /= TMP_MAX_STEPS
    print(f"\tstep_vector: {step_vector} (in px)")
    print(f"\tTMP_MAX_STEPS: {TMP_MAX_STEPS}")

    start = time.time()
    print("\n4. generate shaded maps")
    shaded_mask = calculate_screen_space_shadows_v2(
        light_direction,
        bg_depth_scaled,
        combined_depth,
        fg_full_mask,
        depth_cutoff,
        ignore_fg,
    )
    end = time.time()
    print(f"elapsed: {end-start}s")

    print("\n5. combine shaded mask with image to produce a shadow & final result")

    print("\t5.1. apply blur & intensity to shadow")

    # get to-composite shadow
    blurred_shadow_mask = np.zeros((bg_depth.shape[0], bg_depth.shape[1], 4))
    blurred_shadow_mask[:, :, 3] = shaded_mask * shadow_opacity
    blurred_shadow_mask[:, :, 3] = gaussian_filter(blurred_shadow_mask[:, :, 3], sigma=shadow_blur_px)

    # remove any blured amount that would bleed over the fg image
    BLUR_BLEED = 0.0
    blurred_shadow_mask[:, :, 3][fg_full_mask > 0.0] *= (1 - fg_full_mask[fg_full_mask > 0.0]) * (1.0 - BLUR_BLEED) + BLUR_BLEED

    print("\n6. do albedo harmonization")

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

    print("\n7. run reshading model")

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

    print("\n8. save output")

    np_to_pil(combined_depth / np.max(combined_depth)).save(f"output/{folder_name}/combined_depth.png")
    np_to_pil(shaded_mask).save(f"output/{folder_name}/shaded_mask.png")
    np_to_pil(blurred_shadow_mask).save(f"output/{folder_name}/blurred_shadow_mask.png")
    np_to_pil(final_result['composite']).save(f"output/{folder_name}/shadow_main_result.png")

# --------------------------------------------------- #

# In order to generate shadows, you must first 

if False:
    FOLDER_NAME = "dresser-music-2"
    SHADOW_OPACITY = 0.7
    SHADOW_BLUR_PX = 5

    BG_DEPTH_MULTIPLIER = 512.0,
    FG_SQUISH = 70.0,
    FG_DEPTH_PAD = 5.0,
    FG_DISTANCE = 220.0,

if True:
    FOLDER_NAME = "pillar-bag"
    SHADOW_OPACITY = 0.7
    SHADOW_BLUR_PX = 5

    BG_DEPTH_MULTIPLIER = 512.0,
    FG_SQUISH = 70.0,
    FG_DEPTH_PAD = 5.0,
    FG_DISTANCE = -50.0,

if False:
    FOLDER_NAME = "classroom-soap"
    SHADOW_OPACITY = 0.4
    SHADOW_BLUR_PX = 9

    BG_DEPTH_MULTIPLIER = 512.0,
    FG_SQUISH = 100.0,
    FG_DEPTH_PAD = 2.0,
    FG_DISTANCE = -80.0,

if False:
    FOLDER_NAME = "cone-chair"
    SHADOW_OPACITY = 0.97
    SHADOW_BLUR_PX = 1

    BG_DEPTH_MULTIPLIER = 1024.0,
    FG_SQUISH = 110.0,
    FG_DEPTH_PAD = 4.0,
    FG_DISTANCE = -55.0,

if False:
    FOLDER_NAME = "lamp-robot"
    SHADOW_OPACITY = 0.63
    SHADOW_BLUR_PX = 3

    BG_DEPTH_MULTIPLIER = 512.0,
    FG_SQUISH = 65.0,
    FG_DEPTH_PAD = 10.0,
    FG_DISTANCE = -30.0,

if __name__ == "__main__":

    run_shadow_pipeline(
        FOLDER_NAME, 
        SHADOW_OPACITY, 
        SHADOW_BLUR_PX,

        BG_DEPTH_MULTIPLIER,
        FG_SQUISH,
        FG_DEPTH_PAD,
        FG_DISTANCE,

        ignore_fg=True,
    )
