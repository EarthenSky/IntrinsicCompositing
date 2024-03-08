import os

import numpy as np
from PIL import Image
import skimage.transform

from boosted_depth.depth_util import create_depth_models, get_depth
from chrislib.data_util import load_image
from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan
from chrislib.normal_util import get_omni_normals

import intrinsic.model_util
import intrinsic.pipeline

from intrinsic_compositing.shading.pipeline import (
    load_reshading_model,
    compute_reshading,
    generate_shd,
    get_light_coeffs
)

from intrinsic_compositing import albedo

from omnidata_tools.model_util import load_omni_model

import utils

# ----------------------------------------------------------------- #

#IM_PATH = "../background/map-8526430.jpg"
#IM_PATH = "../background/door-8453898.jpg"
#IM_PATH = "../background/trees-8512979.jpg"
#IM_PATH = "../background/sheet-music-8463988.jpg"
#IM_PATH = "../background/soap-8429699.jpg" # TODO: test removing gamma correction for the soap example
#IM_PATH = "../background/IMG_1520.jpg"
IM_PATH = "../background/cycling-8215973.jpg"
IM_NAME = os.path.basename(IM_PATH).split(".")[0]

FG_IM_PATH = "../foreground/shampoo-1860642.png"
#FG_IM_PATH = "../foreground/dressing-table-947429.png"
#FG_IM_PATH = "../foreground/trolley-2582492.png"
FG_IM_NAME = os.path.basename(FG_IM_PATH).split(".")[0]

MAX_EDGE_SIZE = 1024
FG_RELATIVE_SCALE = 0.25 # how large the fg image should be when compared to the bg
FG_TOP_LEFT_POS = [0.55, 0.04]

if __name__ == "__main__":
    print("1.1 loading depth model")
    depth_model = create_depth_models()

    print("1.2 loading intrinsic decomposition model")
    intrinsic_model = intrinsic.model_util.load_models('paper_weights')

    print("1.3 loading normals model")
    normals_model = load_omni_model()

    print("2.1 load & resize bg im")
    im = load_image(IM_PATH)
    print(f"im shape before: {im.shape}")
    max_dim = max(im.shape[0], im.shape[1])
    scale = MAX_EDGE_SIZE / max_dim
    im = skimage.transform.resize(im, (int(im.shape[0] * scale), int(im.shape[1] * scale)))
    print(f"im shape rescaled: {im.shape}")

    print("2.2 load & resize fg im")
    fg_im_transparent = load_image(FG_IM_PATH)
    fg_im = fg_im_transparent[:, :, 0:3]
    fg_mask = fg_im_transparent[:, :, 3]
    print(f"fg im shape: {fg_im.shape}")
    print(f"fg mask shape: {fg_mask.shape}")

    bb = utils.get_bbox(fg_mask)
    fg_im_crop = fg_im[bb[0]:bb[1], bb[2]:bb[3], :].copy()
    fg_mask_crop = fg_mask[bb[0]:bb[1], bb[2]:bb[3]].copy()
    max_dim = max(fg_im_crop.shape[0], fg_im_crop.shape[1])
    fg_scale = MAX_EDGE_SIZE / max_dim
    fg_im_crop = skimage.transform.resize(fg_im_crop, (
        int(fg_im_crop.shape[0] * fg_scale), 
        int(fg_im_crop.shape[1] * fg_scale)
    ))
    fg_mask_crop = skimage.transform.resize(fg_mask_crop, (
        int(fg_mask_crop.shape[0] * fg_scale), 
        int(fg_mask_crop.shape[1] * fg_scale)
    ))
    print(f"fg_im_crop shape: {fg_im_crop.shape}")
    print(f"fg_mask_crop shape: {fg_mask_crop.shape}")
    
    # TODO: clear out the masked area of fg_im_crop with a solid color

    print("3.1 generate bg depth map") 
    im_depth = get_depth(im, depth_model)

    print("3.2 generate fg depth map")
    fg_im_depth = get_depth(fg_im_crop, depth_model)

    print("4.1 compute bg shading & albedo")
    result = intrinsic.pipeline.run_pipeline(
        intrinsic_model,
        im ** 2.2, # TODO: ask why this for gamma correction? https://en.wikipedia.org/wiki/Gamma_correction
        resize_conf=0.0,
        maintain_size=True,
        linear=True,
    )
    im_inv_shading, im_albedo = result["inv_shading"], result["albedo"]

    print("4.2 compute fg shading & albedo")
    result = intrinsic.pipeline.run_pipeline(
        intrinsic_model,
        fg_im_crop ** 2.2,
        resize_conf=0.0,
        maintain_size=True,
        linear=True,
    )
    fg_im_inv_shading, fg_im_albedo = result["inv_shading"], result["albedo"]

    print("5.1 compute bg normals")
    im_normals = get_omni_normals(normals_model, im)

    print("5.2 compute fg normals")
    # get normals just for the image fragment, it's best to send it through 
    # cropped and scaled to 1024 in order to get the most details, 
    # then we resize it to match the fragment size
    fg_im_normals = get_omni_normals(normals_model, fg_im_crop)

    print("6. create composited images")

    # rescale computed images for the selected composite location
    top, left = int(FG_TOP_LEFT_POS[0] * im.shape[0]), int(FG_TOP_LEFT_POS[1] * im.shape[1])
    fg_im_rescaled = skimage.transform.resize(fg_im_crop, (
        fg_im_crop.shape[0] * FG_RELATIVE_SCALE, 
        fg_im_crop.shape[1] * FG_RELATIVE_SCALE
    ))
    fg_mask_rescaled = skimage.transform.resize(fg_mask_crop, (
        fg_im_crop.shape[0] * FG_RELATIVE_SCALE, 
        fg_im_crop.shape[1] * FG_RELATIVE_SCALE
    ))
    fg_depth_rescaled = skimage.transform.resize(fg_im_depth, (
        fg_im_crop.shape[0] * FG_RELATIVE_SCALE, 
        fg_im_crop.shape[1] * FG_RELATIVE_SCALE
    ))
    fg_inv_shading_rescaled = skimage.transform.resize(fg_im_inv_shading, (
        fg_im_crop.shape[0] * FG_RELATIVE_SCALE, 
        fg_im_crop.shape[1] * FG_RELATIVE_SCALE
    ))
    fg_normals_rescaled = skimage.transform.resize(fg_im_normals, (
        fg_im_crop.shape[0] * FG_RELATIVE_SCALE, 
        fg_im_crop.shape[1] * FG_RELATIVE_SCALE
    ))

    comp = utils.composite_crop(
        im,
        (top, left),
        fg_im_rescaled,
        fg_mask_rescaled,
    )
    comp_depth = utils.composite_depth(
        im_depth,
        (top, left),
        fg_depth_rescaled,
        fg_mask_rescaled,
    ) # TODO: normalize the depth map? it's possible for depth values to get outside the range [0, 1]
    comp_inv_shading = utils.composite_crop(
        im_inv_shading,
        (top, left),
        fg_inv_shading_rescaled,
        fg_mask_rescaled,
    )
    #comp_albedo
    comp_normals = utils.composite_crop(
        im_normals, 
        (top, left),
        fg_normals_rescaled,
        fg_mask_rescaled,
    )

    print("7.1 harmonize albedo")

    '''
    # the albedo comes out gamma corrected so make it linear
    im_albedo_harmonized = albedo.pipeline.harmonize_albedo(
        self.comp_img,
        fg_mask,
        self.comp_shd, 
        self.alb_model,
        reproduce_paper=False,
    ) ** 2.2
    '''

    print("7.2 get shading coefficients")

    '''
    self.orig_coeffs, self.lgt_vis = get_light_coeffs(
        small_bg_shd[:, :, 0], 
        small_bg_nrm, 
        small_bg_img
    )
    '''

    print("7.3 run reshading model")
    
    # run the reshading model using the various composited components,
    # and our lighting coefficients from the user interface
    #main_result = compute_reshading(
    #    im,
    #    np.zeros_like(im), # TODO: is this correct?
    #    im_inv_shading,
    #    im_depth,
    #    im_normals,
    #    self.alb_harm,
    #    self.coeffs,
    #    self.shd_model
    #)

    print("7. write im")
    
    np_to_pil(im).save(f"output/{IM_NAME}.png")
    np_to_pil(im_depth).save(f"output/{IM_NAME}_depth.png")
    np_to_pil(im_inv_shading).save(f"output/{IM_NAME}_inv_shading.png")
    np_to_pil(im_albedo).save(f"output/{IM_NAME}_albedo.png")
    np_to_pil(im_normals).save(f"output/{IM_NAME}_normals.png")

    # NOTE: despite the naming convention, all images are the "cropped" versions
    np_to_pil(fg_im_crop).save(f"output/{FG_IM_NAME}.png")
    np_to_pil(fg_mask_crop).save(f"output/{FG_IM_NAME}_mask.png")
    np_to_pil(fg_im_depth).save(f"output/{FG_IM_NAME}_depth.png")
    np_to_pil(fg_im_inv_shading).save(f"output/{FG_IM_NAME}_inv_shading.png")
    np_to_pil(fg_im_albedo).save(f"output/{FG_IM_NAME}_albedo.png")
    np_to_pil(fg_im_normals).save(f"output/{FG_IM_NAME}_normals.png")

    np_to_pil(comp).save(f"output/{IM_NAME}_{FG_IM_NAME}.png")
    #np_to_pil(comp_mask).save(f"output/{FG_IM_NAME}_mask.png")
    np_to_pil(comp_depth).save(f"output/{IM_NAME}_{FG_IM_NAME}_depth.png")
    np_to_pil(comp_inv_shading).save(f"output/{IM_NAME}_{FG_IM_NAME}_inv_shading.png")
    #np_to_pil(comp_albedo).save(f"output/{FG_IM_NAME}_albedo.png")
    np_to_pil(comp_normals).save(f"output/{IM_NAME}_{FG_IM_NAME}_normals.png")
