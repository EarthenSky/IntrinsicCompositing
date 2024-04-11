import numpy as np

import torch

import skimage.transform

from chrislib.general import uninvert, invert, round_32, view

#import intrinsic_compositing.shading.pipeline
from intrinsic_compositing.shading.pipeline import (
    #load_reshading_model,
    #compute_reshading,
    generate_shd,
    #get_light_coeffs,
)

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rescale(im, scale):
    if scale == 1.0: return im

    h = im.shape[0]
    w = im.shape[1]

    im = skimage.transform.resize(im, (int(h * scale), int(w * scale)))
    return im

def composite_crop(im, loc, fg, mask):
    """
    This function takes `fg`, crops it, masks it, and blends it over `im`, effectively combining the two.
    """

    c_h, c_w = fg.shape[0], fg.shape[1] 
    im = im.copy()

    if len(fg.shape) == 2:
        im_crop = im[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w]
        comp = (im_crop * (1.0 - mask)) + (fg * mask)
        im[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w] = comp

        return im
    else:
        im_crop = im[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w, :]
        mask = mask[:, :, np.newaxis] 
        comp = (im_crop * (1.0 - mask)) + (fg * mask)
        im[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w, :] = comp

        return im

def composite_depth(im, loc, fg, mask):
    """
    Composite the depth of a fragment but try to match wherever the fragment is placed (fake the depth)
    """
    c_h, c_w = fg.shape[:2]

    # get the bottom-center depth of the bg
    bg_bc = loc[0] + c_h, loc[1] + (c_w // 2)
    bg_bc_val = im[bg_bc[0], bg_bc[1]].item()

    # get the bottom center depth of the fragment
    fg_bc = c_h - 1, (c_w // 2)
    fg_bc_val = fg[fg_bc[0], fg_bc[1]].item()

    # compute scale to match the fg values to bg
    scale = bg_bc_val / fg_bc_val

    im = im.copy()
    
    im_crop = im[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w]
    comp = (im_crop * (1.0 - mask)) + (scale * fg * mask)
    im[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w] = comp

    return im

def compute_reshading_with_shadow(
    comp_harmonized, 
    fg_full_mask, 
    blurred_shadow_mask,

    inv_shd, 
    depth, 
    normals, 
    alb, 
    coeffs, 
    model
):

    # expects no channel dim on msk, shd and depth
    if len(inv_shd.shape) == 3:
        inv_shd = inv_shd[:, :, 0]

    if len(fg_full_mask.shape) == 3:
        fg_full_mask = fg_full_mask[:, :, 0]

    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    h, w, _ = comp_harmonized.shape

    # max_dim = max(h, w)
    # if max_dim > 1024:
    #     scale = 1024 / max_dim
    # else:
    #     scale = 1.0

    comp_harmonized     = skimage.transform.resize(comp_harmonized, (round_32(h), round_32(w)))
    alb                 = skimage.transform.resize(alb, (round_32(h), round_32(w)))
    fg_full_mask        = skimage.transform.resize(fg_full_mask, (round_32(h), round_32(w)))
    blurred_shadow_mask = skimage.transform.resize(blurred_shadow_mask, (round_32(h), round_32(w)))
    inv_shd             = skimage.transform.resize(inv_shd, (round_32(h), round_32(w)))
    dpt                 = skimage.transform.resize(depth, (round_32(h), round_32(w)))
    nrm                 = skimage.transform.resize(normals, (round_32(h), round_32(w)))
    fg_full_mask        = fg_full_mask.astype(np.float32)

    hard_mask = (fg_full_mask > 0.5)

    reg_shd = uninvert(inv_shd)
    #img = (alb * reg_shd[:, :, None]).clip(0, 1)
    #orig_alb = comp_harmonized / reg_shd[:, :, None].clip(1e-4)
    
    bad_shd_np = reg_shd.copy()
    inf_shd = generate_shd(nrm, coeffs, hard_mask)
    bad_shd_np[hard_mask == 1] = inf_shd

    bad_img_np = alb * bad_shd_np[:, :, None]

    sem_msk = torch.from_numpy(fg_full_mask).unsqueeze(0)
    bad_img = torch.from_numpy(bad_img_np).permute(2, 0, 1)
    bad_shd = torch.from_numpy(invert(bad_shd_np)).unsqueeze(0)
    in_nrm  = torch.from_numpy(nrm).permute(2, 0, 1)
    in_dpt  = torch.from_numpy(dpt).unsqueeze(0)
    inp     = torch.cat((sem_msk, bad_img, bad_shd, in_nrm, in_dpt), dim=0).unsqueeze(0)
    inp     = inp.cuda()
    
    with torch.no_grad():
        out = model(inp).squeeze()

    fin_shd = out.detach().cpu().numpy()
    fin_shd = uninvert(fin_shd)
    #print(alb.shape)
    #print(fin_shd.shape)
    #print(blurred_shadow_mask.shape)
    fin_img = alb * (fin_shd[:, :, None] * (1.0 - blurred_shadow_mask))
    #fin_img = alb * fin_shd[:, :, None]

    normals    = skimage.transform.resize(nrm, (h, w))
    fin_shd    = skimage.transform.resize(fin_shd, (h, w))
    fin_img    = skimage.transform.resize(fin_img, (h, w))
    bad_shd_np = skimage.transform.resize(bad_shd_np, (h, w))

    result = {}
    result['reshading']    = fin_shd
    result['init_shading'] = bad_shd_np
    result['composite']    = (fin_img ** (1/2.2)).clip(0, 1)
    result['normals']      = normals

    return result
