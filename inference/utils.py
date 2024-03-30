import numpy as np

import skimage.transform

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

