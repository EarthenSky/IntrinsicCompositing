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
    im_h, im_w = im.shape[0], im.shape[1] 
    im = im.copy()
    
    loc = list(loc)

    if loc[0] < 0:
        y_offset = -loc[0]
        loc[0] = 0
    else:
        y_offset = 0
    
    if loc[1] < 0:
        x_offset = -loc[1]
        loc[1] = 0
    else:
        x_offset = 0

    x_past = (loc[1]+c_w - im_w) if (loc[1]+c_w > im_w) else 0
    y_past = (loc[0]+c_h - im_h) if (loc[0]+c_h > im_h) else 0

    if len(fg.shape) == 2:
        im_crop = im[loc[0] : loc[0] + c_h - y_offset - y_past, loc[1] : loc[1] + c_w - x_offset - x_past]
        mask = mask[y_offset : c_h - y_past, x_offset:c_w - x_past] 
        fg = fg[y_offset : c_h - y_past, x_offset:c_w  - x_past]
        comp = (im_crop * (1.0 - mask)) + (fg * mask)
        im[loc[0] : loc[0] + c_h - y_offset - y_past, loc[1] : loc[1] + c_w - x_offset] = comp

        return im
    else:
        im_crop = im[loc[0] : loc[0] + c_h - y_offset - y_past, loc[1] : loc[1] + c_w - x_offset - x_past, :]
        mask = mask[y_offset : c_h - y_past, x_offset : c_w - x_past, np.newaxis] 
        fg = fg[y_offset : c_h - y_past, x_offset : c_w - x_past]
        comp = (im_crop * (1.0 - mask)) + (fg * mask)
        im[loc[0] : loc[0] + c_h - y_offset - y_past, loc[1] : loc[1] + c_w - x_offset - x_past, :] = comp

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

def draw_normal_circle(nrm, loc, rad):
    size = rad * 2

    lin = np.linspace(-1, 1, num=size)
    ys, xs = np.meshgrid(lin, lin)

    zs = np.sqrt((1.0 - (xs**2 + ys**2)).clip(0))
    valid = (zs != 0)
    normals = np.stack((ys[valid], -xs[valid], zs[valid]), 1)

    valid_mask = np.zeros((size, size))
    valid_mask[valid] = 1

    full_mask = np.zeros((nrm.shape[0], nrm.shape[1]))
    x = loc[0] - rad
    y = loc[1] - rad
    full_mask[y : y + size, x : x + size] = valid_mask
    # nrm[full_mask > 0] = (normals + 1.0) / 2.0
    nrm[full_mask > 0] = normals

    return nrm

def viz_coeffs(coeffs, size):
    half_sz = size // 2
    nrm_circ = draw_normal_circle(
        np.zeros((size, size, 3)), 
        (half_sz, half_sz), 
        half_sz
    )
    
    out_shd = (nrm_circ.reshape(-1, 3) @ coeffs[:3]) + coeffs[-1]
    out_shd = out_shd.reshape(size, size)

    lin = np.linspace(-1, 1, num=size)
    ys, xs = np.meshgrid(lin, lin)

    zs = np.sqrt((1.0 - (xs**2 + ys**2)).clip(0))

    out_shd[zs == 0] = 0

    return (out_shd.clip(1e-4) ** (1/2.2)).clip(0, 1)