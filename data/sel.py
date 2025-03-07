import torch
import torch.nn.functional as F
from .sel_constants import SEL_LIST
import os
from model.soft_skeletonize import SoftSkeletonize

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def apply_hit_miss(image, sel_list, device):
    hit_and_miss = torch.zeros(image.shape, dtype=bool).to(device)
    for hit_sel, miss_sel in sel_list:
        hit_sel = hit_sel.to(device)
        miss_sel = miss_sel.to(device)
        hit = F.conv2d(image, hit_sel, padding=1) == torch.sum(hit_sel)
        miss = F.conv2d(1 - image, miss_sel, padding=1) == torch.sum(miss_sel)
        hit_and_miss |= (hit & miss)
        del hit_sel, miss_sel, hit, miss
    return hit_and_miss.byte()

skeleton_mod = SoftSkeletonize()
def thickening_transform(image, num_steps, device, reference=None, sel_list=None, leptonica=False):
    """
        Data augmentation to thicken binary image in a topology-preserving
        manner. This function applies a thickening operation over a 
        specified number of steps. 

        If `leptonica` is True, uses hit-or-miss thickening based on code
        from leptonica. Otherwise, applies iterative dilation and uses
        reference image for topology-preservation.
    """
    with torch.no_grad():
        if leptonica and len(image.shape <= 4):
            if not sel_list:
                sel_list = SEL_LIST
            for _ in range(num_steps):
                hit_miss_result = apply_hit_miss((1 - image), sel_list, device)
                image = torch.clamp(image + hit_miss_result, min=0, max=1)
                del hit_miss_result
            return image
        else:
            if len(image.shape) > 4:
                return (skeleton_mod.iter_dilate3D(image, num_iter=num_steps) * reference)
            else:
                return (skeleton_mod.iter_dilate(image, num_iter=num_steps) * reference)
