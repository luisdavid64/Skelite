import random
import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from model.soft_skeletonize import SoftSkeletonize
import monai.transforms as mt
import torch.nn as nn

skel_module = SoftSkeletonize()

def get_structuring_elements():
    zeros = torch.zeros((3,3))
    ones = torch.ones((3,3))
    cross = zeros.clone()
    cross[1, :] = 1
    cross[:, 1] = 1
    small_square_up_left = ones.clone()
    small_square_up_left[2,:] = 0
    small_square_up_left[:,2] = 0
    small_square_down_right = ones.clone()
    small_square_down_right[0,:] = 0
    small_square_down_right[:,0] = 0

    d = {
        "square": ones,
        "cross": cross,
        "square_dr": small_square_down_right,
        "square_ul": small_square_up_left
    }

    return d

struct_els = get_structuring_elements()

class Random3DCrop(torch.nn.Module):
    def __init__(self, spatial_size=None):
        super().__init__()
        self.size = spatial_size
        return
    def forward(self, image, skel, spatial_size):
        if not self.size:
            self.size = spatial_size
            self.cropper = mt.RandCropByPosNegLabeld(keys=["image", "skel"], label_key="skel", spatial_size=self.size)
        d = {'image': image, 'skel': skel}
        res = self.cropper(d)
        return res[0]["image"], res[0]["skel"]

crop_3d_paired = Random3DCrop()

def random_foreground_crop(image, skel, n=10, out_size=64):
    is_3d = len(image.shape) == 4  # Assuming shape (C, D, H, W) for 3D and (C, H, W) for 2D
    out_size = (len(image.shape) - 1) * [out_size]

    for _ in range(n):
        if is_3d:
            image_crop, skel_crop = crop_3d_paired(image,skel,spatial_size=out_size)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(skel, output_size=out_size)
            skel_crop = TF.crop(skel, i, j, h, w)

        if torch.sum(skel_crop) > 0:
            if not is_3d:
                image_crop = TF.crop(image, i, j, h, w)
            return image_crop, skel_crop

    # Fallback if no valid crop is found
    if is_3d:
        image_crop, skel_crop = crop_3d_paired(image,skel, spatial_size=out_size)
    else:
        i, j, h, w = transforms.RandomCrop.get_params(skel, output_size=out_size)
        image_crop = TF.crop(image, i, j, h, w)
        skel_crop = TF.crop(skel, i, j, h, w)

    return image_crop, skel_crop

class RandomForegroundCrop(nn.Module):
    def __init__(self, n=10, out_size=64):
        super(RandomForegroundCrop, self).__init__()  # Properly initialize the parent class
        self.n = n
        self.out_size = out_size

    def forward(self, sample):
        img, skel = sample["image"], sample["skel"]
        img, skel = random_foreground_crop(img, skel, self.n, self.out_size) 
        return {"image": img, "skel": skel}

def center_crop_3d(sample, target_size=(128, 12128, 32)):
    # Compute the cropping indices
    image, skel = sample["image"], sample["skel"]

    crop_slices = [slice(None)]  # Keep all channels
    for dim_size, target in zip(image.shape[1:], target_size):
        start = (dim_size - target) // 2
        end = start + target
        crop_slices.append(slice(start, end))
    
    # Apply the crop
    cropped_image = image[crop_slices[0], crop_slices[1], crop_slices[2], crop_slices[3]]
    cropped_skel = skel[crop_slices[0], crop_slices[1], crop_slices[2], crop_slices[3]]

    return {"image": cropped_image, "skel": cropped_skel}


class PadTo32(nn.Module):
    def __init__(self):
        super(PadTo32, self).__init__()  # Properly initialize the parent class

    def forward(self, sample):
        img, skel = sample["image"], sample["skel"]
        dims = img.shape[-3:]  # Assuming [C, D, H, W] for 3D or [C, H, W] for 2D
        pad_dims = []
        for dim in reversed(dims):  # Reverse order: W, H, D
            padding = (32 - dim % 32) % 32  # Calculate padding to make divisible by 32
            pad_dims.extend([0, padding])  # (before, after) padding

        # Apply the same padding to both img and skel
        img_padded = F.pad(img, pad_dims, mode='constant', value=0)
        skel_padded = F.pad(skel, pad_dims, mode='constant', value=0)
        return {"image": img_padded, "skel": skel_padded}

padder = PadTo32()
def random_foreground_crop_with_padding(image, skel, n=10, out_size=64):
    res = padder({"image": image, "skel": skel}) 
    image = res["image"]
    skel = res["skel"]
    is_3d = len(image.shape) == 4  # Assuming shape (C, D, H, W) for 3D and (C, H, W) for 2D
    out_size = (len(image.shape) - 1) * [out_size]

    for _ in range(n):
        if is_3d:
            image_crop, skel_crop = crop_3d_paired(image,skel,spatial_size=out_size)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(skel, output_size=out_size)
            skel_crop = TF.crop(skel, i, j, h, w)

        if torch.sum(skel_crop) > 0:
            if not is_3d:
                image_crop = TF.crop(image, i, j, h, w)
            return image_crop, skel_crop

    # Fallback if no valid crop is found
    if is_3d:
        image_crop, skel_crop = crop_3d_paired(image,skel, spatial_size=out_size)
    else:
        i, j, h, w = transforms.RandomCrop.get_params(skel, output_size=out_size)
        image_crop = TF.crop(image, i, j, h, w)
        skel_crop = TF.crop(skel, i, j, h, w)

    return image_crop, skel_crop

def dilation2D(images, strel, origin=(0, 0), border_value=0):
    # Ensure images have the batch and channel dimensions
    if len(images.shape) == 2:  # If images have no batch dimension, add one
        images = images.unsqueeze(0).unsqueeze(0)
    if len(images.shape) == 3:  # If images have no channel dimension, add one
        images = images.unsqueeze(0)
    batch_size, channels, height, width = images.shape
    pad = [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1]
    images_pad = F.pad(images, pad, mode='constant', value=border_value)
    image_unfold = F.unfold(images_pad, kernel_size=strel.shape)
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    strel_flatten = strel_flatten.repeat(batch_size * channels, 1, 1)
    sums = image_unfold + strel_flatten
    result, _ = sums.max(dim=1)
    result = result.view(batch_size, channels, height, width)
    return result

def normalize(img):
    mn = img.min()
    mx = img.max()
    return (img - mn)/(mx - mn)

def dilate_image(img, n=1, device="cpu"):
    # strel = struct_els["square"].to(device=device)
    # for _ in range(n):
        # img = dilation2D(img, strel)
    img = skel_module.thicken(img=img,n_iter=n)
    img = normalize(img)
    # img = normalize(img)
    return img

class DownsampleLabel(object):
    def __call__(self, image, label):
        label_np = label.squeeze(0).numpy()
        w = label.shape[1]
        label_128 = cv2.resize(label_np, (w//2, w//2), interpolation=cv2.INTER_AREA)
        label_64 = cv2.resize(label_np, (w//4, w//4), interpolation=cv2.INTER_AREA)
        label_32 = cv2.resize(label_np, (w//8, w//8), interpolation=cv2.INTER_AREA)
        label_128 = torch.from_numpy(label_128).unsqueeze(0)
        label_64 = torch.from_numpy(label_64).unsqueeze(0)
        label_32 = torch.from_numpy(label_32).unsqueeze(0)

        return (image, [label,label_128, label_64, label_32])

class RandomDownsample3d(mt.Randomizable, mt.Transform):
    def __init__(self, keys, prob=0.1, pixdim=(2.0, 2.0, 2.0), mode='bilinear'):
        self.prob = prob
        self.spacing_transform = mt.SpacingD(keys=keys, pixdim=pixdim, mode=mode)


    def randomize(self):
        self._apply_transform = self.R.random() < self.prob

    def __call__(self, data):
        self.randomize()
        if self._apply_transform:
            data =  self.spacing_transform(data)
        return data  # Return the original image if the transform is not applied

class DownsampleAug(object):
    def __call__(self, image, label):
        ds = random.choice([0,0,0,2,4,8])
        if ds:
            img_np = image.squeeze(0).numpy()
            label_np = label.squeeze(0).numpy()
            w = label.shape[1]
            new_w = w // ds
            label = cv2.resize(label_np, (new_w, new_w), interpolation=cv2.INTER_AREA)
            label = (torch.from_numpy(label).unsqueeze(0) > 0)
            image = cv2.resize(img_np, (new_w, new_w), interpolation=cv2.INTER_AREA)
            image = (torch.from_numpy(image).unsqueeze(0) > 0)
            pad_left = (w - new_w) // 2
            pad_right = w - new_w - pad_left

            # Pad back to original size
            image = F.pad(image, (pad_left, pad_right, pad_left, pad_right), mode='constant', value=0)
            label = F.pad(label, (pad_left, pad_right, pad_left, pad_right), mode='constant', value=0)

        return (image, label)

if __name__ == "__main__":
    data_path = "/Users/luisreyes/Courses/Thesis/learnable_skeletonization_kernels/inr_skel/datasets/binary_bezier_256x256/skel_0.npy"
    data_path_img = "/Users/luisreyes/Courses/Thesis/learnable_skeletonization_kernels/inr_skel/datasets/binary_bezier_256x256/configuration_0.npy"
    skel = np.load(data_path)
    skel = torch.from_numpy(skel).float()
    img = np.load(data_path_img)
    img = torch.from_numpy(img).float()
    aug = DownsampleAug()
    img_down, skel_down = aug(img, skel)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    axes[0].imshow(skel.squeeze(), cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Original Skel')
    axes[1].imshow(skel_down.squeeze(), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Down Skel')
    axes[2].imshow(img.squeeze(), cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Original Image')
    axes[3].imshow(img_down.squeeze(), cmap='gray')
    axes[3].axis('off')
    axes[3].set_title('Down Image')

    fig.suptitle("Example", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
