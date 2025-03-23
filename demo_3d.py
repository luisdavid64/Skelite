import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from utils.model_utils import build_model, get_model, load_model_checkpoint
from data.utils import get_config, get_device
from utils.utils import load_nii
import nibabel as nib
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def demo(config, checkpoint_path, image_path, args_device=""):
    device = get_device()
    if args_device != "":
        device = torch.device(args_device)
    print(device)
    cudnn.benchmark = True
    
    model_module = get_model(config["net_type"])
    print("Model Type: " + config["net_type"])
    
    model = build_model(model_module, config, device)
    model = model.to(device=device)
    model.eval()
    model = load_model_checkpoint(model, checkpoint_path, device)

    with torch.no_grad():
        img, affine = load_nii(image_path)
        # z: legacy parameter from exploring hypernets
        pred, _ = model(img, z= None, no_iter=5, val_mode=True)
        pred = pred.cpu().numpy().squeeze()
        pred = nib.Nifti1Image(pred, affine)
        nib.save(pred, "output.nii.gz") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./pretrained/skelite_3d/config.yaml", help='Path to the config file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/skelite_3d/check/model_best.pt", help='Path to the checkpoint file.')
    parser.add_argument('--image_path', type=str, default="demo_data/topcow_sample.nii.gz")
    parser.add_argument('--vis_only', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', help="outputs path")

    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)

    demo(config=config, 
        checkpoint_path=opts.checkpoint_path,
        image_path=opts.image_path,
        args_device=opts.device,
    )
