import imageio
import os
import torch
import nibabel as nib

def add_loss_data(dict1, dict2):
    if not dict1:
        return dict2.copy() 
    result_dict = dict1.copy() 
    
    for key, value in dict2.items():
        if key in result_dict:
            result_dict[key] += value
        else:
            result_dict[key] = value
    
    return result_dict

def add_dict_data(dict1,dict2):
    return add_loss_data(dict1,dict2)

def batch_scale_loss(dict, length, prepend=""):
    result_dict = {}
    for key, value in dict.items():
        if prepend != "":
            key = "{}_{}".format(prepend,key)
        result_dict[key] = value / length
    
    return result_dict

def scale_dict_data(dict,length):
    return batch_scale_loss(dict,length)
    
def print_loss(loss_dict, epoch, phase="Train", flush=False):
    print_str = "{} Epoch: {} ".format(phase, epoch)
    for key, val in loss_dict.items():
        print_str = "{} | {}: {}".format(print_str, key, val)
    print(print_str, flush=flush)

def load_weights(model, checkpoint_dir):
    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_files = [f for f in checkpoint_files if f.endswith('.pt')]
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]

    checkpoint_files.sort(key=os.path.getmtime)
    if not len(checkpoint_files):
        return None
    last_checkpoint = checkpoint_files[-1] if checkpoint_files else None
    checkpoint = torch.load(last_checkpoint, map_location=torch.device(device="cpu"))
    model.load_state_dict(checkpoint["net"])
    kernel_coords = checkpoint["noise_vec"].to(device="cpu")
    if "theta_0" in checkpoint:
        model.set_theta_0(checkpoint["theta_0"])
    del checkpoint
    torch.cuda.empty_cache()
    return kernel_coords

def load_image(path):
    img = imageio.imread(path) / 255.
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img

def load_nii(path):
    img = nib.load(path)
    affine = img.affine
    img = img.get_fdata().squeeze()
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).unsqueeze(0)
    return img, affine