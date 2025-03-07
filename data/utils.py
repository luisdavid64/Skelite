import json
import os
import yaml
import torch
import yaml

def get_device():
    device = ("cuda" if torch.cuda.is_available() else
              ("mps" if torch.backends.mps.is_available() else "cpu"))
    return torch.device(device)


def get_config(config):
    if not config:
        return None
    if config.endswith(".json"):
        with open(config, 'r') as jf:
            return json.load(jf)
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)

def save_drive_copy(conf_path, folder_path):
    with open(conf_path, 'r') as file:
        config = yaml.safe_load(file)
        config['data'] = 'drive'
        config['data_root'] = 'datasets/drive_test'
    with open(os.path.join(folder_path, "config_drive.yaml"), 'w') as file:
        yaml.safe_dump(config, file)
    print("Saved drive copy at {}".format(folder_path))

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'check')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    results_directory = os.path.join(output_directory, 'results')
    if not os.path.exists(results_directory):
        print("Creating directory: {}".format(results_directory))
        os.makedirs(results_directory)
    return checkpoint_directory, image_directory, results_directory

def create_kernel_coords(dim, mn=-1, mx=1):
    """
    Created coordinate system with coil dimension

    Parameters:
    - dim (int): Dimensionality of the grid (2 or 3)
    - mn (float): Minimum coordinate value
    - mx (float): Maximum coordinate value

    Returns:
    - torch.Tensor: Coordinate grid
    """
    grid = None
    if len(dim) == 2: 
        W,H = dim
        Y, X = torch.meshgrid(torch.linspace(mn, mx, H),
                              torch.linspace(mn, mx, W))
        grid = torch.hstack((Y.reshape(-1, 1),
                             X.reshape(-1, 1)))
    elif len(dim) == 3: 
        D,W,H = dim
        Z, Y, X = torch.meshgrid(torch.linspace(mn, mx, D),
                                  torch.linspace(mn, mx, H),
                                  torch.linspace(mn, mx, W))
        grid = torch.hstack((Z.reshape(-1, 1),
                             Y.reshape(-1, 1),
                             X.reshape(-1, 1)))
    else:
        raise ValueError("Invalid value for 'dim'. Supported values are 2 or 3.")

    grid.requires_grad = True
    return grid