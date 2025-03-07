import torch
from model.skeleton_net import ErosionSkeletonNet2D, SkeletonNet2D
from model.skeleton_net_threed import ErosionSkeletonNet3D


def get_model(model_type):
    if model_type == "reg_erosion":
        return ErosionSkeletonNet2D
    if model_type == "reg_erosion_3d":
        return ErosionSkeletonNet3D
    return SkeletonNet2D

def build_model(model_module, config, device):
    model = model_module(
        config['net']['kernel_size'],
        config['net']['skel_num_iter'],
        hyper_in_features=config['net']['network_input_size'],
        hyper_hidden_layers=config['net']['network_depth'],
        hyper_hidden_features=config['net']['network_width'],
        use_morphology_layer=config['net']['use_morphology_layer'],
        skel_channels=config['net']['skel_channels'],
        mip_style=config['net']['mip_style'],
        binarize=config['net']['binarize'],
        hypo_in_channels=config['hypo_net']['network_input_channels'],
        use_inception=config['net']['use_inception'],
        hypo_depth=config['hypo_net']['network_depth'],
        device=device,
        use_extraction_module=config['net']["use_extraction_module"],
        extract_feats=config['net']['extract_feats']
    )
    return model

def load_model_checkpoint(model, checkpoint_path, device):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device=device))
        model.load_state_dict(checkpoint["net"])
    return model