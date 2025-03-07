from model.soft_skeletonize import SoftSkeletonize
from model.skeleton_net import SkeletonNetwork
import torch
import torch.nn as nn

class ErosionSkeletonNet3D(SkeletonNetwork):

    def __init__(self, 
        kernel_size=[3,3], 
        num_iter=3, 
        **kwargs
    ):
        super(ErosionSkeletonNet3D, self).__init__(kernel_size=kernel_size, num_iter=num_iter, **kwargs)
        self.erode_module = SoftSkeletonize()
        self.show_intermediates = False
        self.beta = 0.5
        self.tau = 1
        self.use_channel_attention = True
        self.backbone=None
        self.conv = nn.Conv3d
        self.act = nn.ReLU
        self.final_act = nn.Sigmoid()
        
        self.final_conv = self.conv
        self.is_3d = True
        self.device = kwargs.get("device", "cuda")
        self.use_extraction_module = False
        self.extraction_module = None
        self.meta_mode = False

        self.skel_net = nn.Sequential(
            *self.build_net(
                in_channels=self.hypo_in_channels,
                kernel_size=self.kernel_size,
                conv=self.conv,
                act=self.act,
                final_conv=self.final_conv,
                meta_mode=self.meta_mode,
                depth=self.depth,
                is_3d=True
            )
        )

    def erosion_difference(self, img, num_iter=2):
        eroded = self.erode_module.iter_erode3D(img,num_iter=num_iter)
        bound = img - eroded
        return bound, eroded

    def dilate(self, img, n=2):
        return self.erode_module.iter_dilate3D(img, num_iter=n)

    def get_crossing_border(self, bound, center):
        dilated = self.erode_module.soft_dilate3D(bound)
        dilated_center = self.erode_module.soft_dilate3D(center)
        return (dilated * center) + (dilated_center * bound)

    def build_net(self, in_channels, kernel_size, conv, act, final_conv, depth=1, is_3d=False, **kwargs):
        """Builds a network pased on the depth provided"""
        layers = []
        pairs = self.generate_channel_pairs(depth=depth, in_channels=in_channels) 
        in_chan, out_chan = pairs[0]
        layers.append(conv(in_channels=in_chan,out_channels=out_chan,kernel_size=kernel_size,stride=1,padding=self.padding))
        layers.append(act())
        for (in_chan, out_chan) in pairs[1:-1]:
            layers.append(conv(in_channels=in_chan,out_channels=out_chan,kernel_size=kernel_size,stride=1,padding=self.padding))
            layers.append(act())
        in_chan, out_chan = pairs[-1]
        layers.append(final_conv(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=1, padding = 0))
        return layers
    
    def get_theta_0(self):
        return 0

    # Setter for theta_0, since we cannot use ParameterDict
    def set_theta_0(self, theta_0):
        self.theta_0 = 0

    def forward(self,img, z, val_mode=False, no_iter = None):

        if not no_iter:
            no_iter = self.num_iter

        center = img
        out = img.clone()
        for _ in range(no_iter):
            bound, inner = self.erosion_difference(center)
            bound_mask = (bound == 1)
            delta = self.skel_net(torch.cat((bound, center, out),dim=1))
            delta = self.final_act(delta) * bound_mask
            out -= delta
            center = inner 

        # Binarize when evaluating
        out = self.binarize_out(out,val_mode)
        return out, None