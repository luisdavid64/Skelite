import torch
import torch.nn as nn
from model.soft_skeletonize import SoftSkeletonize
import torch.nn as nn

class SkeletonNetwork(nn.Module):

    def __init__(self, kernel_size, num_iter, *args, **kwargs) -> None:
        super(SkeletonNetwork, self).__init__()
        self.kernel_size = kernel_size
        self.num_iter = num_iter
        self.calculate_padding()
        self.skel_channels = int(kwargs.get("skel_channels",128))
        self.hypo_in_channels = kwargs.get('hypo_in_channels', 1)
        self.depth = kwargs.get('hypo_depth', 1)
        self.bin_threshold = 0.5 

    def binarize_out(self, out, val_mode):
        if val_mode:
            out = (out.detach() > self.bin_threshold).float() - out.detach() + out
        return out

    def set_bin_threshold(self, thresh):
        self.bin_threshold = thresh
    
    def get_theta_0(self):
        return 0

    def get_no_iters(self):
        return self.num_iter

    def set_no_iters(self, num_iter):
        self.num_iter = num_iter

    # Setter for theta_0, since we cannot use ParameterDict
    def set_theta_0(self, theta_0):
        pass

    def calculate_padding(self):
        self.padding = (self.kernel_size[0] - 1) // 2
    
    def generate_channel_pairs(self, depth, in_channels):
        """Generates a UNet like pyramid of channel pairs"""
        possible_nums = [self.skel_channels // (2**(depth - (i+1))) for i in range(depth)]
        possible_nums.extend(possible_nums[:-1][::-1])
        pairs = []
        for i in range(len(possible_nums)):
            if i == 0: 
                pairs.append([in_channels, possible_nums[i]])
            else:
                pairs.append([possible_nums[i - 1], possible_nums[i]])
        pairs.append([possible_nums[-1], 1])
        return pairs

    def build_net(self, in_channels, kernel_size, conv, act, final_conv, depth=1, **kwargs):
        """Builds a network pased on the depth provided"""
        layers = []
        pairs = self.generate_channel_pairs(depth=depth, in_channels=in_channels) 
        for (in_chan, out_chan) in pairs[:-1]:
            layers.append(conv(in_channels=in_chan,out_channels=out_chan,kernel_size=kernel_size,stride=1,padding=self.padding, **kwargs))
            layers.append(act())
        in_chan, out_chan = pairs[-1]
        layers.append(final_conv(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=1, padding = 0, **kwargs))
        return layers
    


    def forward_filters(self, z=None):
        z = z.unsqueeze(0)
        predicted_kernel = self.hyper_net(z)
        return predicted_kernel

class SkeletonNet2D(SkeletonNetwork):
    
    def __init__(self, 
                kernel_size=[3,3], 
                num_iter=3, 
                **kwargs
    ):
        super(SkeletonNet2D, self).__init__(kernel_size=kernel_size, num_iter=num_iter, **kwargs)
        # Coords of convolutional kernel

        self.conv = nn.Conv2d
        self.act = nn.ReLU
        self.final_act = nn.Sigmoid()

        self.final_conv = self.conv
        # Create Skeletonization Step
        self.skel_net = nn.Sequential(
            *self.build_net(
                in_channels=self.hypo_in_channels,
                kernel_size=self.kernel_size,
                conv=self.conv,
                act=self.act,
                final_conv=self.final_conv,
                depth=self.depth
            )
        )

    def forward(self,img, z=None, val_mode=False, no_iter=None):
        
        if no_iter == None:
            no_iter = self.num_iter

        out = img 
        for _ in range(no_iter):
            out = self.skel_net(out)
        out = self.final_act(out)

        # Binarize when evaluating
        out = self.binarize_out(out,val_mode)
        return out, None
    
class ErosionSkeletonNet2D(SkeletonNet2D):

    def __init__(self, 
        kernel_size=[3,3], 
        num_iter=3, 
        **kwargs
    ):
        """TODO: Find a proper way to use this model"""
        super(ErosionSkeletonNet2D, self).__init__(kernel_size=kernel_size, num_iter=num_iter, **kwargs)
        self.erode_module = SoftSkeletonize()

    def erosion_difference(self, img):
        eroded = self.erode_module.iter_erode(img, num_iter=2)
        bound = img - eroded
        return bound, eroded

    def get_crossing_border(self, bound, center):
        dilated = self.erode_module.soft_dilate(bound)
        dilated_center = self.erode_module.soft_dilate(center)
        return (dilated * center) + (dilated_center * bound)

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
