import torch
from torch import nn
from compress.heuristics import *

def reshape_filter(filter):
    """
    Rearrange a Conv2d weight tensor according to OSSCAR paper.

    Parameters
    ----------
    filter : torch.Tensor
        Weight tensor of shape (C_out, C_in, K_h, K_w).

    Returns
    -------
    torch.Tensor
        Reshaped tensor of shape (C_in*K_h*K_w, C_out).
    """
    assert isinstance(filter, torch.Tensor)
    assert filter.ndim==4, "Filter shape must be (Cout, Cin, Kh, Kw)"
    cout, _, _, _ = filter.size()
    reshaped_filter = filter.permute(1, 2, 3, 0)
    reshaped_filter = reshaped_filter.reshape(-1, cout)

    return reshaped_filter

def reshape_conv_layer_input(input, layer):
    """
    Unfold an input tensor using a Conv2d layer's settings.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (C, H, W) or (B, C, H, W).
    layer : nn.Conv2d
        Conv2d layer whose kernel/stride/dilation/padding define the unfolding.

    Returns
    -------
    torch.Tensor
        Unfolded tensor of shape (C_in*K_h*K_w, B*L),
        where L is the number of sliding locations.
    """
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    assert isinstance(layer, nn.Conv2d), "Layer must be a nn.Conv2d layer"
    assert input.ndim == 3 or input.ndim==3, "Input tensors must be either (C, H, W) or (B, C, H, W)"
    
    if input.ndim==3:
        input = input.unsqueeze(dim=0)
    
    _, _, h, w = input.shape

    # Effective size of a kernel changes in a dilated conv op
    k_eff_y = (layer.kernel_size[0] - 1) * layer.dilation[0] + 1
    k_eff_x = (layer.kernel_size[1] - 1) * layer.dilation[1] + 1

    if layer.padding == "same":
        y_padding = ((layer.stride[0] * h - h) + k_eff_y - layer.stride[0]) // 2
        x_padding = ((layer.stride[1] * w - w) + k_eff_x - layer.stride[1]) // 2    
    else:
        y_padding = x_padding = 0

    unfold = nn.Unfold(
                kernel_size=layer.kernel_size,
                dilation=layer.dilation,
                padding=(y_padding, x_padding),
                stride=layer.stride,
    )

    input = unfold(input)
    input = input.permute(1, 0, 2)
    input = input.flatten(1)

    return input
    
if __name__ == "__main__":
    rand_filter = torch.randn((8, 3, 3, 3))
    reshaped = reshape_filter(rand_filter)
    print(reshaped.size())
    unfold = nn.Unfold(
                kernel_size=3,
                dilation=1,
                padding=1,
                stride=1
    )
    inp = unfold(rand_filter)
    inp = inp.permute([1, 0, 2])
    inp = inp.flatten(1)

    print(inp.shape)

    assert inp.shape == reshaped.shape