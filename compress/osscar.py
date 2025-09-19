import random
from typing import cast

import numpy as np
import torch
from torch import nn

from compress.heuristics import *
from model.resnet import resnet50
from data.load_data import build_eval_dataset, build_eval_dataloader


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
    assert filter.ndim == 4, "Filter shape must be (Cout, Cin, Kh, Kw)"
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
        Unfolded tensor of shape (B*L, C_in*K_h*K_w),
        where L is the number of sliding locations.
    """
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    assert isinstance(layer, nn.Conv2d), "Layer must be a nn.Conv2d layer"
    assert (
        input.ndim == 3 or input.ndim == 4
    ), "Input tensors must be either (C, H, W) or (B, C, H, W)"

    if input.ndim == 3:
        input = input.unsqueeze(dim=0)

    _, _, h, w = input.shape

    # Effective size of a kernel changes in a dilated conv op
    k_eff_y = (layer.kernel_size[0] - 1) * layer.dilation[0] + 1
    k_eff_x = (layer.kernel_size[1] - 1) * layer.dilation[1] + 1

    if isinstance(layer.padding, str) and layer.padding == "same":
        y_padding = ((layer.stride[0] * h - h) + k_eff_y - layer.stride[0]) // 2
        x_padding = ((layer.stride[1] * w - w) + k_eff_x - layer.stride[1]) // 2
    elif isinstance(layer.padding, tuple):
        y_padding, x_padding = layer.padding
    else:
        y_padding = x_padding = layer.padding

    # Silence pylance's static check
    y_padding = cast(int, y_padding)
    x_padding = cast(int, x_padding)

    unfold = nn.Unfold(
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=(y_padding, x_padding),
        stride=layer.stride,
    )

    input = unfold(input)
    input = input.permute(1, 0, 2)
    input = input.flatten(1)
    input = input.T

    return input


def get_coeff_h(design_matrix):
    """
    Compute the input autocorrelation (H) matrix from a 2D design matrix.

    Parameters
    ----------
    design_matrix : torch.Tensor
        2D tensor of shape (N, D), where N is the number of samples
        (e.g., unfolded spatial positions across the batch) and
        D is the feature dimension (e.g., C_in * K_h * K_w).

    Returns
    -------
    torch.Tensor
        Square tensor of shape (D, D) representing the autocorrelation
        matrix H = XᵀX of the input features.
    """
    assert isinstance(design_matrix, torch.Tensor)
    assert design_matrix.ndim == 2, "Requires the reshaped design matrix"

    return design_matrix.T @ design_matrix


def get_XtY(X, Y):
    """
    Compute the cross-correlation matrix XᵀY between two unfolded tensors.

    Parameters
    ----------
    X : torch.Tensor
        2D tensor of shape (N, D₁), typically an unfolded input/design matrix
        from `reshape_conv_layer_input`, where N is the number of samples
        (e.g., batch × sliding locations) and D₁ is the feature dimension.
    Y : torch.Tensor
        2D tensor of shape (N, D₂), typically another unfolded tensor of
        the same number of rows as `X`, but possibly with a different
        feature dimension D₂.

    Returns
    -------
    torch.Tensor
        Matrix of shape (D₁, D₂) representing the cross-correlation
        XᵀY between the two inputs.
    """
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)

    A = X.T

    assert A.shape[1] == Y.shape[0]
    return A @ Y


def get_coeff_g(dense_layer_weights, layer_input):
    """
    Compute the G coefficient for a dense (reshaped) layer weight matrix.

    Parameters
    ----------
    dense_layer_weights : torch.Tensor
        Layer weights reshaped to 2D, shape (out_features, in_features).
    layer_input : torch.Tensor
        Layer input activations before non-linearity.
        Shape (B, C, H, W) or (C, H, W).

    Returns
    -------
    torch.Tensor
        G matrix capturing the projection of inputs onto the weight space.
    """
    assert isinstance(layer_input, torch.Tensor)
    assert isinstance(dense_layer_weights, torch.Tensor)

    layer_input_dim = layer_input.ndim

    assert dense_layer_weights.ndim == 2, "get_coeff_g takes in the reshaped weights"
    assert (
        layer_input_dim == 3 or layer_input_dim == 4
    ), "Layer input must be of shape (B, C, H, W) or (C, H, W)"

    G = torch.transpose(dense_layer_weights, dim0=0, dim1=1) @ layer_input

    return G
    
def compute_layer_loss(dense_weights, pruned_weights, input):
    """
    Compute the layer reconstruction loss using H and G coefficients.

    Parameters
    ----------
    dense_weights : torch.Tensor
        Original (dense) layer weights reshaped to 2D.
    pruned_weights : torch.Tensor
        Pruned layer weights reshaped to 2D.
    input : torch.Tensor
        Layer input activations before non-linearity.
        Shape (B, C, H, W) or (C, H, W).

    Returns
    -------
    torch.Tensor
        Scalar loss measuring how well the pruned layer reconstructs the original.
    """
    G = get_coeff_g(dense_layer_weights=dense_weights, layer_input=input)
    H = get_coeff_h(design_matrix=input)
    A = (pruned_weights.T @ H) @ pruned_weights
    B = G.T @ pruned_weights

    assert A.ndim == B.ndim == 2, "Trace can be computed only for 2D matrices"
    loss = 0.5 * torch.trace(A) + torch.trace(B)

    return loss


def num_params_in_prune_channels(layers):
    """
    Compute the total number of parameters across a list of convolutional layers.

    Parameters
    ----------
    layers : list of nn.Conv2d
        The convolutional layers whose parameters you want to count.

    Returns
    -------
    int
        Total number of parameters (weights + biases if present) in the given layers.
    """
    params = 0

    for layer in layers:
        assert isinstance(layer, nn.Conv2d)
        params += count_parameters(layer, in_millions=False)

    return params


def recalculate_importance(rem_channels_layer_wise):
    """
    Recalculate normalized importance weights for each layer
    based on the remaining number of channels per layer.

    Parameters
    ----------
    rem_channels_layer_wise : array-like of int
        Number of channels remaining in each layer.

    Returns
    -------
    numpy.ndarray
        Normalized importance values for each layer (sum to 1).
    """
    total = np.sum(rem_channels_layer_wise)
    rem_imp = np.divide(rem_channels_layer_wise, total)

    return rem_imp


def distribute_remaining_parameters(
    rem_params, rem_channels_per_layer, layers, num_iters=20, allowable_tol=250
):
    """
    Stochastically allocate leftover parameter removals across layers.

    At each iteration, a layer is sampled according to a probability
    distribution proportional to its remaining channels. One input channel
    is removed from the chosen layer if possible, and the remaining parameter
    budget is updated. The loop stops when the budget is within the allowable
    tolerance or after `num_iters` iterations.

    Parameters
    ----------
    rem_params : int
        Remaining number of parameters to remove.
    rem_channels_per_layer : list of int
        Remaining number of channels per layer (mutable, will be updated).
    layers : list of nn.Conv2d
        Convolutional layers eligible for pruning.
    num_iters : int, optional
        Maximum number of allocation iterations. Default is 20.
    allowable_tol : int, optional
        Stop when the remaining parameter budget is within this tolerance. Default is 250.

    Returns
    -------
    tuple
        p : numpy.ndarray
            Array of additional channels removed per layer.
        rem_params : int
            Remaining number of parameters still to remove after allocation.
    """
    num_layers = len(layers)
    layer_choices = np.arange(num_layers)
    p = np.zeros(num_layers, dtype=np.int32)
    rng = random.Random(3)

    rem_imp = recalculate_importance(rem_channels_layer_wise=rem_channels_per_layer)

    for i in range(num_iters):
        random_layer_idx = rng.choices(layer_choices, weights=rem_imp)[0]
        assert isinstance(random_layer_idx, (int, np.integer))

        layer = layers[random_layer_idx]

        assert isinstance(layer, nn.Conv2d)
        params_removed = (
            layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        )
        count_remove_params = rem_params - params_removed
        if rem_channels_per_layer[random_layer_idx] > 1 and count_remove_params >= 0:
            p[random_layer_idx] += 1
            rem_channels_per_layer[random_layer_idx] -= 1

            if rem_params - count_remove_params < 0:
                continue

            rem_params = count_remove_params
            rem_imp = recalculate_importance(rem_channels_per_layer)

        if rem_params <= allowable_tol:
            break

    return p, rem_params


def get_count_prune_channels(model, prune_percentage, allowable_tol=250):
    """
    Compute per-layer channel pruning counts to reach a target global prune percentage.

    This function:
      1. Collects the convolutional layers eligible for pruning.
      2. Computes how many parameters to remove from each layer based on an
         importance heuristic.
      3. Converts parameter removals into channel counts per layer.
      4. Distributes any remaining parameter removals stochastically to meet
         the target budget within the allowable tolerance.

    Parameters
    ----------
    model : nn.Module
        The model to prune.
    prune_percentage : float
        Target fraction of total model parameters to remove (0–1).
    allowable_tol : int, optional
        Tolerance for how far from the target parameter count to allow. Default is 250.

    Returns
    -------
    tuple
        num_channels_left_per_layer : list of int
            Number of input channels to keep per layer after pruning.
        p : list or numpy.ndarray
            Number of input channels to prune per layer (sum of deterministic and random).
        remaining_params_to_prune : int
            Number of parameters still to prune after allocation (ideally <= allowable_tol).
    """
    model_total_params = count_parameters(model, in_millions=False)

    # Collect layers eligible to be pruned
    layers, _ = collect_convolution_layers_to_prune(model)

    # Quick sanity check : ensures that total params to be removed doesnt exceed those that can be provided by eligible layers
    eligible_prune_params_count = num_params_in_prune_channels(layers=layers)
    total_params_to_prune = int(model_total_params * prune_percentage)

    assert eligible_prune_params_count > total_params_to_prune

    # Computes relative importance of each layer, higher importance => More channels pruned from this layer
    importance_list = compute_layer_importance_heuristic(layers)

    assert math.isclose(
        np.sum(importance_list), 1.0, abs_tol=0.00001
    ), "importance scores must sum to 1"

    # Number of params to remove from every eligible layer
    num_prune_params_by_layer = total_params_to_prune * importance_list
    num_prune_params_by_layer = np.floor(total_params_to_prune * importance_list)

    revised_prune_params_count = 0
    p = []
    num_channels_left_per_layer = []

    for idx, layer in enumerate(layers):
        assert isinstance(layer, nn.Conv2d)

        num_spatial_params = layer.kernel_size[0] * layer.kernel_size[1]
        num_channels_per_filter = layer.in_channels
        num_filters = layer.out_channels

        # Num channels to remove from every player(defined as per osscar)
        num_channels_to_remove = num_prune_params_by_layer[idx] // (
            num_spatial_params * num_filters
        )
        p.append(num_channels_to_remove)

        assert (
            num_channels_to_remove < num_channels_per_filter
        ), "Cant remove all channels in a filter"

        num_params_removed = num_spatial_params * num_channels_to_remove * num_filters
        revised_prune_params_count += num_params_removed
        num_channels_left = num_channels_per_filter - num_channels_to_remove
        num_channels_left_per_layer.append(num_channels_left)

    remaining_params_to_prune = total_params_to_prune - revised_prune_params_count

    if remaining_params_to_prune > allowable_tol:
        p_rem, remaining_params_to_prune = distribute_remaining_parameters(
            rem_params=remaining_params_to_prune,
            rem_channels_per_layer=num_channels_left_per_layer,
            layers=layers,
            allowable_tol=allowable_tol,
        )

        p = np.array(p) + np.array(p_rem)

    return num_channels_left_per_layer, p, remaining_params_to_prune

def save_layer_input(activations, layer_name):
    """
    Factory function to create a forward hook that saves a layer’s output.

    Parameters
    ----------
    activations : dict
        A dictionary (mutable) where the captured outputs will be stored.
        Keys are layer names, values are the output tensors.
    layer_name : str
        Name under which to store this layer’s output in the `activations` dict.

    Returns
    -------
    hook : callable
        A forward hook function with signature (module, input, output)
        that can be passed to `register_forward_hook`.
    """
    def hook(model, input, output):
        activations[layer_name] = output.detach()
        return hook


def register_hooks_to_collect_outs(prune_modules, prune_module_names, hook_fn):
    """
    Register a forward hook on each module in `prune_modules` to collect outputs.

    Parameters
    ----------
    prune_modules : list[nn.Module]
        List of modules to attach hooks to (e.g. layers to prune).
    prune_module_names : list[str]
        Names corresponding to each module in `prune_modules`.
        Must be the same length as `prune_modules`.
    hook_fn : callable
        A factory function that accepts `activations` (dict) and `layer_name` (str)
        and returns a forward hook with signature (module, input, output).

    Returns
    -------
    activations : dict
        A dictionary that will be populated with {layer_name: output_tensor}
        during a forward pass.
    """
    activations = {}
    for idx, module in enumerate(prune_modules):
        assert isinstance(module, nn.Conv2d)
        module_name = prune_module_names[idx]
        module.register_forward_hook(hook=hook_fn(activations=activations, layer_name=module_name))
    
    return activations

def recompute_X(prune_mask, X, layer_in_channels, kernel_height, kernel_width):
    """
    Recompute the unfolded input matrix X after pruning input channels.

    Parameters
    ----------
    prune_mask : array-like of bool
        Boolean mask of length `layer_in_channels` indicating which input channels to keep (`True`) or drop (`False`).
    X : torch.Tensor or np.ndarray
        2D unfolded input matrix of shape (N, M) where N corresponds to flattened channel–kernel elements and
        M to the number of sliding positions or samples.
    layer_in_channels : int
        Number of input channels in the convolution layer prior to pruning.
    kernel_height : int
        Height of the convolution kernel.
    kernel_width : int
        Width of the convolution kernel.

    Returns
    -------
    torch.Tensor or np.ndarray
        Pruned unfolded input matrix with rows corresponding only to kept input channels.
    """
    assert X.ndim == 2, "Weight matrix must have already been reshaped"
    assert len(prune_mask) == layer_in_channels, "The length of the indicator vector and the number of in_channels must be the same"

    # Number of elements needed to represent one filter in_channel in the reshaped 2d weight matrix 
    numel_one_channel = kernel_height * kernel_width
    _, X_width = X.shape

    slice_indices = np.arange(layer_in_channels)
    mask = np.ones(X_width, dtype=bool)
    slice_indices = [(start * numel_one_channel, start * numel_one_channel + numel_one_channel) for start in slice_indices]
    
    for idx, indicator in enumerate(prune_mask):
        if not indicator:
            start, stop = slice_indices[idx]
            mask[start:stop] = False
    
    return X[:, mask]
    
def recompute_H(prune_mask, H, kernel_height, kernel_width, activation_out_channels):
    """
    Recompute the coefficient matrix H after pruning output channels.

    Parameters
    ----------
    prune_mask : array-like of bool
        Boolean mask of length `activation_out_channels` indicating which output channels to keep (`True`) or drop (`False`).
    H : torch.Tensor or np.ndarray
        2D square matrix of shape (N, N) typically representing a Hessian or covariance term over activations.
    kernel_height : int
        Height of the convolution kernel.
    kernel_width : int
        Width of the convolution kernel.
    activation_out_channels : int
        Number of output channels (activations) prior to pruning.

    Returns
    -------
    torch.Tensor or np.ndarray
        Pruned square matrix containing only rows and columns corresponding to kept output channels.
    """
    assert len(prune_mask) == activation_out_channels
    assert H.ndim == 2

    kept_indices = []
    numel_one_channel = kernel_height * kernel_width
    slice_indices = np.arange(activation_out_channels)
    slice_indices = [(start * numel_one_channel, start * numel_one_channel + numel_one_channel) for start in slice_indices]
    
    for idx, indicator in enumerate(prune_mask):
        if not indicator:
            start, stop = slice_indices[idx]
            kept_indices.extend(np.arange(start=start, stop=stop))

    H_updated = H[np.ix_(kept_indices, kept_indices)]

    return H_updated
    
def recompute_W(prune_mask, W, layer_in_channels, kernel_height, kernel_width):
    """
    Recompute the weight matrix W after pruning input channels.

    Parameters
    ----------
    prune_mask : array-like of bool
        Boolean mask of length `layer_in_channels` indicating which input channels to keep (`True`) or drop (`False`).
    W : torch.Tensor or np.ndarray
        2D weight matrix of shape (N, M) where N corresponds to flattened channel–kernel elements.
    layer_in_channels : int
        Number of input channels in the convolution layer prior to pruning.
    kernel_height : int
        Height of the convolution kernel.
    kernel_width : int
        Width of the convolution kernel.

    Returns
    -------
    torch.Tensor or np.ndarray
        Pruned weight matrix with rows corresponding only to kept input channels.
    """
    assert W.ndim == 2, "Weight matrix must have already been reshaped"
    assert len(prune_mask) == layer_in_channels, "The length of the indicator vector and the number of in_channels must be the same"

    # Number of elements needed to represent one filter in_channel in the reshaped 2d weight matrix 
    numel_one_channel = kernel_height * kernel_width
    W_height, _ = W.shape

    slice_indices = np.arange(layer_in_channels)
    mask = np.ones(W_height, dtype=bool)

    slice_indices = [(start * numel_one_channel, start * numel_one_channel + numel_one_channel) for start in slice_indices]
    print(slice_indices)

    for idx, indicator in enumerate(prune_mask):
        if not indicator:
            start, stop = slice_indices[idx]
            mask[start:stop] = False
    
    return W[mask, :]


def get_optimal_W(pruned_activation, dense_activation, dense_weights):
    assert isinstance(pruned_activation, torch.Tensor)
    assert isinstance(dense_activation, torch.Tensor)
    assert isinstance(dense_weights, torch.Tensor)

    assert pruned_activation.ndim == 2, "Pruned activation should be 2D"
    assert dense_activation.ndim == 2, "Dense activation should have been reshaped to 2D"
    assert dense_weights.ndim == 2, "Weights should have been reshaped to 2D"

def get_calibration_dataset():


if __name__ == "__main__":
    # rand_filter = torch.randn((8, 3, 3, 3))
    # reshaped = reshape_filter(rand_filter)
    # print(reshaped.size())
    # unfold = nn.Unfold(
    #             kernel_size=3,
    #             dilation=1,
    #             padding=1,
    #             stride=1
    # )
    # inp = unfold(rand_filter)
    # inp = inp.permute([1, 0, 2])
    # inp = inp.flatten(1)

    # # print(inp.shape)
    # model = resnet50(pretrained=True)
    # overall_prune_percentage = 0.3
    # # _, p, rem_params = get_count_prune_channels(
    # #     model=model, prune_percentage=overall_prune_percentage
    # # )

    # # print(f"P: {p}")
    # # print(f"Rem parameters to prune: {rem_params}")
    # layer_in_channels = 3
    # numel_one_channel = 9
    # slice_indices = np.arange(layer_in_channels, dtype=np.int32)
    # slice_indices = [(start * numel_one_channel, start * numel_one_channel + numel_one_channel - 1) for start in slice_indices]
    # print(slice_indices)
    

    # z = np.random.randint(low=0, high=2, size=layer_in_channels)
    # print("Z:", z)
    # W = np.random.randn(27, 1)
    # print("W before: ", W)
    # print("W after: ", get_matrix_I(z, W, layer_in_channels, 3, 3))
    
    # layer_input = torch.randn((3, 3, 4))
    # output = get_coeff_h(layer_input)
    # print(output.shape)

    input_activation = torch.randn(2, 3, 32, 32)
    pruned_activation = torch.randn(2, 2, 32, 32)
    mini_conv = nn.Conv2d(padding=1, kernel_size=3, in_channels=3, out_channels=8)
    
    out = reshape_conv_layer_input(input_activation, mini_conv)

    W = reshape_filter(mini_conv.weight)
    print(f"Reshaped W before pruning: {W.shape}")

    prune_mask = [1, 1, 0]
    W_pruned = recompute_W(prune_mask, W, layer_in_channels=3, kernel_height=3, kernel_width=3)

    print(f"Reshaped W after pruning: {W_pruned.shape}")
    
    print("Reshaped activation shape before pruning:", out.shape)

    X = recompute_X(prune_mask=prune_mask, X=out, layer_in_channels=3, kernel_height=3, kernel_width=3)

    print("Reshaped activation shape after pruning:", X.shape)

    corr = get_XtY(out, X)
    print("XtY shape", corr.shape)

    print(get_coeff_h(out).shape)