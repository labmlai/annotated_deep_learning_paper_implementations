import torch


def weight_standardization(weight: torch.Tensor, eps: float):
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    std, mean = torch.std_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (std + eps)
    return weight.view(c_out, c_in, *kernel_shape)
