"""
---
title: Adam Optimizer for Half Precision Training
summary: A simple PyTorch implementation/tutorial of Adam optimizer
---

# Adam Optimizer for Half Precision Training
"""

from typing import Dict, Tuple, Optional, Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.cuda.amp import grad_scaler
from collections import defaultdict, abc

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.adam import Adam


class AdamFP16(Adam):
    """
    ## Adam Optimizer for Half Precision Training

    We extend [Adam Optimizer](adam.html) but use FP32 to store gradients and moments.
    """

    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-16,
                 weight_decay: WeightDecay = WeightDecay(), optimized_update: bool = True,
                 defaults: Optional[Dict[str, Any]] = None):
        # Parameter to store 32 bit gradients. This get populated by the `GradScaler` defined below.
        self.grad_fp32 = {}
        # Call the [Adam Optimizer](adam.html) initializer
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, defaults)

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        """
        ### Initialize a parameter state

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `param` is the parameter tensor $\theta_{t-1}$

        All the state tensors use FP32.
        """

        # This is the number of optimizer steps taken on the parameter, $t$
        state['step'] = 0
        # Exponential moving average of gradients, $m_t$
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format, dtype=torch.float)
        # Exponential moving average of squared gradient values, $v_t$
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format, dtype=torch.float)
        # Maintain a FP32 copy of the parameters
        state['fp32_copy'] = param.to(torch.float)

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        """
        ### Take an update step for a given parameter tensor

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor  $g_t$ for the parameter $\theta_{t-1}$
        * `param` is the parameter tensor $\theta_{t-1}$
        """

        # Get the FP32 parameters
        param_fp32 = state['fp32_copy']
        # Get the FP32 gradients if available
        grad_fp32 = self.grad_fp32.get(param, None)
        if grad_fp32 is not None:
            del self.grad_fp32[param]
            grad = grad_fp32
        else:
            # Otherwise, convert the gradients to FP32
            grad = grad.to(torch.float)

        # Calculate weight decay
        grad = self.weight_decay(param_fp32, grad, group)

        # Get $m_t$ and $v_t$
        m, v = self.get_mv(state, group, grad)

        # Increment $t$ the number of optimizer steps
        state['step'] += 1

        # Perform *Adam* update
        self.adam_update(state, group, param_fp32, m, v)

        # Set the parameters
        param.data = param_fp32.to(param.dtype)


class GradScalerFP16(grad_scaler.GradScaler):
    """
    ## Gradient Scaler with half precision gradients

    We extend PyTorch gradient scaler to use FP32 gradients.
    """

    def _unscale_grads_(self, optimizer: Optimizer, inv_scale: torch.Tensor, found_inf: torch.Tensor,
                        allow_fp16: bool) -> Dict[torch.device, torch.Tensor]:
        per_device_inv_scale = grad_scaler._MultiDeviceReplicator(inv_scale)
        per_device_found_inf = grad_scaler._MultiDeviceReplicator(found_inf)

        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]

        with torch.no_grad():
            # Loop through parameters
            for group in optimizer.param_groups:
                for param in group["params"]:
                    # Skip non-trainable parameters
                    if param.grad is None:
                        continue
                    # Not implemented for sparse tensors
                    if param.grad.is_sparse:
                        raise NotImplementedError

                    # If we are using the `AdamFP16` optimizer set `optimizer.grad_fp32[param]` to the FP32 gradients
                    if isinstance(optimizer, AdamFP16):
                        grad = param.grad.to(torch.float)
                        optimizer.grad_fp32[param] = grad
                    # Otherwise, do not convert the gradients to FP32
                    else:
                        grad = param.grad

                    per_device_and_dtype_grads[grad.device][grad.dtype].append(grad)

            # Unscale all the gradients
            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._amp_foreach_non_finite_check_and_unscale_(grads,
                                                                     per_device_found_inf.get(device),
                                                                     per_device_inv_scale.get(device))
        #
        return per_device_found_inf._per_device_tensors
