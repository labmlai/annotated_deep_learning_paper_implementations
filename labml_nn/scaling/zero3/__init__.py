import functools
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn


class Zero3Layer(nn.Module):
    # Current nodes chunk is split into a list, for different types.
    # In this case first is the trainable parameters and the next is non-trainable
    TRAINING_PARAMS_IDX = 0

    param_refs: List[List[nn.Parameter]]

    chunk: List[nn.Parameter]
    chunk_size: List[int]

    fetch_stream: Optional[torch.cuda.Stream]
    backup_stream: Optional[torch.cuda.Stream]
    prev_layer: List['Zero3Layer']
    next_layer: List['Zero3Layer']

    device: torch.device
    dtype: torch.dtype
    module: nn.Module
    # Whether parameters are fetched
    is_fetched: bool
    world_size: int
    # Used this for debugging logs
    layer_idx: int

    def __init__(self, module: nn.Module, rank: int, world_size: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.module = module
        self.prev_layer = []
        self.next_layer = []
        self.is_fetched = False
        self.world_size = world_size
        self.layer_idx = -1
        self.fetch_stream = None
        self.backup_stream = None

        with torch.no_grad():
            # Collect all the parameters of the layer
            all_param_refs = [p for p in self.parameters()]
            for p in all_param_refs:
                p._orig_shape = p.shape

            # All parameters should have the same type
            for p in all_param_refs:
                assert p.dtype == dtype, "All parameters should have same dtype"

            # Separate parameters as trainable and fixed
            self.param_refs = [[p for p in all_param_refs if p.requires_grad],
                               [p for p in all_param_refs if not p.requires_grad]]

            # **Calculate the size each device/node should store**
            if rank == 0:
                # Merge and pad parameters
                merged_params = [self._merge_and_pad_params(ps) for ps in self.param_refs]
                # Calculate the chunk sizes for trainable and fixed params
                self.chunk_size = [(len(p) // world_size if p is not None else 0) for p in merged_params]
                # Broadcast the sizes
                dist.broadcast(torch.tensor(self.chunk_size, device=device), src=0)
            else:
                # Create an empty tensor to receive the sizes
                chunk_size = torch.tensor([0, 0], device=device)
                # Receive the sizes
                dist.broadcast(chunk_size, src=0)
                #
                self.chunk_size = chunk_size.tolist()

            # Create parameters for trainable and fixed-params to be stored in current device/node
            self.chunk = [nn.Parameter(self._empty((s,)), requires_grad=i == self.TRAINING_PARAMS_IDX)
                          for i, s in enumerate(self.chunk_size)]

            # An empty tensor to receive the parameters
            chunk = self._empty((sum(self.chunk_size),))

            if rank == 0:
                # Concatenate trainable and fixed params
                all_params = torch.cat([p.view(world_size, -1) for p in merged_params], dim=-1).view(-1)
                del merged_params

                # Scatter them to all the nodes/devices
                dist.scatter(chunk, list(all_params.split(sum(self.chunk_size))))
                del all_params
            else:
                # Receive the parameters
                dist.scatter(chunk)

            # Collect the chunk data
            chunk = chunk.split(self.chunk_size)
            for i, c in enumerate(chunk):
                self.chunk[i].data = c
            del chunk

            # Cleanup the normal parameters
            self._cleanup_params()

            # Add a backward hook
            self._backward_hook_ref = self.register_full_backward_hook(self._backward_hook)  # type: ignore

    def _merge_and_pad_params(self, params: List[nn.Parameter]) -> torch.Tensor:
        """
        Merge all the parameters and pad it so that it's divisible by `world_size`.
        """
        size = sum(p.shape.numel() for p in params)

        if size % self.world_size != 0:
            padding_fixed = self.world_size - (size % self.world_size)
        else:
            padding_fixed = 0

        padding = self._empty((padding_fixed,))
        return torch.cat([p.view(-1) for p in params] + [padding], dim=0)

    def get_trainable_chunk(self):
        if len(self.chunk[self.TRAINING_PARAMS_IDX]) == 0:
            return []

        return [self.chunk[self.TRAINING_PARAMS_IDX]]

    def _empty(self, shape):
        return torch.empty(shape, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def fetch_params(self):
        """Load parameters"""
        if self.is_fetched:
            return

        self.is_fetched = True

        if sum(self.chunk_size) == 0:
            return

        with torch.cuda.stream(self.fetch_stream):
            buffer = self._empty((self.world_size * sum(self.chunk_size),))
            buffers = list(buffer.split(sum(self.chunk_size)))

            chunk = torch.cat(self.chunk, dim=0)
            dist.all_gather(buffers, chunk)

            # Split by type
            params = buffer.view(-1, sum(self.chunk_size)).split(self.chunk_size, dim=1)
            buffer.record_stream(self.fetch_stream)
            for b in buffers:
                b.record_stream(self.fetch_stream)
            buffer.record_stream(self.fetch_stream)
            del buffer
            del buffers
            params = [p.reshape(-1) for p in params]

            # Put them to params
            for cont, ps in zip(params, self.param_refs):
                if not ps:
                    continue

                offset = 0
                for p in ps:
                    shape = p._orig_shape  # type: ignore[attr-defined]
                    p.data.storage().resize_(shape.numel())
                    p.data[:] = cont[offset: offset + shape.numel()].reshape(shape)
                    p.data.record_stream(torch.cuda.current_stream())
                    offset += shape.numel()

                cont.record_stream(self.fetch_stream)

            # torch.cuda.current_stream().wait_stream()

            del params

    @torch.no_grad()
    def _cleanup_params(self):
        """Empty the parameter data"""
        self.is_fetched = False

        for ps in self.param_refs:
            for p in ps:
                p.data.record_stream(torch.cuda.current_stream())
                assert p.data.storage_offset() == 0, "The tensor is not the sole occupant of the storage."
                p.data.storage().resize_(0)  # This is what actually clears the memory
                assert p.grad is None, 'Gradients should be None'

    def _add_backward_hooks(self):
        self._backward_hook_handles = 0
        for p in self.parameters():
            if hasattr(p, "_hook_handle"):
                continue
            if not p.requires_grad:
                continue
            p_tmp = p.expand_as(p)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            handle = grad_acc.register_hook(
                functools.partial(self._post_backward_hook, p))
            p._hook_handle = (grad_acc, handle)
            self._backward_hook_handles += 1

    def _backward_event(self):
        self._backward_hook_handles -= 1

        if self._backward_hook_handles == -1:
            self._backup_grads()
            self._cleanup_params()

        # Previous layer get ready
        for layer in self.prev_layer:
            layer.fetch_params()

    def _post_backward_hook(self, p: nn.Parameter, *args):
        p._hook_handle[1].remove()  # type: ignore[attr-defined]
        delattr(p, "_hook_handle")

        self._backward_event()

    def _backward_hook(self, *args, **kwargs):
        self._backward_event()

        # The previous layer will start computing gradients. We need to make sure it has finished fetching params
        torch.cuda.current_stream().wait_stream(self.fetch_stream)

        return None

    def forward(self, *args, **kwargs):
        if not self.prev_layer:
            self.fetch_params()

        assert self.is_fetched, "Parameters not fetched before forward pass"
        # TWe need to make sure we have finished fetching params
        torch.cuda.current_stream().wait_stream(self.fetch_stream)

        for layer in self.next_layer:
            layer.fetch_params()

        if torch.is_grad_enabled():
            self._add_backward_hooks()

        res = self.module(*args, **kwargs)

        # Do not cleanup if this is the last layer
        if not torch.is_grad_enabled() or self.next_layer:
            self._cleanup_params()

        return res

    @torch.no_grad()
    def _backup_grads(self):
        if self.chunk_size[self.TRAINING_PARAMS_IDX] == 0:
            return

        with torch.cuda.stream(self.backup_stream):
            """Accumulate gradients into respective nodes"""
            buffer = self._empty((self.world_size * self.chunk_size[self.TRAINING_PARAMS_IDX],))
            buffers = list(buffer.split(self.chunk_size[self.TRAINING_PARAMS_IDX]))

            # Collect gradients
            offset = 0
            for p in self.param_refs[self.TRAINING_PARAMS_IDX]:
                shape = p._orig_shape  # type: ignore[attr-defined]
                buffer[offset: offset + shape.numel()] = p.grad.view(-1)
                offset += shape.numel()
                p.grad = None

            grad = self._empty((self.chunk_size[self.TRAINING_PARAMS_IDX],))
            dist.reduce_scatter(grad, buffers)
            for b in buffers:
                b.record_stream(self.fetch_stream)
            buffer.record_stream(self.fetch_stream)
            del buffer
            del buffers

            # DEBUG
            self.chunk[self.TRAINING_PARAMS_IDX].grad = grad  # .clone()
            del grad


class Zero3Sequential(nn.Module):
    def __init__(self, modules: List[Zero3Layer]):
        super().__init__()

        self.fetch_stream = torch.cuda.Stream()
        self.backup_stream = torch.cuda.Stream()

        for i in range(len(modules)):
            modules[i].layer_idx = i
            modules[i].fetch_stream = self.fetch_stream
            modules[i].backup_stream = self.backup_stream
            if i + 1 < len(modules):
                modules[i].next_layer.append(modules[i + 1])
            if i - 1 >= 0:
                modules[i].prev_layer.append(modules[i - 1])

        self.module_list = nn.ModuleList(modules)

    def get_trainable_chunk(self):
        return sum([m.get_trainable_chunk() for m in self.module_list], [])

    def forward(self, x):
        torch.cuda.current_stream().wait_stream(self.backup_stream)

        for m in self.module_list:
            x = m(x)

        return x
