import torch

from labml.configs import BaseConfigs, hyperparams, option


class DeviceInfo:
    def __init__(self, *,
                 use_cuda: bool,
                 cuda_device: int):
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.cuda_count = torch.cuda.device_count()

        self.is_cuda = self.use_cuda and torch.cuda.is_available()
        if not self.is_cuda:
            self.device = torch.device('cpu')
        else:
            if self.cuda_device < self.cuda_count:
                self.device = torch.device('cuda', self.cuda_device)
            else:
                self.device = torch.device('cuda', self.cuda_count - 1)

    def __str__(self):
        if not self.is_cuda:
            return "CPU"

        if self.cuda_device < self.cuda_count:
            return f"GPU:{self.cuda_device} - {torch.cuda.get_device_name(self.cuda_device)}"
        else:
            return (f"GPU:{self.cuda_count - 1}({self.cuda_device}) "
                    f"- {torch.cuda.get_device_name(self.cuda_count - 1)}")


class DeviceConfigs(BaseConfigs):
    r"""
    This is a configurable module to get a single device to train model on.
    It can pick up CUDA devices and it will fall back to CPU if they are not available.

    It has other small advantages such as being able to view the
    actual device name on configurations view of
    `labml app <https://github.com/labmlai/labml/tree/master/app>`_

    Arguments:
        cuda_device (int): The CUDA device number. Defaults to ``0``.
        use_cuda (bool): Whether to use CUDA devices. Defaults to ``True``.
    """
    cuda_device: int = 0
    use_cuda: bool = True

    device_info: DeviceInfo

    device: torch.device

    def __init__(self):
        super().__init__(_primary='device')


@option(DeviceConfigs.device)
def _device(c: DeviceConfigs):
    return c.device_info.device


hyperparams(DeviceConfigs.cuda_device, DeviceConfigs.use_cuda,
            is_hyperparam=False)


@option(DeviceConfigs.device_info)
def _device_info(c: DeviceConfigs):
    return DeviceInfo(use_cuda=c.use_cuda,
                      cuda_device=c.cuda_device)
