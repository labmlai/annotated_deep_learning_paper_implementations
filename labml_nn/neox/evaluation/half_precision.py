import torch
from torch import nn

from labml import monit
from labml_nn.neox.evaluation import run_eval_harness
from labml_nn.neox.model import LayerGenerator

if __name__ == '__main__':
    device = torch.device('cuda:0')
    layers = list(LayerGenerator(is_clone_layers=True,
                                 filter_layers=None,
                                 dtype=torch.float16,
                                 device=device
                                 ).load())

    with monit.section('Sequential'):
        model = nn.Sequential(*layers)

    print(run_eval_harness(model, 'half_precision', ['lambada'], device))
