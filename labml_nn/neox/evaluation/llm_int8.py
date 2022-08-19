import torch
from torch import nn

from labml import monit
from labml_nn.neox.evaluation import run_eval_harness
from labml_nn.neox.model import LayerGenerator

if __name__ == '__main__':
    device = torch.device('cuda:0')
    layer_generator = LayerGenerator(is_clone_layers=False,
                                     dtype=torch.float16,
                                     device=torch.device('cpu'),
                                     )
    # Load layers
    layers = list(layer_generator.load())

    # This reduces CUDA memory fragmentation
    for layer in monit.iterate('Convert to int8', layers, is_children_silent=True):
        layer_generator.post_load_prepare(layer,
                                          device=device,
                                          is_llm_int8=True,
                                          llm_int8_threshold=6.0,
                                          )
        layer.to(device)

    with monit.section('Sequential'):
        model = nn.Sequential(*layers)

    print(run_eval_harness(model, 'half_precision', [], device))
