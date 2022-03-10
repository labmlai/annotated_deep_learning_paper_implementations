import json
import re
from pathlib import Path

from labml import logger
from labml.logger import Text

HOME = Path('./labml_nn')

REGEX = re.compile(r"""
 \(
 https://papers\.labml\.ai/paper/  # Start of a numeric entity reference
 (?P<id>[0-9\.]+)  # Paper ID
 \)
""", re.VERBOSE)

IGNORE = {
    'transformers/index.html',
    'transformers/configs.html',
    'optimizers/noam.html',
    'transformers/basic/autoregressive_experiment.html',
    'transformers/xl/relative_mha.html',
    'capsule_networks/mnist.html',
}

IGNORE_PAPERS = {
    '2002.04745',  # On Layer Normalization in the Transformer Architecture
    '1606.08415',  # Gaussian Error Linear Units (GELUs)
    '1710.10196',  # Progressive Growing of GANs for Improved Quality, Stability, and Variation
    '1904.11486',  # Making Convolutional Networks Shift-Invariant Again
    '1801.04406',  # Which Training Methods for GANs do actually Converge?
    '1812.04948',  # A Style-Based Generator Architecture for Generative Adversarial Networks
    '1705.10528',  # Constrained Policy Optimization
}


def collect(path: Path):
    if path.is_file():
        html = path.relative_to(HOME)
        if html.suffix not in {'.py'}:
            return []

        if html.stem == '__init__':
            html = html.parent / 'index.html'
        else:
            html = html.parent / f'{html.stem}.html'

        if str(html) in IGNORE:
            return []

        with open(str(path), 'r') as f:
            contents = f.read()
            papers = set()
            for m in REGEX.finditer(contents):
                if m.group('id') in IGNORE_PAPERS:
                    continue
                papers.add(m.group('id'))

            if len(papers) > 1:
                logger.log([(str(html), Text.key), ': ', str(papers)])
            return [{'url': str(html), 'arxiv_id': p} for p in papers]

    urls = []
    for f in path.iterdir():
        urls += collect(f)

    return urls


def main():
    papers = []
    for f in HOME.iterdir():
        papers += collect(f)

    papers.sort(key=lambda p: p['arxiv_id'])

    by_id = {}
    for p in papers:
        if p['arxiv_id'] not in by_id:
            by_id[p['arxiv_id']] = []
        by_id[p['arxiv_id']].append(f'''https://nn.labml.ai/{p['url']}''')

    logger.log([('Papers', Text.key), ': ', f'{len(by_id) :,}'])

    with open(str(HOME.parent / 'docs' / 'papers.json'), 'w') as f:
        f.write(json.dumps(by_id, indent=1))


if __name__ == '__main__':
    main()
