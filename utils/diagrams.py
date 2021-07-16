import shutil
from pathlib import Path
from typing import List
from xml.dom import minidom

from labml import monit

HOME = Path('.').absolute()

STYLES = """
.black-stroke {
    stroke: #aaa;
}

rect.black-stroke {
    stroke: #444;
}

.black-fill {
    fill: #ddd;
}

.white-fill {
    fill: #333;
}

.blue-stroke {
    stroke: #5b8fab;
}

.blue-fill {
    fill: #356782;
}

.yellow-stroke {
    stroke: #bbab52;
}

.yellow-fill {
    fill: #a7942b;
}

.grey-stroke {
    stroke: #484d5a;
}

.grey-fill {
    fill: #2e323c;
}

.red-stroke {
    stroke: #bb3232;
}

.red-fill {
    fill: #901c1c;
}

.orange-stroke {
    stroke: #a5753f;
}

.orange-fill {
    fill: #82531e;
}

.purple-stroke {
    stroke: #a556a5;
}

.purple-fill {
    fill: #8a308a;
}

.green-stroke {
    stroke: #80cc92;
}

.green-fill {
    fill: #499e5d;
}

switch foreignObject div div div {
    color: #ddd !important;
}

switch foreignObject div div div span {
    color: #ddd !important;
}

.has-background {
    background-color: #1d2127 !important;
}
"""

STROKES = {
    '#000000': 'black',
    '#6c8ebf': 'blue',
    '#d6b656': 'yellow',
    '#666666': 'grey',
    '#b85450': 'red',
    '#d79b00': 'orange',
    '#9673a6': 'purple',
    '#82b366': 'green',
}

FILLS = {
    '#000000': 'black',
    '#ffffff': 'white',
    '#dae8fc': 'blue',
    '#fff2cc': 'yellow',
    '#f5f5f5': 'grey',
    '#f8cecc': 'red',
    '#ffe6cc': 'orange',
    '#e1d5e7': 'purple',
    '#d5e8d4': 'green',
}


def clear_switches(doc: minidom.Document):
    switches = doc.getElementsByTagName('switch')
    for s in switches:
        children = s.childNodes
        assert len(children) == 2
        if children[0].tagName == 'g' and 'requiredFeatures' in children[0].attributes:
            s.parentNode.removeChild(s)
            s.unlink()
            continue
        assert children[0].tagName == 'foreignObject'
        assert children[1].tagName == 'text'
        c = children[1]
        s.removeChild(c)
        s.parentNode.insertBefore(c, s)
        s.parentNode.removeChild(s)


def add_class(node: minidom.Node, class_name: str):
    if 'class' not in node.attributes:
        node.attributes['class'] = class_name
        return

    node.attributes['class'] = node.attributes['class'].value + f' {class_name}'


def add_bg_classes(nodes: List[minidom.Node]):
    for node in nodes:
        if 'style' in node.attributes:
            s = node.attributes['style'].value
            if s.count('background-color'):
                add_class(node, 'has-background')


def add_stroke_classes(nodes: List[minidom.Node]):
    for node in nodes:
        if 'stroke' in node.attributes:
            stroke = node.attributes['stroke'].value
            if stroke not in STROKES:
                continue

            node.removeAttribute('stroke')
            add_class(node, f'{STROKES[stroke]}-stroke')


def add_fill_classes(nodes: List[minidom.Node]):
    for node in nodes:
        if 'fill' in node.attributes:
            fill = node.attributes['fill'].value
            if fill not in FILLS:
                continue

            node.removeAttribute('fill')
            add_class(node, f'{FILLS[fill]}-fill')


def add_classes(doc: minidom.Document):
    paths = doc.getElementsByTagName('path')
    add_stroke_classes(paths)
    add_fill_classes(paths)

    rects = doc.getElementsByTagName('rect')
    add_stroke_classes(rects)
    add_fill_classes(rects)

    ellipse = doc.getElementsByTagName('ellipse')
    add_stroke_classes(ellipse)
    add_fill_classes(ellipse)

    text = doc.getElementsByTagName('text')
    add_fill_classes(text)

    div = doc.getElementsByTagName('div')
    add_bg_classes(div)

    span = doc.getElementsByTagName('span')
    add_bg_classes(span)


def parse(source: Path, dest: Path):
    doc: minidom.Document = minidom.parse(str(source))

    svg = doc.getElementsByTagName('svg')

    assert len(svg) == 1
    svg = svg[0]

    if 'content' in svg.attributes:
        svg.removeAttribute('content')
    # svg.attributes['height'] = str(int(svg.attributes['height'].value[:-2]) + 30) + 'px'
    # svg.attributes['width'] = str(int(svg.attributes['width'].value[:-2]) + 30) + 'px'

    view_box = svg.attributes['viewBox'].value.split(' ')
    view_box = [float(v) for v in view_box]
    view_box[0] -= 10
    view_box[1] -= 10
    view_box[2] += 20
    view_box[3] += 20
    svg.attributes['viewBox'] = ' '.join([str(v) for v in view_box])

    svg.attributes['style'] = 'background: #1d2127;'  # padding: 10px;'

    # clear_switches(doc)

    style = doc.createElement('style')
    style.appendChild(doc.createTextNode(STYLES))
    svg.insertBefore(style, svg.childNodes[0])
    add_classes(doc)

    with open(str(dest), 'w') as f:
        doc.writexml(f)


def recurse(path: Path):
    files = []
    if path.is_file():
        files.append(path)
        return files

    for f in path.iterdir():
        files += recurse(f)

    return files


def main():
    diagrams_path = HOME / 'diagrams'
    docs_path = HOME / 'docs'

    for p in recurse(diagrams_path):
        source_path = p
        p = p.relative_to(diagrams_path)
        dest_path = docs_path / p
        if not dest_path.parent.exists():
            dest_path.parent.mkdir(parents=True)

        with monit.section(str(p)):
            if source_path.suffix == '.svg':
                parse(source_path, dest_path)
            else:
                shutil.copy(str(source_path), str(dest_path))


if __name__ == '__main__':
    main()
