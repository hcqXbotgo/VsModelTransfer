#!/usr/bin/env python3
"""Generate a YOLOv8/v11/yolo26 mixed-precision config for StatlasQuant.

The detection head's box/cls branches have a large dynamic range (bbox
coords up to the image size vs confidences in 0..1). Sharing one INT8
scale zeros out confidences, so the head is kept FP16 while the backbone
stays INT8.

A wrong MP config (e.g. every layer FP16 but a few activations missed)
makes each Conv's input/weight/output land on different storage types and
StatlasCompile aborts with "storage type of conv input, weight and output
should be the same". This script avoids that by setting FP16 on a
**complete, contiguous** head region: every ``model/.../cv2.*`` (box) and
``model/.../cv3.*`` (cls) detect-branch Conv, its weight, its output
activation, and the backbone-fork tensor feeding the first head conv -
so every head Conv sees input=weight=output=FP16, and the only INT8<->FP16
boundary is at the backbone fork (also marked FP16).

Usage:
    python gen_yolo26_mp.py --input_model <deploy>.onnx \
                            --output modes/<mode>/configs/mixed_precision.yaml
"""
import argparse
from collections import OrderedDict

import onnx
from onnx import shape_inference


def head_layer_names(graph):
    """Detect-head layer names that must be FP16.

    Covers cv2 (box) and cv3 (cls) Conv weights under the detect module
    (``model/23/`` or ``model/24/``) only - NOT backbone C2f blocks whose
    inner convs are also named ``cv2``/``cv3`` (e.g. ``model/2/cv2``).
    Plus each head conv's output activation and the backbone-fork tensor
    feeding the first head conv.
    """
    names = OrderedDict()
    for n in graph.node:
        if n.op_type != 'Conv':
            continue
        weights = [i for i in n.input if i.endswith('Conv.weight')]
        if not weights:
            continue
        w = weights[0]
        if 'model/23/' not in w and 'model/24/' not in w:
            continue
        if 'cv2/' not in w and 'cv3/' not in w:
            continue
        names[w] = None              # weight -> FP16
        names[n.output[0]] = None    # output activation -> FP16
        names[n.input[0]] = None     # backbone-fork input -> FP16
    return list(names.keys())


def write_yaml(out_path, names, model_name):
    lines = [
        '# Auto-generated mixed-precision config for {}.'.format(model_name),
        '# Only the detection head (cv2.* box + cv3.* cls) Conv weights, their',
        '# output activations, and the backbone-fork inputs are FP16; everything',
        '# else stays INT8 (per quant.yaml). Keeps each head Conv input/weight/',
        '# output at the same storage type to avoid "storage type should be the',
        '# same" compile errors. Regenerate via common/tools/gen_yolo26_mp.py',
        '# if the deploy ONNX changes.',
        'layers:',
    ]
    for name in names:
        lines += [
            '- layername: {}'.format(name),
            '  nbit: 16',
            '  observer: MinMaxObserver',
            '  symmetry: true',
            '  per_channel: false',
            '  round_type: None',
            '  static: 1',
        ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--input_model', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    m = onnx.load(args.input_model)
    m = shape_inference.infer_shapes(m)
    names = head_layer_names(m.graph)
    if not names:
        raise SystemExit('no cv2/cv3 detect-branch Conv found; is this a YOLO detect model?')
    model_name = onnx.load(args.input_model).graph.name or 'model'
    write_yaml(args.output, names, model_name)
    print('[gen_mp] {} FP16 layers -> {}'.format(len(names), args.output))


if __name__ == '__main__':
    main()
