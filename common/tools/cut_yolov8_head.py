#!/usr/bin/env python3
"""Cut the DFL decode head off a YOLOv8/v11 deploy ONNX.

YOLOv8 and YOLOv11 detection heads do DFL (Softmax) + dist2bbox
(Slice/Sub/Div) decoding on a flattened spatial axis. The Statlas NPU
compiler cannot tile those ops on that axis and core-dumps. Cutting the
graph at the detect-branch conv outputs leaves 4D NCHW feature maps that
tile cleanly; the DFL + decode is then done on the host.

The cut is structural (no node-name assumptions), so it works for both
ultralytics v8 (``/model.22/cv2.x``) and v11 (``input_tensor.x``) naming.
YOLOv5 heads reshape a 5D tensor ``[1,3,85,H,W]`` (not 4D), so they do not
match the cut pattern and this script is a safe no-op for them.

Usage:
    python cut_yolov8_head.py --input_model <deploy>.onnx \
                              --output_model <deploy>_headcut.onnx
"""
import argparse
import os
import sys
from pathlib import Path

import onnx
from onnx import shape_inference, helper


def _shapes(graph):
    """name -> list[int] dim values for every value_info / IO tensor."""
    out = {}
    for v in list(graph.input) + list(graph.output) + list(graph.value_info):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        out[v.name] = dims
    return out


def _is_spatial_flatten_reshape(node, shapes):
    """True if node Reshapes a 4D [1,C,H,W] into a 3D [1,C,H*W].

    This is exactly the head-entry pattern (detect-branch conv output ->
    flattened decode). It excludes channel-merging reshapes inside the DFL
    (e.g. [1,1,4,G] -> [1,4,G], where the channel dim is NOT preserved).
    """
    if node.op_type != 'Reshape':
        return False
    inp = shapes.get(node.input[0])
    out = shapes.get(node.output[0])
    if not inp or not out:
        return False
    if len(inp) != 4 or len(out) != 3:
        return False
    _, c, h, w = inp
    _, oc, g = out
    # channel preserved and last two spatial dims flattened
    return c == oc and c > 0 and h > 1 and w > 1 and h * w == g


def _find_dfl_softmax(graph, shapes):
    """Return a DFL Softmax node, or None.

    A YOLOv8/v11 DFL Softmax consumes a reshaped
    ``[1, reg_max, 4, N]`` (or ``[reg_max, 4, N]``) tensor - the box branch
    flattened and regrouped so softmax runs over ``reg_max`` (typically 16,
    >=4 here to stay specific). This shape is the reliable signature of a
    DFL head that tiles badly on the NPU.

    YOLOv5 has no such Softmax; YOLOv26 (NMS-free) has Softmax but on a
    ``[1, 2, H, W]`` attention-style tensor, not ``[1, R, 4, N]``, so it is
    correctly skipped here and compiled with its native head.
    """
    for n in graph.node:
        if n.op_type != 'Softmax':
            continue
        s = shapes.get(n.input[0])
        if not s:
            continue
        if len(s) == 4:
            _, reg_max, four, n_anchors = s
        elif len(s) == 3:
            reg_max, four, n_anchors = s
        else:
            continue
        if four == 4 and reg_max >= 4:
            return n
    return None


def find_cut_points(graph):
    """Return ordered, de-duplicated 4D conv outputs that feed the head.

    Only cuts when a real DFL head (``[1, reg_max, 4, N]`` Softmax) is present.
    Each cut point is a detect-branch 1x1 conv output feeding a 4D->3D
    spatial-flatten Reshape; cutting there removes the entire DFL + dist2bbox
    decode. Returns an empty list (=> no-op) for v5 and for NMS-free heads
    (yolo26) that compile natively.
    """
    shapes = _shapes(graph)
    if _find_dfl_softmax(graph, shapes) is None:
        return [], shapes
    cut = []
    seen = set()
    for n in graph.node:
        if _is_spatial_flatten_reshape(n, shapes):
            src = n.input[0]
            if src not in seen:
                seen.add(src)
                cut.append(src)
    return cut, shapes


def _ancestors(graph, seeds):
    """Names of all nodes (and tensors) needed to produce `seeds`."""
    needed_tensors = set(seeds)
    keep_nodes = []
    for n in reversed(graph.node):
        if any(o in needed_tensors for o in n.output):
            keep_nodes.append(n)
            for i in n.input:
                needed_tensors.add(i)
    keep_nodes.reverse()
    return keep_nodes, needed_tensors


def cut_head(in_path, out_path):
    m = onnx.load(in_path)
    m = shape_inference.infer_shapes(m)
    g = m.graph

    cut_points, shapes = find_cut_points(g)
    if not cut_points:
        if _find_dfl_softmax(g, shapes) is None:
            print('[cut_head] no DFL [1, reg_max, 4, N] softmax found; '
                  'nothing to do (v5-style or NMS-free head compiles natively).')
        else:
            print('[cut_head] DFL softmax found but no 4D->3D cut point; '
                  'nothing to do.')
        return False

    keep_nodes, needed = _ancestors(g, cut_points)
    total_nodes = len(g.node)

    # rebuild graph: same inputs, new outputs = cut points, kept initializers
    src_init = {i.name: i for i in g.initializer}
    kept_init = [src_init[name] for name in src_init if name in needed]

    del g.node[:]
    g.node.extend(keep_nodes)
    del g.initializer[:]
    g.initializer.extend(kept_init)

    # set outputs = cut points with their static shapes
    del g.output[:]
    for name in cut_points:
        dims = shapes[name]
        g.output.extend([helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, dims)])

    # Keep type metadata for retained custom/vendor nodes. Generic ONNX shape
    # inference cannot reconstruct every Statlas deploy tensor type.
    kept_value_info = [value for value in g.value_info
                       if value.name in needed]
    del g.value_info[:]
    g.value_info.extend(kept_value_info)
    m = shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, out_path)

    print('[cut_head] kept {}/{} nodes, dropped {} head nodes'.format(
        len(keep_nodes), total_nodes, total_nodes - len(keep_nodes)))
    print('[cut_head] outputs:')
    for name in cut_points:
        print('   {} = {}'.format(name, shapes[name]))
    _write_spec(in_path, out_path, cut_points, shapes)
    return True


def _write_spec(in_path, out_path, cut_points, shapes):
    """Drop a small decode-spec yaml next to the head-cut model."""
    # group cut outputs by scale (H, W); box branch = larger C (4*reg_max)
    scales = {}
    input_dims = None
    m = onnx.load(in_path)
    for i in m.graph.input:
        if len(i.type.tensor_type.shape.dim) == 4:
            input_dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    for name in cut_points:
        _, c, h, w = shapes[name]
        scales.setdefault((h, w), []).append((name, c))
    spec_path = Path(str(out_path).replace('.onnx', '_spec.yaml'))
    lines = ['# Auto-generated head-cut decode spec for host-side DFL+dist2bbox.']
    if input_dims:
        lines.append('input: [{}]'.format(', '.join(str(d) for d in input_dims)))
    lines.append('num_outputs: {}'.format(len(cut_points)))
    lines.append('outputs:')
    reg_max = None
    nc = None
    for (h, w) in sorted(scales):
        outs = sorted(scales[(h, w)], key=lambda x: -x[1])
        box = outs[0]  # largest C
        cls = outs[-1] if len(outs) > 1 else None
        rm = box[1] // 4
        reg_max = rm if reg_max is None else reg_max
        if cls:
            nc = cls[1] if nc is None else nc
        stride = (input_dims[2] // h) if input_dims else None
        lines.append('  - scale: [{}, {}]'.format(h, w))
        if stride:
            lines.append('    stride: {}'.format(stride))
        lines.append('    box:')
        lines.append('      name: {}'.format(box[0]))
        lines.append('      shape: {}  # [1, 4*reg_max, H, W]'.format(shapes[box[0]]))
        lines.append('      reg_max: {}'.format(rm))
        if cls:
            lines.append('    cls:')
            lines.append('      name: {}'.format(cls[0]))
            lines.append('      shape: {}  # [1, nc, H, W]'.format(shapes[cls[0]]))
            lines.append('      nc: {}'.format(cls[1]))
    lines.append('')
    lines.append('# host decode: DFL(softmax over reg_max) on box branch, '
                'sigmoid on cls branch,')
    lines.append('# dist2bbox with stride to xyxy, then NMS.')
    spec_path.write_text('\n'.join(lines), encoding='utf-8')
    print('[cut_head] decode spec: {}'.format(spec_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--input_model', required=True)
    ap.add_argument('--output_model', required=True)
    args = ap.parse_args()
    if not Path(args.input_model).exists():
        sys.exit('input not found: {}'.format(args.input_model))
    print('[cut_head] input:  {}'.format(args.input_model))
    print('[cut_head] output: {}'.format(args.output_model))
    cut = cut_head(args.input_model, args.output_model)
    if cut:
        print('[cut_head] done. size: {:.1f} MB'.format(
            os.path.getsize(args.output_model) / 1024 / 1024))
    else:
        print('[cut_head] no head-cut model written.')


if __name__ == '__main__':
    main()
