#!/usr/bin/env python3
"""Cut an exported YOLOv5 Detect head into three C++-decodable outputs.

The output contract matches YoloV5PostProcessor::process_native_nhwc:
three tensors ordered by stride (8, 16, 32), each shaped
``[1, H, W, 3 * (5 + num_classes)]`` and containing sigmoid probabilities.
Grid, stride and anchor decoding remain on the host.
"""
import argparse
import os
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper, shape_inference


def _shape_map(graph):
    result = {}
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        result[value.name] = [
            dim.dim_value or dim.dim_param
            for dim in value.type.tensor_type.shape.dim
        ]
    return result


def _input_hw(graph, shapes):
    for value in graph.input:
        shape = shapes.get(value.name)
        if shape and len(shape) == 4 and all(
                isinstance(item, int) and item > 0 for item in shape[2:]):
            return shape[2], shape[3]
    raise ValueError('YOLOv5 input must have a static NCHW shape')


def find_cut_points(model):
    """Find Detect sigmoid tensors and describe C++-compatible outputs."""
    graph = model.graph
    shapes = _shape_map(graph)
    input_h, input_w = _input_hw(graph, shapes)
    producer = {output: node for node in graph.node for output in node.output}
    points = []

    for node in graph.node:
        if node.op_type != 'Sigmoid' or len(node.input) != 1:
            continue
        shape = shapes.get(node.output[0])
        if not shape or len(shape) != 5:
            continue
        batch, anchors, height, width, properties = shape
        if batch != 1 or anchors != 3 or properties < 6:
            continue
        if input_h % height or input_w % width:
            continue
        stride_h = input_h // height
        stride_w = input_w // width
        if stride_h != stride_w or stride_h not in (8, 16, 32):
            continue

        transpose = producer.get(node.input[0])
        reshape = producer.get(transpose.input[0]) if transpose else None
        conv = producer.get(reshape.input[0]) if reshape else None
        if not transpose or transpose.op_type != 'Transpose':
            continue
        if not reshape or reshape.op_type != 'Reshape':
            continue
        conv_shape = shapes.get(reshape.input[0])
        channels = anchors * properties
        if (not conv or conv.op_type != 'Conv' or
                conv_shape != [1, channels, height, width]):
            continue

        points.append({
            'source': node.output[0],
            'stride': stride_h,
            'height': height,
            'width': width,
            'anchors': anchors,
            'properties': properties,
            'num_classes': properties - 5,
            'output_shape': [1, height, width, channels],
        })

    points.sort(key=lambda item: item['stride'])
    if len(points) != 3 or [item['stride'] for item in points] != [8, 16, 32]:
        return []
    if len({item['num_classes'] for item in points}) != 1:
        return []
    return points


def _ancestors(graph, outputs):
    needed = set(outputs)
    kept = []
    for node in reversed(graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(node.input)
    kept.reverse()
    return kept, needed


def cut_head(input_path, output_path):
    model = shape_inference.infer_shapes(onnx.load(str(input_path)))
    graph = model.graph
    points = find_cut_points(model)
    if not points:
        print('[cut_yolov5_head] no three-scale YOLOv5 Detect head found')
        return False

    source_names = [item['source'] for item in points]
    kept_nodes, needed = _ancestors(graph, source_names)
    initializers = [item for item in graph.initializer if item.name in needed]

    output_names = []
    for item in points:
        stride = item['stride']
        transposed = 'yolov5_stride{}_nhwa'.format(stride)
        output = 'yolov5_stride{}_output'.format(stride)
        shape_name = 'yolov5_stride{}_shape'.format(stride)
        kept_nodes.append(helper.make_node(
            'Transpose', [item['source']], [transposed],
            name='YoloV5Stride{}Transpose'.format(stride),
            perm=[0, 2, 3, 1, 4]))
        initializers.append(helper.make_tensor(
            shape_name, TensorProto.INT64, [4], item['output_shape']))
        kept_nodes.append(helper.make_node(
            'Reshape', [transposed, shape_name], [output],
            name='YoloV5Stride{}Reshape'.format(stride)))
        output_names.append(output)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.output[:]
    for name, item in zip(output_names, points):
        graph.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, item['output_shape']))
    del graph.value_info[:]

    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    onnx.save(model, str(output_path))
    _write_spec(input_path, output_path, points, output_names)
    print('[cut_yolov5_head] outputs:')
    for name, item in zip(output_names, points):
        print('  stride {}: {} {}'.format(
            item['stride'], name, item['output_shape']))
    return True


def _write_spec(input_path, output_path, points, output_names):
    spec_path = Path(output_path).with_name(
        Path(output_path).stem + '_spec.yaml')
    lines = [
        '# Auto-generated YOLOv5 host-decode specification.',
        'source: {}'.format(input_path),
        'layout: NHWC',
        'anchors_per_scale: 3',
        'num_classes: {}'.format(points[0]['num_classes']),
        'properties_per_anchor: {}'.format(points[0]['properties']),
        'outputs:',
    ]
    for name, item in zip(output_names, points):
        lines.extend([
            '  - name: {}'.format(name),
            '    stride: {}'.format(item['stride']),
            '    shape: [{}]'.format(', '.join(
                str(value) for value in item['output_shape'])),
        ])
    lines.extend([
        '',
        '# Host must apply: xy=(sigmoid*2-0.5+grid)*stride,',
        '# wh=(sigmoid*2)^2*anchor, score=objectness*class, then NMS.',
    ])
    spec_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    args = parser.parse_args()
    if not Path(args.input_model).is_file():
        sys.exit('input not found: {}'.format(args.input_model))
    if not cut_head(args.input_model, args.output_model):
        sys.exit('input is not a supported three-scale YOLOv5 model')
    print('[cut_yolov5_head] done. size: {:.1f} MB'.format(
        os.path.getsize(args.output_model) / 1024 / 1024))


if __name__ == '__main__':
    main()
