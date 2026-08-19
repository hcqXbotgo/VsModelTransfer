#!/usr/bin/env python3
"""Cut a YOLO26 NMS-free model into six INT8-friendly raw outputs."""
import argparse
import os
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper, shape_inference


def _shape_map(graph):
    return {
        value.name: [dim.dim_value or dim.dim_param
                     for dim in value.type.tensor_type.shape.dim]
        for value in list(graph.input) + list(graph.output) +
        list(graph.value_info)
    }


def _ancestors(graph, tensors):
    needed = set(tensors)
    nodes = []
    for node in reversed(graph.node):
        if any(output in needed for output in node.output):
            nodes.append(node)
            needed.update(node.input)
    nodes.reverse()
    return nodes, needed


def _reshape_sources(graph, shapes, ancestor_names):
    producer = {output: node for node in graph.node for output in node.output}
    result = []
    for node in graph.node:
        if node.op_type != 'Reshape' or node.name not in ancestor_names:
            continue
        source_shape = shapes.get(node.input[0])
        output_shape = shapes.get(node.output[0])
        source = producer.get(node.input[0])
        if (not source or source.op_type != 'Conv' or
                not source_shape or not output_shape or
                len(source_shape) != 4 or len(output_shape) != 3):
            continue
        batch, channels, height, width = source_shape
        if (batch == 1 and height > 1 and width > 1 and
                output_shape == [1, channels, height * width]):
            result.append((node.input[0], channels, height, width))
    return result


def find_cut_points(model):
    """Return three ordered box/class output pairs for a YOLO26 graph."""
    graph = model.graph
    shapes = _shape_map(graph)
    if len(graph.output) != 1:
        return []
    packed_shape = shapes.get(graph.output[0].name)
    if not packed_shape or len(packed_shape) != 3 or packed_shape[0] != 1:
        return []
    num_classes = packed_shape[1] - 4
    if num_classes <= 0:
        return []

    producer = {output: node for node in graph.node for output in node.output}
    final_concat = producer.get(graph.output[0].name)
    if not final_concat or final_concat.op_type != 'Concat' or len(final_concat.input) != 2:
        return []

    class_input = None
    box_input = None
    for name in final_concat.input:
        node = producer.get(name)
        if node and node.op_type == 'Sigmoid':
            class_input = name
        else:
            box_input = name
    if not class_input or not box_input:
        return []

    class_nodes, _ = _ancestors(graph, [class_input])
    box_nodes, _ = _ancestors(graph, [box_input])
    class_sources = _reshape_sources(
        graph, shapes, {node.name for node in class_nodes})
    box_sources = _reshape_sources(
        graph, shapes, {node.name for node in box_nodes})

    input_shape = shapes.get(graph.input[0].name) if graph.input else None
    if not input_shape or len(input_shape) != 4:
        return []
    input_h, input_w = input_shape[2:]
    points = []
    for box_name, box_channels, height, width in box_sources:
        matches = [item for item in class_sources
                   if item[2:] == (height, width)]
        if len(matches) != 1 or box_channels != 4:
            continue
        class_name, class_channels, _, _ = matches[0]
        if class_channels != num_classes or input_h % height or input_w % width:
            continue
        stride_h, stride_w = input_h // height, input_w // width
        if stride_h != stride_w or stride_h not in (8, 16, 32):
            continue
        points.append({
            'stride': stride_h,
            'height': height,
            'width': width,
            'box': box_name,
            'cls': class_name,
            'num_classes': num_classes,
        })
    points.sort(key=lambda item: item['stride'])
    if len(points) != 3 or [item['stride'] for item in points] != [8, 16, 32]:
        return []
    return points


def cut_head(input_path, output_path):
    model = shape_inference.infer_shapes(onnx.load(str(input_path)))
    points = find_cut_points(model)
    if not points:
        print('[cut_yolo26_head] no supported NMS-free head found')
        return False

    graph = model.graph
    output_names = []
    for item in points:
        output_names.extend([item['box'], item['cls']])
    nodes, needed = _ancestors(graph, output_names)
    initializers = [item for item in graph.initializer if item.name in needed]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.output[:]
    for name in output_names:
        shape = _shape_map(graph).get(name)
        graph.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, shape))
    del graph.value_info[:]

    model = shape_inference.infer_shapes(model)
    checker.check_model(model)
    onnx.save(model, str(output_path))
    _write_spec(input_path, output_path, points)
    print('[cut_yolo26_head] outputs:')
    for item in points:
        print('  stride {}: box={} cls={} shape=[1, {}, {}, {}]'.format(
            item['stride'], item['box'], item['cls'], item['num_classes'],
            item['height'], item['width']))
    return True


def _write_spec(input_path, output_path, points):
    spec_path = Path(output_path).with_name(
        Path(output_path).stem + '_spec.yaml')
    lines = [
        '# Auto-generated YOLO26 host-decode specification.',
        'source: {}'.format(input_path),
        'layout: NCHW',
        'box_encoding: ltrb_distance',
        'num_classes: {}'.format(points[0]['num_classes']),
        'outputs:',
    ]
    for item in points:
        lines.extend([
            '  - stride: {}'.format(item['stride']),
            '    box: {}'.format(item['box']),
            '    cls: {}'.format(item['cls']),
            '    box_shape: [1, 4, {}, {}]'.format(
                item['height'], item['width']),
            '    cls_shape: [1, {}, {}, {}]'.format(
                item['num_classes'], item['height'], item['width']),
        ])
    lines.extend([
        '',
        '# Host decode: xyxy=(grid_center + [-l,-t,r,b])*stride,',
        '# sigmoid on class logits, then per-class NMS.',
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
        sys.exit('input is not a supported YOLO26 NMS-free model')
    print('[cut_yolo26_head] done. size: {:.1f} MB'.format(
        os.path.getsize(args.output_model) / 1024 / 1024))


if __name__ == '__main__':
    main()
