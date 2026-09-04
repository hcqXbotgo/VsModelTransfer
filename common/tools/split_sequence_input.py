#!/usr/bin/env python3
"""Split one channel-stacked ONNX input into multiple tensor inputs.

The source graph keeps its original computation: ``input0``, ``input1``, ...
are concatenated on the channel axis and feed the original input name.  This
is useful for temporal RGB models whose exported input is ``[N, 3*T, H, W]``
but whose runtime should receive T independent ``[N, 3, H, W]`` tensors.
"""

import argparse
from pathlib import Path

def _shape(value_info):
    tensor = value_info.type.tensor_type
    if tensor.elem_type == 0:
        raise ValueError('input {} has no element type'.format(value_info.name))
    dims = []
    for dim in tensor.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return tensor.elem_type, dims


def split_input(input_path, output_path, names, axis=1):
    import onnx
    from onnx import helper

    model = onnx.load(str(input_path))
    if len(model.graph.input) != 1:
        raise ValueError(
            'expected exactly one graph input, found {}'.format(
                len(model.graph.input)))
    source = model.graph.input[0]
    elem_type, source_shape = _shape(source)
    if len(source_shape) != 4:
        raise ValueError('expected a 4D NCHW input, got {}'.format(source_shape))
    if axis != 1:
        raise ValueError('only channel-axis splitting is supported')
    channels = source_shape[axis]
    if not isinstance(channels, int) or channels % len(names):
        raise ValueError(
            'input channel count {} is not divisible by {} inputs'.format(
                channels, len(names)))
    channels_per_input = channels // len(names)

    if any(value.name in names for value in model.graph.input):
        raise ValueError('new input names must differ from the original input')
    if len(set(names)) != len(names):
        raise ValueError('new input names must be unique')
    existing_names = {
        value.name for value in model.graph.input
    } | {value.name for value in model.graph.output}
    existing_names.update(initializer.name for initializer in model.graph.initializer)
    collisions = [name for name in names if name in existing_names]
    if collisions:
        raise ValueError('input name(s) already exist: {}'.format(', '.join(collisions)))

    # The concat output deliberately reuses the old input name, so every
    # downstream node remains unchanged.
    new_shape = list(source_shape)
    new_shape[axis] = channels_per_input
    new_inputs = [helper.make_tensor_value_info(name, elem_type, new_shape)
                  for name in names]
    concat = helper.make_node(
        'Concat', list(names), [source.name], axis=axis,
        name='split_sequence_inputs_concat')

    del model.graph.input[:]
    model.graph.input.extend(new_inputs)
    model.graph.node.insert(0, concat)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument(
        '--output', type=Path,
        help='output ONNX; defaults to <input_stem>_3input.onnx')
    parser.add_argument(
        '--names', nargs='+', default=['input0', 'input1', 'input2'],
        help='new input tensor names in temporal order '
             '(default: input0 input1 input2)')
    args = parser.parse_args()
    if len(args.names) < 2:
        parser.error('--names requires at least two tensor names')
    output = args.output or args.input.with_name(
        '{}_{}input.onnx'.format(args.input.stem, len(args.names)))
    split_input(args.input, output, args.names)
    print('wrote {}'.format(output))


if __name__ == '__main__':
    main()
