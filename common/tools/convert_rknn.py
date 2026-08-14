#!/usr/bin/env python3
"""Convert a mode's configured ONNX model to an RKNN model."""
import argparse
import json
import sys
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
QUANTIZED_DTYPES = {'w8a8', 'w8a16', 'w16a16i', 'w16a16i_dfp', 'w4a16'}
QUANTIZED_ALGORITHMS = {'normal', 'mmse', 'kl_divergence', 'gdq'}
QUANTIZED_METHODS = {'layer', 'channel'} | {
    'group{}'.format(size) for size in range(32, 257, 32)}


def resolve_path(path, workspace):
    path = Path(path)
    return path if path.is_absolute() else Path(workspace) / path


def rknn_model_path(config, workspace):
    return resolve_path(config['model']['onnx_model'], workspace)


def write_dataset(config, output_path, workspace):
    dataset = config['dataset']
    image_dir = resolve_path(dataset['root'], workspace)
    if not image_dir.is_dir():
        raise SystemExit('RKNN calibration directory not found: {}'.format(
            image_dir))
    images = sorted(path for path in image_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    sample_count = int(dataset.get('sample_count', 0))
    if sample_count > 0:
        images = images[:sample_count]
    if not images:
        raise SystemExit('No RKNN calibration images found under {}'.format(
            image_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        ''.join('{}\n'.format(path.resolve()) for path in images),
        encoding='utf-8')
    return images


def should_quantize(head_type):
    """Packed native outputs stay floating for the embedded C++ contract."""
    return head_type in ('yolov5_headcut', 'yolov8_headcut')


def conversion_options(config, platform, head_type):
    """Translate rknn.yaml into RKNN Toolkit2 config/build arguments."""
    configured_platform = config.get('target_platform', platform)
    if configured_platform != platform:
        raise SystemExit(
            'RKNN config target_platform {} does not match --platform {}'.format(
                configured_platform, platform))

    quant = config.get('quant', {})
    build = config.get('build', {})
    preprocess = config.get('preprocess', {})
    do_quantization = bool(config.get(
        'do_quantization', should_quantize(head_type)))
    dtype = quant.get('dtype', 'w8a8')
    algorithm = quant.get('algorithm', 'normal')
    method = quant.get('method', 'channel')
    float_dtype = build.get('float_dtype', 'float16')
    for name, value, choices in (
            ('quant.dtype', dtype, QUANTIZED_DTYPES),
            ('quant.algorithm', algorithm, QUANTIZED_ALGORITHMS),
            ('quant.method', method, QUANTIZED_METHODS),
            ('build.float_dtype', float_dtype, {'float16'})):
        if value not in choices:
            raise SystemExit(
                'Unsupported RKNN {} {!r}; choose one of {}'.format(
                    name, value, ', '.join(sorted(choices))))

    config_kwargs = {
        'mean_values': [[float(value) * 255.0 for value in
                         preprocess.get('mean', [0.0, 0.0, 0.0])]],
        'std_values': [[float(value) * 255.0 for value in
                        preprocess.get('std', [1.0, 1.0, 1.0])]],
        'target_platform': platform,
        'quantized_dtype': dtype,
        'quantized_algorithm': algorithm,
        'quantized_method': method,
        'quantized_hybrid_level': int(quant.get('hybrid_level', 0)),
        'optimization_level': int(build.get('optimization_level', 3)),
        'float_dtype': float_dtype,
    }
    build_kwargs = {
        'do_quantization': do_quantization,
        'auto_hybrid': bool(quant.get('auto_hybrid', False)),
    }
    return config_kwargs, build_kwargs


def _static_shapes(model):
    return [[dim.dim_value or dim.dim_param
             for dim in value.type.tensor_type.shape.dim]
            for value in model.graph.output]


def classify_outputs(model):
    shapes = _static_shapes(model)
    if (len(shapes) == 3 and all(len(shape) == 4 for shape in shapes) and
            all(shape[-1] % 3 == 0 for shape in shapes)):
        return 'yolov5_headcut'
    if len(shapes) == 6 and all(len(shape) == 4 for shape in shapes):
        return 'yolov8_headcut'
    if len(shapes) == 1 and len(shapes[0]) == 3:
        return 'native'
    raise SystemExit('Unsupported RKNN output contract: {}'.format(shapes))


def standardize_vendor_ops(input_path):
    """Expand supported Statlas-only operators for generic ONNX consumers."""
    import onnx
    from onnx import helper

    model = onnx.load(str(input_path))
    replacements = []
    changed = False
    for node in model.graph.node:
        if node.domain == 'vsdeploy' and node.op_type == 'Silu':
            sigmoid = '{}_rknn_sigmoid'.format(node.output[0])
            replacements.extend([
                helper.make_node(
                    'Sigmoid', [node.input[0]], [sigmoid],
                    name='{}_RknnSigmoid'.format(node.name)),
                helper.make_node(
                    'Mul', [node.input[0], sigmoid], list(node.output),
                    name='{}_RknnMul'.format(node.name)),
            ])
            changed = True
        else:
            replacements.append(node)
    if not changed:
        return input_path

    del model.graph.node[:]
    model.graph.node.extend(replacements)
    if not any(node.domain == 'vsdeploy' for node in model.graph.node):
        kept = [item for item in model.opset_import if item.domain != 'vsdeploy']
        del model.opset_import[:]
        model.opset_import.extend(kept)
    onnx.checker.check_model(model)
    output = input_path.with_name(input_path.stem + '_rknn.onnx')
    onnx.save(model, str(output))
    print('standardized: {} (vsdeploy::Silu -> Sigmoid + Mul)'.format(output))
    return output


def prepare_onnx(input_path):
    """Use the shared head cutter when the model has a supported YOLO head."""
    import onnx
    from onnx import shape_inference
    import cut_yolov5_head
    import cut_yolov8_head

    model = shape_inference.infer_shapes(onnx.load(str(input_path)))
    existing_type = classify_outputs(model)
    if existing_type != 'native':
        return input_path, existing_type
    if cut_yolov5_head.find_cut_points(model):
        output = input_path.with_name(input_path.stem + '_headcut_raw.onnx')
        if not output.is_file():
            if not cut_yolov5_head.cut_head(input_path, output):
                raise SystemExit('Failed to cut YOLOv5 head: {}'.format(
                    input_path))
        return output, 'yolov5_headcut'

    points, _ = cut_yolov8_head.find_cut_points(model.graph)
    if points:
        output = input_path.with_name(input_path.stem + '_headcut_raw.onnx')
        if not output.is_file():
            if not cut_yolov8_head.cut_head(str(input_path), str(output)):
                raise SystemExit('Failed to cut YOLOv8/11 head: {}'.format(
                    input_path))
        return output, 'yolov8_headcut'

    return input_path, classify_outputs(model)


def checked(result, operation):
    if result != 0:
        raise SystemExit('{} failed with status {}'.format(operation, result))


def convert(config_path, platform, output_path, workspace):
    try:
        from rknn.api import RKNN
    except ImportError:
        raise SystemExit(
            'rknn.api is not installed in this Python. Configure RKNN_PYTHON '
            'to a Python 3.9 RKNN Toolkit2 environment.')

    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    input_path = rknn_model_path(config, workspace)
    if not input_path.is_file():
        raise SystemExit('RKNN input ONNX not found: {}'.format(input_path))
    prepared_path, head_type = prepare_onnx(input_path)
    prepared_path = standardize_vendor_ops(prepared_path)
    config_kwargs, build_kwargs = conversion_options(
        config, platform, head_type)
    do_quantization = build_kwargs['do_quantization']
    dataset_path = Path(output_path).with_suffix('.dataset.txt')
    images = (write_dataset(config, dataset_path, workspace)
              if do_quantization else [])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print('RKNN platform: {}'.format(platform))
    print('input ONNX:   {}'.format(input_path))
    print('prepared:     {} ({})'.format(prepared_path, head_type))
    print('precision:    {}'.format('int8' if do_quantization else 'fp'))
    print('quant config: dtype={}, algorithm={}, method={}, hybrid={}'.format(
        config_kwargs['quantized_dtype'],
        config_kwargs['quantized_algorithm'],
        config_kwargs['quantized_method'],
        config_kwargs['quantized_hybrid_level']))
    print('calibration:  {}'.format(
        '{} image(s)'.format(len(images)) if do_quantization else 'not used'))
    print('output:       {}'.format(output_path))

    rknn = RKNN(verbose=True)
    try:
        rknn.config(**config_kwargs)
        checked(rknn.load_onnx(model=str(prepared_path)), 'RKNN load_onnx')
        checked(rknn.build(
            dataset=str(dataset_path) if do_quantization else None,
            **build_kwargs),
                'RKNN build')
        checked(rknn.export_rknn(str(output_path)), 'RKNN export')
    finally:
        rknn.release()

    manifest = {
        'platform': platform,
        'precision': 'int8' if do_quantization else 'fp',
        'head_type': head_type,
        'source_onnx': str(input_path),
        'prepared_onnx': str(prepared_path),
        'calibration_dataset': str(dataset_path) if do_quantization else None,
        'toolkit_config': config_kwargs,
        'toolkit_build': build_kwargs,
        'output': str(output_path),
    }
    output_path.with_suffix('.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='Mode configs/<platform>/rknn.yaml')
    parser.add_argument('--platform', required=True, choices=('rk3576',))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workspace', type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    convert(args.config, args.platform, args.output, args.workspace)


if __name__ == '__main__':
    main()
