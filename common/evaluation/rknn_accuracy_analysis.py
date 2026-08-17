#!/usr/bin/env python3
"""Run RKNN Toolkit2 layer-by-layer quantization error analysis."""
import argparse
import json
import sys
from pathlib import Path

import yaml


TOOLS_DIR = Path(__file__).resolve().parents[1] / 'tools'
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import convert_rknn


def resolve_path(value, workspace):
    path = Path(value)
    return path if path.is_absolute() else Path(workspace) / path


def analyze(config_path, workspace, output_root, input_override=None,
            target=None, device_id=None):
    try:
        from rknn.api import RKNN
    except ImportError:
        raise SystemExit('rknn.api is unavailable; use RKNN_PYTHON')

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise SystemExit('RKNN config must be a YAML mapping: {}'.format(
            config_path))

    input_path = convert_rknn.rknn_model_path(config, workspace)
    if not input_path.is_file():
        raise SystemExit('RKNN input ONNX not found: {}'.format(input_path))
    prepared_path, head_type = convert_rknn.prepare_onnx(input_path)
    prepared_path = convert_rknn.standardize_vendor_ops(prepared_path)
    config_kwargs, build_kwargs = convert_rknn.conversion_options(
        config, 'rk3576', head_type)
    do_quantization = build_kwargs['do_quantization']
    if not do_quantization:
        raise SystemExit(
            'RKNN accuracy analysis requires do_quantization: true')

    algorithm = config.get('quant', {}).get('algorithm', 'normal')
    runtime_name = target or 'simulator'
    output_dir = Path(output_root) / algorithm / runtime_name
    dataset_path = output_dir / 'calibration.dataset.txt'
    calibration_images = convert_rknn.write_dataset(
        config, dataset_path, workspace)

    analysis_input = (resolve_path(input_override, workspace)
                      if input_override else calibration_images[0])
    if not analysis_input.is_file():
        raise SystemExit('RKNN analysis input not found: {}'.format(
            analysis_input))

    output_dir.mkdir(parents=True, exist_ok=True)
    print('RKNN config:    {}'.format(config_path))
    print('input ONNX:     {}'.format(input_path))
    print('prepared ONNX:  {} ({})'.format(prepared_path, head_type))
    print('analysis input: {}'.format(analysis_input))
    print('algorithm:      {}'.format(algorithm))
    print('runtime:        {}'.format(runtime_name))
    print('output:         {}'.format(output_dir))

    rknn = RKNN(verbose=True)
    try:
        convert_rknn.checked(rknn.config(**config_kwargs), 'RKNN config')
        convert_rknn.checked(
            rknn.load_onnx(model=str(prepared_path)), 'RKNN load_onnx')
        convert_rknn.checked(
            rknn.build(
                dataset=str(dataset_path) if do_quantization else None,
                **build_kwargs),
            'RKNN build')
        convert_rknn.checked(
            rknn.accuracy_analysis(
                inputs=[str(analysis_input)],
                output_dir=str(output_dir),
                target=target,
                device_id=str(device_id) if target and device_id else None),
            'RKNN accuracy analysis')
    finally:
        rknn.release()

    manifest = {
        'config': str(config_path),
        'source_onnx': str(input_path),
        'prepared_onnx': str(prepared_path),
        'head_type': head_type,
        'analysis_input': str(analysis_input),
        'algorithm': algorithm,
        'target': target,
        'device_id': str(device_id) if device_id else None,
        'output_dir': str(output_dir),
    }
    (output_dir / 'analysis.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--input', type=Path,
                        help='representative image or NPY; default: first calibration image')
    parser.add_argument('--target', choices=('rk3576',),
                        help='connected target; omit for PC simulator')
    parser.add_argument('--device-id', help='connected RK3576 device id')
    args = parser.parse_args()
    analyze(args.config, args.workspace, args.output_dir, args.input,
            args.target, args.device_id)


if __name__ == '__main__':
    main()
