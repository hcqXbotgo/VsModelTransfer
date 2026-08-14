#!/usr/bin/env python3
"""Evaluate an RKNN detection model on a COCO-format dataset."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

try:
    from .yolo_coco_metric import CocoEvalBase
except ImportError:
    from yolo_coco_metric import CocoEvalBase

TOOLS_DIR = Path(__file__).resolve().parents[1] / 'tools'
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import convert_rknn


def resolve_path(value, workspace):
    path = Path(value)
    return path if path.is_absolute() else Path(workspace) / path


def checked(result, operation):
    if result != 0:
        raise SystemExit('{} failed with status {}'.format(operation, result))


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise SystemExit('RKNN eval config must be a YAML mapping: {}'.format(path))
    return config


def runtime_arguments(config, target_override=None, device_override=None):
    runtime = config.get('runtime', {})
    target = target_override if target_override is not None else runtime.get('target')
    device_id = (device_override if device_override is not None
                 else runtime.get('device_id'))
    arguments = {
        'perf_debug': bool(runtime.get('perf_debug', False)),
        'eval_mem': bool(runtime.get('eval_mem', False)),
    }
    if target:
        arguments['target'] = target
    if target and device_id:
        arguments['device_id'] = str(device_id)
    return arguments


def prepare_image(path, input_size, color_order='RGB'):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('cannot decode image: {}'.format(path))
    original_h, original_w = image.shape[:2]
    if color_order.upper() == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_order.upper() != 'BGR':
        raise ValueError('unsupported color_order {!r}'.format(color_order))
    input_h, input_w = input_size
    image = cv2.resize(image, (input_w, input_h),
                       interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(image, axis=0), original_h, original_w


def selected_images(coco, limit):
    image_ids = sorted(coco.getImgIds())
    if limit and limit > 0:
        image_ids = image_ids[:limit]
    return [(image_id, coco.loadImgs(image_id)[0]) for image_id in image_ids]


def initialize_runtime(rknn, config, workspace, runtime_args):
    """Load an exported model on RK3576 or rebuild ONNX for PC simulation."""
    model_path = resolve_path(config['model'], workspace)
    if runtime_args.get('target'):
        checked(rknn.load_rknn(str(model_path)), 'RKNN load_rknn')
        checked(rknn.init_runtime(**runtime_args), 'RKNN init_runtime')
        return 'RK3576 exported model'

    build_config_path = resolve_path(config['build_config'], workspace)
    build_config = yaml.safe_load(
        build_config_path.read_text(encoding='utf-8'))
    input_path = convert_rknn.rknn_model_path(build_config, workspace)
    prepared_path, head_type = convert_rknn.prepare_onnx(input_path)
    prepared_path = convert_rknn.standardize_vendor_ops(prepared_path)
    config_kwargs, build_kwargs = convert_rknn.conversion_options(
        build_config, 'rk3576', head_type)
    do_quantization = build_kwargs['do_quantization']
    dataset_path = model_path.with_suffix('.dataset.txt')
    if do_quantization:
        convert_rknn.write_dataset(
            build_config, dataset_path, workspace)
    rknn.config(**config_kwargs)
    checked(rknn.load_onnx(model=str(prepared_path)), 'RKNN load_onnx')
    checked(rknn.build(
        dataset=str(dataset_path) if do_quantization else None,
        **build_kwargs), 'RKNN build')
    checked(rknn.init_runtime(**runtime_args), 'RKNN init_runtime')
    return 'PC simulator rebuilt from {}'.format(build_config_path)


def write_results(config, workspace, metric, ap50_95, ap50, summary):
    output = config.get('output', {})
    result_file = resolve_path(
        output.get('result_file', 'rknn_metric_result.csv'), workspace)
    predictions_file = resolve_path(
        output.get('predictions_file', 'rknn_predictions.json'), workspace)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'value'])
        writer.writerow(['AP50_95', '{:.8f}'.format(ap50_95)])
        writer.writerow(['AP50', '{:.8f}'.format(ap50)])
    predictions_file.write_text(
        json.dumps(metric.data_list, indent=2) + '\n', encoding='utf-8')
    summary_file = result_file.with_suffix('.txt')
    summary_file.write_text(summary, encoding='utf-8')
    return result_file, predictions_file, summary_file


def evaluate(config_path, workspace, num_override=None,
             target_override=None, device_override=None):
    try:
        from rknn.api import RKNN
    except ImportError:
        raise SystemExit('rknn.api is unavailable; use RKNN_PYTHON')
    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise SystemExit('pycocotools is required for RKNN COCO evaluation')

    config = load_config(config_path)
    model_path = resolve_path(config['model'], workspace)
    dataset = config['dataset']
    annotation_path = resolve_path(dataset['ann_file'], workspace)
    image_dir = resolve_path(dataset['img_dir'], workspace)
    for path, description in ((model_path, 'RKNN model'),
                              (annotation_path, 'COCO annotations'),
                              (image_dir, 'evaluation image directory')):
        if not path.exists():
            raise SystemExit('{} not found: {}'.format(description, path))

    input_size = tuple(int(value) for value in dataset['input_size'])
    if len(input_size) != 2:
        raise SystemExit('dataset.input_size must be [height, width]')
    decode = config['decode']
    output = config.get('output', {})
    decoder_options = {}
    if decode['mode'] == 'yolov5_headcut':
        decoder_options['anchors'] = decode['anchors']
    metric = CocoEvalBase(
        str(annotation_path), input_size, decode_mode=decode['mode'],
        conf_threshold=float(decode.get('conf_threshold', 0.001)),
        iou_threshold=float(decode.get('iou_threshold', 0.65)),
        img_dir=str(image_dir),
        vis_dir=str(resolve_path(output['vis_dir'], workspace))
        if output.get('vis_dir') else None,
        vis_num=int(output.get('vis_num', 0)),
        vis_conf_threshold=float(output.get('vis_conf_threshold', 0.25)),
        vis_max_boxes=int(output.get('vis_max_boxes', 100)),
        decoder_options=decoder_options,
        class_map=decode.get('class_map'))

    coco = COCO(str(annotation_path))
    limit = (num_override if num_override is not None
             else int(dataset.get('num_samples', 0)))
    images = selected_images(coco, limit)
    print('RKNN model: {}'.format(model_path))
    runtime_args = runtime_arguments(config, target_override, device_override)
    print('dataset:    {} image(s)'.format(len(images)))
    print('decode:     {}'.format(decode['mode']))

    rknn = RKNN(verbose=bool(config.get('runtime', {}).get('verbose', False)))
    try:
        runtime_description = initialize_runtime(
            rknn, config, workspace, runtime_args)
        print('runtime:    {}'.format(runtime_description))
        for index, (image_id, info) in enumerate(images, 1):
            image_path = image_dir / info['file_name']
            input_data, image_h, image_w = prepare_image(
                image_path, input_size,
                dataset.get('color_order', 'RGB'))
            outputs = rknn.inference(
                inputs=[input_data], data_format=['nhwc'])
            if outputs is None:
                raise SystemExit('RKNN inference returned no outputs: {}'.format(
                    image_path))
            tensors = [torch.from_numpy(np.asarray(value)).float()
                       for value in outputs]
            prediction = (tensors if decode['mode'].endswith('_headcut')
                          else tensors[0])
            metric.process_output(
                prediction, image_h, image_w, image_id)
            if index == 1 or index % 10 == 0 or index == len(images):
                print('[{}/{}] {}'.format(index, len(images), info['file_name']))
    finally:
        rknn.release()

    ap50_95, ap50, summary = metric.compute()
    result_paths = write_results(
        config, workspace, metric, ap50_95, ap50, summary)
    print(summary)
    print('AP50-95: {:.6f}'.format(ap50_95))
    print('AP50:    {:.6f}'.format(ap50))
    for path in result_paths:
        print('output:  {}'.format(path))
    return ap50_95, ap50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--num', type=int,
                        help='override dataset.num_samples; 0 means all')
    parser.add_argument('--target', help='connected target, for example rk3576')
    parser.add_argument('--device-id', help='connected RK3576 device id')
    args = parser.parse_args()
    evaluate(args.config, args.workspace, args.num, args.target, args.device_id)


if __name__ == '__main__':
    main()
