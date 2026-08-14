#!/usr/bin/env python3
"""Convert a YOLO detection dataset (image + .txt label pairs) to COCO format.

YOLO label files store one box per line as ``class cx cy w h`` with all
geometry normalized to [0, 1] (cx/cy = box center, w/h = box size). This
script reads every ``<stem>.<ext>`` + ``<stem>.txt`` pair in an input
directory, copies the images into an output directory, and writes a COCO
``instances.json`` next to them.

Class-index mapping (the correctness-critical part):
    The repo's metric (``common/evaluation/yolo_coco_metric.py``) maps a
    model's predicted class index ``cls`` to a COCO category via
    ``sorted(getCatIds())[cls]``. So the COCO categories MUST be listed in
    the same order the model was trained, with contiguous 1-based ids, i.e.
    category_id = class_index + 1. ``--names`` takes that ordered list
    (position 0 -> class 0 -> category_id 1). Verify it against the ONNX
    model's ``names`` metadata before running, e.g.:

        python -c "import onnx;print(onnx.load(MODEL).metadata_props)" | grep names

Simple usage:
    python yolo_to_coco.py /path/to/test

Specify class names explicitly (recommended for formal evaluation):
    python yolo_to_coco.py /path/to/test \
        --names person ball hoop ballhoop

For input ``/path/to/test``, this creates ``/path/to/evaluation/images`` and
``/path/to/evaluation/annotations/instances.json`` next to the input directory.
The input may contain ``images/`` and ``labels/`` subdirectories, or
image/label pairs in the same directory. Class names are read from a nearby
``data.yaml`` when available; otherwise ``class_0``, ``class_1``, ... are used.

Advanced/legacy usage:
    python yolo_to_coco.py \
        --input /path/to/yolo_dir \
        --img-out modes/<mode>/datasets/evaluation/images \
        --ann-out modes/<mode>/datasets/evaluation/annotations/instances.json \
        --names person ball hoop ballhoop \
        [--ext jpg] [--no-copy] [--description "..."]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage='%(prog)s [-h] [--names NAME [NAME ...]] TEST_DIR')
    ap.add_argument('dataset', nargs='?', type=Path, metavar='TEST_DIR',
                    help='Test dataset directory (simple mode)')
    ap.add_argument('--input', type=Path,
                    help='YOLO dataset dir with <stem>.<ext> + <stem>.txt pairs')
    ap.add_argument('--img-out', type=Path,
                    help='Destination dir for images (created if missing)')
    ap.add_argument('--ann-out', type=Path,
                    help='Destination COCO instances.json path (created if missing)')
    ap.add_argument('--names', nargs='+', metavar='NAME',
                    help='Optional ordered class names, for example: '
                         '--names person ball hoop ballhoop. Position i = '
                         'model class i = COCO category_id i+1; the order '
                         'must match model training. If omitted, names are '
                         'read from data.yaml or generated as class_N.')
    ap.add_argument('--ext',
                    help='Only process this image extension (default: common image formats)')
    ap.add_argument('--no-copy', action='store_true',
                    help='Do not copy images (only write instances.json; '
                         'img-out must already contain the images)')
    ap.add_argument('--description', default='',
                    help='COCO info.description text')
    return ap.parse_args()


def _dataset_dirs(input_dir):
    """Return image and label directories for common YOLO layouts."""
    images_dir = input_dir / 'images'
    if images_dir.is_dir():
        labels_dir = input_dir / 'labels'
        return images_dir, labels_dir if labels_dir.is_dir() else images_dir
    return input_dir, input_dir


def _find_images(images_dir, extension):
    if extension:
        return sorted(images_dir.glob('*.{}'.format(extension.lstrip('.'))))
    return sorted(path for path in images_dir.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _names_from_data_yaml(input_dir):
    candidates = (input_dir / 'data.yaml', input_dir / 'dataset.yaml',
                  input_dir.parent / 'data.yaml',
                  input_dir.parent / 'dataset.yaml')
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        except (ImportError, OSError, ValueError):
            continue
        names = data.get('names')
        if isinstance(names, list):
            return [str(name) for name in names]
        if isinstance(names, dict):
            try:
                return [str(names[index]) for index in
                        sorted(names, key=lambda value: int(value))]
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _infer_names(images, labels_dir, configured_names, dataset_dir):
    if configured_names:
        return configured_names
    class_ids = set()
    for image_path in images:
        label_path = labels_dir / (image_path.stem + '.txt')
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if parts:
                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    sys.exit('error: invalid class id in {}: {!r}'.format(
                        label_path, line))
    if not class_ids:
        sys.exit('error: cannot infer classes: no labeled objects found')
    max_class_id = max(class_ids)
    names = _names_from_data_yaml(dataset_dir)
    if names is not None:
        if len(names) <= max_class_id:
            sys.exit('error: data.yaml has {} names but labels use class id {}'.format(
                len(names), max_class_id))
        return names
    return ['class_{}'.format(index) for index in range(max_class_id + 1)]


def main():
    args = parse_args()

    if args.dataset is not None and args.input is not None:
        sys.exit('error: use either the dataset path or --input, not both')
    in_dir = args.dataset or args.input
    if in_dir is None:
        sys.exit('error: test dataset directory is required')
    if not in_dir.is_dir():
        sys.exit('error: input dir not found: {}'.format(in_dir))

    simple_mode = args.dataset is not None
    if simple_mode:
        output_root = in_dir.resolve().parent / 'evaluation'
        args.img_out = args.img_out or output_root / 'images'
        args.ann_out = args.ann_out or output_root / 'annotations' / 'instances.json'
    elif args.img_out is None or args.ann_out is None:
        sys.exit('error: --img-out and --ann-out are required with --input')

    images_dir, labels_dir = _dataset_dirs(in_dir)
    images = _find_images(images_dir, args.ext)
    if not images:
        expected = '*.{}'.format(args.ext.lstrip('.')) if args.ext else 'images'
        sys.exit('error: no {} found in {}'.format(expected, images_dir))
    args.names = _infer_names(images, labels_dir, args.names, in_dir)

    if len(set(args.names)) != len(args.names):
        sys.exit('error: --names contains duplicates: {}'.format(args.names))

    # categories in training order, 1-based contiguous ids
    categories = [
        {'id': i + 1, 'name': name, 'supercategory': name}
        for i, name in enumerate(args.names)
    ]
    name_for_id = {c['id']: c['name'] for c in categories}

    if not args.no_copy:
        args.img_out.mkdir(parents=True, exist_ok=True)
    elif not args.img_out.is_dir():
        sys.exit('error: --no-copy given but img-out does not exist: {}'.format(args.img_out))

    coco_images = []
    coco_annotations = []
    ann_id = 1
    skipped_no_label = 0
    cls_counts = {c['id']: 0 for c in categories}

    for img_id, img_path in enumerate(images, start=1):
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            print('  warn: no label for {} (skipped)'.format(img_path.name),
                  file=sys.stderr)
            skipped_no_label += 1
            continue

        # copy image
        dst_img = args.img_out / img_path.name
        if not args.no_copy:
            shutil.copy2(img_path, dst_img)

        # real dimensions from the pixel data (eval scales preds by these)
        with Image.open(img_path) as im:
            width, height = im.size

        coco_images.append({
            'id': img_id,
            'file_name': img_path.name,
            'width': width,
            'height': height,
        })

        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    sys.exit('error: bad label line in {} (expected 5 cols, '
                             'got {}: {!r})'.format(label_path, len(parts), line))
                c = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:])
                if c < 0 or c >= len(categories):
                    sys.exit('error: class id {} out of range [0,{}] in {}'
                             .format(c, len(categories) - 1, label_path))
                for v in (cx, cy, w, h):
                    if v < 0.0 or v > 1.0:
                        sys.exit('error: normalized coord out of [0,1] in {}: '
                                 '{!r}'.format(label_path, line))

                # YOLO cxcywh (normalized) -> COCO xywh (absolute, top-left)
                abs_w = w * width
                abs_h = h * height
                x = (cx - w / 2.0) * width
                y = (cy - h / 2.0) * height
                category_id = c + 1  # 1-based, in --names order
                coco_annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': category_id,
                    'bbox': [round(x, 2), round(y, 2),
                             round(abs_w, 2), round(abs_h, 2)],
                    'area': round(abs_w * abs_h, 2),
                    'iscrowd': 0,
                    'segmentation': [],
                })
                ann_id += 1
                cls_counts[category_id] += 1

    coco = {
        'info': {
            'description': args.description or 'YOLO->COCO converted dataset',
        },
        'categories': categories,
        'images': coco_images,
        'annotations': coco_annotations,
    }

    args.ann_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.ann_out, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)

    # summary
    print('images: {}'.format(len(coco_images)))
    print('annotations: {}'.format(len(coco_annotations)))
    print('categories (id: name):')
    for c in categories:
        print('  {}: {} ({} boxes)'.format(c['id'], c['name'], cls_counts[c['id']]))
    if skipped_no_label:
        print('skipped (no label): {}'.format(skipped_no_label))
    if not args.no_copy:
        print('images copied to: {}'.format(args.img_out))
    print('instances.json: {}'.format(args.ann_out))


if __name__ == '__main__':
    main()
