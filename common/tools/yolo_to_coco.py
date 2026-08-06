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

Usage:
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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, type=Path,
                    help='YOLO dataset dir with <stem>.<ext> + <stem>.txt pairs')
    ap.add_argument('--img-out', required=True, type=Path,
                    help='Destination dir for images (created if missing)')
    ap.add_argument('--ann-out', required=True, type=Path,
                    help='Destination COCO instances.json path (created if missing)')
    ap.add_argument('--names', required=True, nargs='+',
                    help='Ordered class names: position i = model class i '
                         '= COCO category_id i+1. Must match the model names.')
    ap.add_argument('--ext', default='jpg',
                    help='Image extension to look for (default: jpg)')
    ap.add_argument('--no-copy', action='store_true',
                    help='Do not copy images (only write instances.json; '
                         'img-out must already contain the images)')
    ap.add_argument('--description', default='',
                    help='COCO info.description text')
    return ap.parse_args()


def main():
    args = parse_args()

    if len(set(args.names)) != len(args.names):
        sys.exit('error: --names contains duplicates: {}'.format(args.names))

    # categories in training order, 1-based contiguous ids
    categories = [
        {'id': i + 1, 'name': name, 'supercategory': name}
        for i, name in enumerate(args.names)
    ]
    name_for_id = {c['id']: c['name'] for c in categories}

    in_dir = args.input
    if not in_dir.is_dir():
        sys.exit('error: input dir not found: {}'.format(in_dir))

    images = sorted(in_dir.glob('*.{}'.format(args.ext.lstrip('.'))))
    if not images:
        sys.exit('error: no *.{} images found in {}'.format(args.ext, in_dir))

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
        label_path = img_path.with_suffix('.txt')
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
