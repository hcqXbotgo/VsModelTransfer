"""Maintain the small 3328x1024 soccer calibration/evaluation dataset."""
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def image_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def inspect_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.verify()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.width, image.height


def copy_unique(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if target.exists() and target.read_bytes() != source.read_bytes():
        stem, suffix = source.stem, source.suffix.lower()
        index = 2
        while target.exists():
            target = destination / '{}_{:02d}{}'.format(stem, index, suffix)
            index += 1
    shutil.copy2(source, target)
    return target


def empty_coco(categories, description):
    return {
        'info': {'description': description, 'version': '1.0'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': categories,
    }


def seed(args):
    root = Path(args.root)
    approved_dir = root / 'evaluation' / 'images'
    draft_dir = root / 'draft' / 'images'
    calibration_dir = root / 'calibration' / 'images'
    for path in (approved_dir, draft_dir, calibration_dir,
                 root / 'evaluation' / 'annotations',
                 root / 'draft' / 'annotations', root / 'reports'):
        path.mkdir(parents=True, exist_ok=True)

    coco = load_json(args.coco_annotations)
    predictions = load_json(args.predictions)
    coco_images = {item['file_name']: item for item in coco['images']}
    coco_annotations = {}
    for annotation in coco['annotations']:
        coco_annotations.setdefault(annotation['image_id'], []).append(annotation)

    manifest = load_json(args.image_manifest)
    manifest_images = {item['id']: item for item in manifest['images']}
    predictions_by_image = {}
    for prediction in predictions:
        predictions_by_image.setdefault(prediction['image_id'], []).append(prediction)

    approved = empty_coco(coco['categories'], 'Reviewed soccer evaluation set')
    draft = empty_coco(coco['categories'], 'Unreviewed soccer pre-annotations')
    next_approved_ann = 1
    next_draft_ann = 1

    for source in image_files(args.images):
        width, height = inspect_image(source)
        if source.name in coco_images:
            original = coco_images[source.name]
            target = copy_unique(source, approved_dir)
            image_id = len(approved['images']) + 1
            approved['images'].append({
                'id': image_id,
                'file_name': target.name,
                'width': width,
                'height': height,
                'source': 'coco_val2017',
                'review_status': 'approved',
            })
            for item in coco_annotations.get(original['id'], []):
                annotation = dict(item)
                annotation['id'] = next_approved_ann
                annotation['image_id'] = image_id
                annotation['source'] = 'coco_ground_truth'
                approved['annotations'].append(annotation)
                next_approved_ann += 1
        else:
            target = copy_unique(source, draft_dir)
            copy_unique(source, calibration_dir)
            image_id = len(draft['images']) + 1
            draft['images'].append({
                'id': image_id,
                'file_name': target.name,
                'width': width,
                'height': height,
                'source': 'soccer_capture',
                'review_status': 'needs_review',
            })
            manifest_id = next((item_id for item_id, item in manifest_images.items()
                                if item['file_name'] == source.name), None)
            for item in predictions_by_image.get(manifest_id, []):
                if item['score'] < args.draft_confidence:
                    continue
                x, y, box_width, box_height = item['bbox']
                draft['annotations'].append({
                    'id': next_draft_ann,
                    'image_id': image_id,
                    'category_id': item['category_id'],
                    'bbox': [x, y, box_width, box_height],
                    'area': box_width * box_height,
                    'iscrowd': 0,
                    'segmentation': [],
                    'source': 'float_model_preannotation',
                    'preannotation_score': item['score'],
                    'review_status': 'needs_review',
                })
                next_draft_ann += 1

    save_json(root / 'evaluation' / 'annotations' / 'instances.json', approved)
    save_json(root / 'draft' / 'annotations' / 'instances.json', draft)
    print('approved: {} images, {} annotations'.format(
        len(approved['images']), len(approved['annotations'])))
    print('draft: {} images, {} pre-annotations'.format(
        len(draft['images']), len(draft['annotations'])))
    print('calibration: {} images'.format(len(image_files(calibration_dir))))


def add_images(args):
    root = Path(args.root)
    destination = root / 'calibration' / 'images'
    copied = 0
    for value in args.paths:
        for source in image_files(value):
            try:
                width, height = inspect_image(source)
            except Exception as error:
                print('REJECT {}: {}'.format(source, error))
                continue
            target = copy_unique(source, destination)
            print('ADD {} -> {} ({}x{})'.format(source, target, width, height))
            copied += 1
    print('added {} image(s)'.format(copied))


def import_coco(args):
    root = Path(args.root)
    target_path = root / 'evaluation' / 'annotations' / 'instances.json'
    target_images = root / 'evaluation' / 'images'
    target = load_json(target_path)
    incoming = load_json(args.annotations)
    incoming_dir = Path(args.images)

    unreviewed_images = [
        item for item in incoming.get('images', [])
        if item.get('review_status') == 'needs_review'
    ]
    unreviewed_annotations = [
        item for item in incoming.get('annotations', [])
        if item.get('review_status') == 'needs_review'
    ]
    if unreviewed_images or unreviewed_annotations:
        raise SystemExit(
            'refusing unreviewed annotations: {} image(s), {} annotation(s) '
            'still have review_status=needs_review; review/export them first'.format(
                len(unreviewed_images), len(unreviewed_annotations)))

    target_category_by_name = {
        item['name']: item['id'] for item in target['categories']
    }
    incoming_category_name = {
        item['id']: item['name'] for item in incoming['categories']
    }
    existing_names = {item['file_name'] for item in target['images']}
    annotations_by_image = {}
    for annotation in incoming['annotations']:
        annotations_by_image.setdefault(annotation['image_id'], []).append(annotation)

    next_image_id = max((item['id'] for item in target['images']), default=0) + 1
    next_annotation_id = max(
        (item['id'] for item in target['annotations']), default=0) + 1
    imported_images = 0
    imported_annotations = 0

    for image in incoming['images']:
        if image['file_name'] in existing_names:
            print('SKIP existing image:', image['file_name'])
            continue
        source = incoming_dir / image['file_name']
        if not source.exists():
            raise SystemExit('missing import image: {}'.format(source))
        width, height = inspect_image(source)
        target_file = copy_unique(source, target_images)
        new_image_id = next_image_id
        next_image_id += 1
        target['images'].append({
            'id': new_image_id,
            'file_name': target_file.name,
            'width': width,
            'height': height,
            'source': args.source,
            'review_status': 'approved',
        })
        existing_names.add(target_file.name)
        imported_images += 1

        for annotation in annotations_by_image.get(image['id'], []):
            category_name = incoming_category_name[annotation['category_id']]
            if category_name not in target_category_by_name:
                raise SystemExit('unknown category name: {}'.format(category_name))
            new_annotation = dict(annotation)
            new_annotation['id'] = next_annotation_id
            new_annotation['image_id'] = new_image_id
            new_annotation['category_id'] = target_category_by_name[category_name]
            new_annotation['source'] = args.source
            new_annotation.pop('score', None)
            new_annotation.pop('preannotation_score', None)
            new_annotation.pop('review_status', None)
            target['annotations'].append(new_annotation)
            next_annotation_id += 1
            imported_annotations += 1

    save_json(target_path, target)
    print('imported {} image(s), {} annotation(s)'.format(
        imported_images, imported_annotations))


def validate(args):
    data = load_json(args.annotations)
    image_dir = Path(args.images)
    categories = {item['id']: item['name'] for item in data['categories']}
    images = {item['id']: item for item in data['images']}
    errors = []
    counts = Counter()

    for image in images.values():
        path = image_dir / image['file_name']
        if not path.exists():
            errors.append('missing image: {}'.format(path))
            continue
        width, height = inspect_image(path)
        if (width, height) != (image['width'], image['height']):
            errors.append('size mismatch: {}'.format(image['file_name']))

    annotation_ids = set()
    for annotation in data['annotations']:
        if annotation['id'] in annotation_ids:
            errors.append('duplicate annotation id: {}'.format(annotation['id']))
        annotation_ids.add(annotation['id'])
        image = images.get(annotation['image_id'])
        if image is None:
            errors.append('unknown image id: {}'.format(annotation['image_id']))
            continue
        if annotation['category_id'] not in categories:
            errors.append('unknown category id: {}'.format(annotation['category_id']))
            continue
        x, y, width, height = annotation['bbox']
        if width <= 0 or height <= 0:
            errors.append('non-positive bbox: annotation {}'.format(annotation['id']))
        if x < 0 or y < 0 or x + width > image['width'] + 1 or y + height > image['height'] + 1:
            errors.append('out-of-bounds bbox: annotation {}'.format(annotation['id']))
        counts[categories[annotation['category_id']]] += 1

    print('images={}, annotations={}, classes={}'.format(
        len(images), len(data['annotations']), dict(counts)))
    if errors:
        for error in errors:
            print('ERROR:', error)
        raise SystemExit('validation failed with {} error(s)'.format(len(errors)))
    print('validation passed')


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    seed_parser = subparsers.add_parser('seed')
    seed_parser.add_argument('--root', required=True)
    seed_parser.add_argument('--images', required=True)
    seed_parser.add_argument('--image-manifest', required=True)
    seed_parser.add_argument('--coco-annotations', required=True)
    seed_parser.add_argument('--predictions', required=True)
    seed_parser.add_argument('--draft-confidence', type=float, default=0.25)
    seed_parser.set_defaults(func=seed)

    add_parser = subparsers.add_parser('add')
    add_parser.add_argument('--root', required=True)
    add_parser.add_argument('--kind', choices=('calibration',), required=True)
    add_parser.add_argument('paths', nargs='+')
    add_parser.set_defaults(func=add_images)

    import_parser = subparsers.add_parser('import-coco')
    import_parser.add_argument('--root', required=True)
    import_parser.add_argument('--annotations', required=True)
    import_parser.add_argument('--images', required=True)
    import_parser.add_argument('--source', default='manual_coco_annotation')
    import_parser.set_defaults(func=import_coco)

    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('--annotations', required=True)
    validate_parser.add_argument('--images', required=True)
    validate_parser.set_defaults(func=validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
