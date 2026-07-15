"""
YOLO COCO metric with NMS + confidence filtering.
- StatlasQuant eval: config uses type=extern_python, source=yolo_coco_metric.py
- Standalone demo:  python yolo_coco_metric.py --config <eval.yaml> [--num N] [--conf C]
"""
import sys, os, json, tempfile, contextlib, io
from collections import defaultdict
import numpy as np
import torch

from statlas_quant.qat_tool.utils.metrics import Metric
from statlas_quant.qat_tool.utils.generate_module import _get_absolute_path


# ── Decoders ──────────────────────────────────────────

def _decode_yolox(output, img_h, img_w, img_size):
    bboxes = output[:, 0:4].clone()
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes /= scale
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
    return bboxes, scores, cls


def _decode_yolov5(output, img_h, img_w, img_size):
    bboxes = output[:, 0:4].clone()
    bboxes[:, 0] *= float(img_w) / img_size[1]
    bboxes[:, 1] *= float(img_h) / img_size[0]
    bboxes[:, 2] *= float(img_w) / img_size[1]
    bboxes[:, 3] *= float(img_h) / img_size[0]
    bboxes[:, 0] -= bboxes[:, 2] / 2
    bboxes[:, 1] -= bboxes[:, 3] / 2
    cls_scores = output[:, 5:]
    cls = cls_scores.argmax(dim=1)
    scores = output[:, 4] * cls_scores[range(len(cls)), cls]
    return bboxes, scores, cls


DECODERS = {
    'yolox': _decode_yolox,
    'yolov5': _decode_yolov5,
    'yolov8': _decode_yolov5,
    'yolov11': _decode_yolov5,
}

# ── NMS ─────────────────────────────────────────────────

def _box_xywh_to_xyxy(boxes):
    """COCO [x, y, w, h] -> NMS [x1, y1, x2, y2]."""
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    return torch.stack([x1, y1, x2, y2], dim=1)


def _nms_per_class(bboxes_xyxy, scores, cls_ids, iou_threshold):
    """Apply per-class NMS using torchvision. Returns indices to keep."""
    from torchvision.ops import batched_nms
    keep = batched_nms(bboxes_xyxy.float(), scores.float(), cls_ids, iou_threshold)
    return keep


# ── COCO eval core ─────────────────────────────────────

class CocoEvalBase:
    def __init__(self, annfile, img_size, decode_mode='yolov5',
                 conf_threshold=0.001, iou_threshold=0.65,
                 img_dir=None, vis_dir=None, vis_num=0,
                 vis_conf_threshold=0.25, vis_max_boxes=100,
                 visualize_only=False):
        from pycocotools.coco import COCO
        annfile = _get_absolute_path(annfile)
        self.coco = COCO(annfile)
        self.class_ids = sorted(self.coco.getCatIds())
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.decode_fn = DECODERS.get(decode_mode, _decode_yolox)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_dir = _get_absolute_path(img_dir) if img_dir else None
        self.vis_dir = _get_absolute_path(vis_dir) if vis_dir else None
        self.vis_num = int(vis_num)
        self.vis_conf_threshold = float(vis_conf_threshold)
        self.vis_max_boxes = int(vis_max_boxes)
        self.visualize_only = bool(visualize_only)
        self.data_list = []
        self.img_ids = set()
        self._tmpfile = None

    def reset(self):
        self.data_list = []
        self.img_ids = set()

    def process_output(self, output, img_h, img_w, img_id):
        self.img_ids.add(int(img_id))
        bboxes, scores, cls = self.decode_fn(output, img_h, img_w, self.img_size)

        # confidence filter
        keep = scores > self.conf_threshold
        if keep.sum() == 0:
            return
        bboxes, scores, cls = bboxes[keep], scores[keep], cls[keep]

        # per-class NMS
        bboxes_xyxy = _box_xywh_to_xyxy(bboxes)
        keep_nms = _nms_per_class(bboxes_xyxy, scores, cls, self.iou_threshold)
        bboxes, scores, cls = bboxes[keep_nms], scores[keep_nms], cls[keep_nms]

        for ind in range(bboxes.shape[0]):
            self.data_list.append({
                'image_id': int(img_id),
                'category_id': self.class_ids[int(cls[ind])],
                'bbox': bboxes[ind].numpy().tolist(),
                'score': scores[ind].numpy().item(),
                'segmentation': [],
            })

    def save_visualizations(self):
        if not self.img_dir or not self.vis_dir or self.vis_num <= 0:
            return

        from PIL import Image, ImageDraw, ImageFont

        os.makedirs(self.vis_dir, exist_ok=True)
        detections = defaultdict(list)
        for det in self.data_list:
            if det['score'] >= self.vis_conf_threshold:
                detections[det['image_id']].append(det)

        categories = {
            cat['id']: cat['name'] for cat in self.coco.loadCats(self.class_ids)
        }
        font = ImageFont.load_default()
        saved = 0
        for img_id in sorted(self.img_ids)[:self.vis_num]:
            info = self.coco.loadImgs(img_id)[0]
            image_path = os.path.join(self.img_dir, info['file_name'])
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            line_width = max(2, round(min(image.size) / 250))
            image_dets = sorted(
                detections.get(img_id, []),
                key=lambda item: item['score'], reverse=True
            )[:self.vis_max_boxes]

            for det in image_dets:
                x, y, w, h = det['bbox']
                x1 = max(0, min(image.width - 1, x))
                y1 = max(0, min(image.height - 1, y))
                x2 = max(0, min(image.width - 1, x + w))
                y2 = max(0, min(image.height - 1, y + h))
                category_id = det['category_id']
                color = (
                    (37 * category_id + 67) % 256,
                    (17 * category_id + 149) % 256,
                    (29 * category_id + 211) % 256,
                )
                draw.rectangle((x1, y1, x2, y2), outline=color,
                               width=line_width)
                label = '{} {:.3f}'.format(
                    categories.get(category_id, str(category_id)),
                    det['score'])
                label_box = draw.textbbox((x1, y1), label, font=font)
                text_w = label_box[2] - label_box[0]
                text_h = label_box[3] - label_box[1]
                text_y = max(0, y1 - text_h - 4)
                draw.rectangle((x1, text_y, x1 + text_w + 4,
                                text_y + text_h + 4), fill=color)
                draw.text((x1 + 2, text_y + 2), label, fill='white', font=font)

            summary = 'predictions: {}  threshold: {:.3f}'.format(
                len(image_dets), self.vis_conf_threshold)
            summary_box = draw.textbbox((4, 4), summary, font=font)
            draw.rectangle((2, 2, summary_box[2] + 6, summary_box[3] + 6),
                           fill=(0, 0, 0))
            draw.text((4, 4), summary, fill='white', font=font)

            stem = os.path.splitext(os.path.basename(info['file_name']))[0]
            image.save(os.path.join(self.vis_dir, stem + '_pred.jpg'),
                       quality=95)
            saved += 1

        print('Saved {} visualization image(s) to {}'.format(
            saved, self.vis_dir))

    def compute(self):
        self.save_visualizations()
        if self.visualize_only:
            return 0.0, 0.0, 'Visualization only; no ground-truth metrics.'
        if not self.data_list:
            return 0.0, 0.0, 'No predictions.'
        _, tmp = tempfile.mkstemp()
        json.dump(self.data_list, open(tmp, 'w'))
        cocoDt = self.coco.loadRes(tmp)
        from pycocotools.cocoeval import COCOeval
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.params.imgIds = sorted(self.img_ids)
        cocoEval.evaluate()
        cocoEval.accumulate()
        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            cocoEval.summarize()
        os.unlink(tmp)
        return cocoEval.stats[0], cocoEval.stats[1], s.getvalue()


# ── StatlasQuant metric entry point ────────────────────

class metric_target(Metric):
    """StatlasQuant extern_python metric. Supports decode_mode, conf_threshold, iou_threshold params."""

    def __init__(self, annfile, img_size=(416, 416),
                 decode_mode='yolov5', conf_threshold=0.001,
                 iou_threshold=0.65, img_dir=None, vis_dir=None,
                 vis_num=0, vis_conf_threshold=0.25,
                 vis_max_boxes=100, visualize_only=False, **kwargs):
        super().__init__()
        self.eval = CocoEvalBase(annfile, img_size, decode_mode,
                                 conf_threshold=conf_threshold,
                                 iou_threshold=iou_threshold,
                                 img_dir=img_dir, vis_dir=vis_dir,
                                 vis_num=vis_num,
                                 vis_conf_threshold=vis_conf_threshold,
                                 vis_max_boxes=vis_max_boxes,
                                 visualize_only=visualize_only)
        self.data_list = [[], []]
        self.img_ids = [set(), set()]
        self.results = [[-1, -1, ''], [-1, -1, '']]
        self.reset(True)
        self.reset(False)

    def reset(self, is_convert_model=True):
        id = 0 if is_convert_model else 1
        self.data_list[id] = []
        self.img_ids[id] = set()
        self.results[id] = [-1, -1, '']

    def update(self, preds, target, is_convert_model=True):
        id = 0 if is_convert_model else 1
        _, info_imgs, ids = target
        for (output, img_h, img_w, img_id) in zip(
                preds, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue
            output = output.cpu()
            saved = self.eval.data_list
            saved_img_ids = self.eval.img_ids
            self.eval.data_list = self.data_list[id]
            self.eval.img_ids = self.img_ids[id]
            self.eval.process_output(output, img_h, img_w, img_id)
            self.data_list[id] = self.eval.data_list
            self.img_ids[id] = self.eval.img_ids
            self.eval.data_list = saved
            self.eval.img_ids = saved_img_ids

    def compute(self, is_convert_model=True):
        id = 0 if is_convert_model else 1
        if len(self.data_list[id]) > 0:
            saved = self.eval.data_list
            saved_img_ids = self.eval.img_ids
            self.eval.data_list = self.data_list[id]
            self.eval.img_ids = self.img_ids[id]
            ap50_95, ap50, info = self.eval.compute()
            self.data_list[id] = self.eval.data_list
            self.img_ids[id] = self.eval.img_ids
            self.eval.data_list = saved
            self.eval.img_ids = saved_img_ids
            self.results[id] = [ap50_95, ap50, info]
        else:
            self.results[id] = [0, 0, '']

    def get_result(self, is_convert_model=True):
        return self.results[0 if is_convert_model else 1]

    def get_result_dict(self, is_convert_model=True):
        ap50_95, ap50, _ = self.results[0 if is_convert_model else 1]
        return {'AP50_95': ap50_95, 'AP50': ap50}

    def get_result_string(self, is_convert_model=True):
        ap50_95, ap50, summary = self.results[0 if is_convert_model else 1]
        return f'AP50_95: {ap50_95}, AP50: {ap50}.\nSummary: {summary}'


# ── Standalone demo mode ──────────────────────────────

def demo():
    """
    Standalone COCO eval driven by YAML config.
    Usage: python yolo_coco_metric.py --config modes/demo/configs/eval.yaml [--num 50] [--conf 0.25]
    """
    import argparse, torch, yaml
    from torchvision import transforms
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Eval yaml config path')
    ap.add_argument('--model', help='Override model path from the eval config')
    ap.add_argument('--num', type=int, default=50, help='Max images')
    ap.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    ap.add_argument('--pred-json', help='Write decoded predictions as COCO JSON')
    ap.add_argument('--skip-metric', action='store_true',
                    help='Skip COCO metric computation (for unlabeled images)')
    args = ap.parse_args()

    # Read yaml, resolve relative paths from CWD (same as StatlasQuant)
    cwd = os.getcwd()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model or cfg['model']['onnx_model']
    d = cfg['dataset']['eval']['parameters']
    m = cfg['metric']['parameters']

    if not os.path.isabs(model_path):
        model_path = os.path.join(cwd, model_path)
    annfile = d['ann_file'] if os.path.isabs(d['ann_file']) else os.path.join(cwd, d['ann_file'])
    img_dir = d['img_dir'] if os.path.isabs(d['img_dir']) else os.path.join(cwd, d['img_dir'])
    img_size = tuple(m['img_size'])
    decode_mode = m.get('decode_mode', 'yolov5')
    configured_num = d.get('num_samples', 5000)
    num = args.num if configured_num <= 0 else min(args.num, configured_num)
    resize_size = tuple(d.get('resize_size', img_size))
    crop_size = tuple(d.get('crop_size', img_size))

    print(f'Model:  {model_path}')
    print(f'Anns:   {annfile}')
    print(f'Images: {num} from {img_dir}')
    print(f'Size:   resize={resize_size} crop={crop_size} decode={decode_mode}')
    print()

    # Load model
    import onnx
    from statlas_quant.third_party.onnx2torch.onnx2torch import convert
    model = convert(onnx.load(model_path))[0]
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

    evaluator = CocoEvalBase(annfile, img_size, decode_mode)
    img_ids = sorted(evaluator.coco.imgs.keys())
    if num < len(img_ids):
        import random
        random.seed(42)
        img_ids = random.sample(img_ids, num)

    for i, img_id in enumerate(img_ids):
        info = evaluator.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(img_dir, info['file_name'])).convert('RGB')
        w, h = img.size

        inp = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
        if isinstance(out, dict):
            out = list(out.values())[0]
        out = out[0]

        # Confidence filter
        cls_scores = out[:, 5:]
        scores = out[:, 4] * cls_scores.max(dim=1).values
        keep = scores > args.conf
        if keep.sum() > 0:
            evaluator.process_output(out[keep], h, w, img_id)

        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(img_ids)}] img_id={img_id} {w}x{h} '
                  f'dets={keep.sum().item()}')

    if args.pred_json:
        pred_json = args.pred_json
        if not os.path.isabs(pred_json):
            pred_json = os.path.join(cwd, pred_json)
        os.makedirs(os.path.dirname(pred_json), exist_ok=True)
        with open(pred_json, 'w', encoding='utf-8') as f:
            json.dump(evaluator.data_list, f, indent=2)
        print('Predictions: {}'.format(pred_json))

    if args.skip_metric:
        return

    print()
    ap50_95, ap50, summary = evaluator.compute()
    print(summary)
    print(f'AP50_95={ap50_95:.4f} AP50={ap50:.4f}')


if __name__ == '__main__':
    demo()
