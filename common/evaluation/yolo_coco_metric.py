"""
YOLO COCO metric with NMS + confidence filtering.
- StatlasQuant eval: config uses type=extern_python, source=yolo_coco_metric.py
- Standalone demo:  python yolo_coco_metric.py --config <eval.yaml> [--num N] [--conf C]
"""
import sys, os, json, tempfile, contextlib, io
from collections import defaultdict
import numpy as np
import torch

try:
    from statlas_quant.qat_tool.utils.metrics import Metric
    from statlas_quant.qat_tool.utils.generate_module import _get_absolute_path
except ImportError:
    class Metric:
        """Fallback base so shared decoders work outside Statlas."""

    def _get_absolute_path(path):
        return os.path.abspath(path)


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


def _decode_post_nms(output, img_h, img_w, img_size):
    """For models with built-in NMS. Output: [N, 6] = [x1, y1, x2, y2, conf, cls]
    in input scale. Returns xywh in original image scale."""
    if output.dim() == 3:
        output = output[0]
    bboxes = output[:, 0:4].clone()
    scale_x = float(img_w) / float(img_size[1])
    scale_y = float(img_h) / float(img_size[0])
    bboxes[:, 0] *= scale_x
    bboxes[:, 2] *= scale_x
    bboxes[:, 1] *= scale_y
    bboxes[:, 3] *= scale_y
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    scores = output[:, 4]
    cls = output[:, 5].long()
    return bboxes, scores, cls


def _decode_yolov8_raw(output, img_h, img_w, img_size):
    """For YOLOv8/v11 raw output (no NMS). Shape: [B, 4+nc, N] or [4+nc, N].
    bbox is already DFL-decoded xywh in input scale; class scores are
    raw logits, sigmoid is applied here."""
    if output.dim() == 3:
        output = output[0]  # [4+nc, N] or [N, 4+nc]
    # Detect orientation: channels dim (4+nc) is smaller than anchors dim
    if output.shape[0] < output.shape[1]:
        output = output.transpose(0, 1)  # [N, 4+nc]
    nc = output.shape[1] - 4
    bboxes = output[:, 0:4].clone()
    scale_x = float(img_w) / float(img_size[1])
    scale_y = float(img_h) / float(img_size[0])
    bboxes[:, 0] *= scale_x
    bboxes[:, 2] *= scale_x
    bboxes[:, 1] *= scale_y
    bboxes[:, 3] *= scale_y
    bboxes[:, 0] -= bboxes[:, 2] / 2
    bboxes[:, 1] -= bboxes[:, 3] / 2
    # cls_scores = torch.sigmoid(output[:, 4:4 + nc])
    cls_scores = output[:, 4:4 + nc]
    scores, cls = cls_scores.max(dim=1)
    return bboxes, scores, cls.long()


def _feature_chw(tensor, img_size):
    """Normalize an RKNN/ONNX feature map to CHW using its stride."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError('feature map must be 3D/4D, got {}'.format(
            tuple(tensor.shape)))
    in_h, in_w = img_size
    valid = []
    for candidate in (tensor, tensor.permute(2, 0, 1)):
        _, height, width = candidate.shape
        if (height > 0 and width > 0 and in_h % height == 0 and
                in_w % width == 0 and in_h // height == in_w // width and
                in_h // height in (8, 16, 32)):
            valid.append(candidate)
    if not valid:
        raise ValueError('cannot infer feature layout for {} at input {}'.format(
            tuple(tensor.shape), img_size))
    return min(valid, key=lambda item: item.shape[0])


def _feature_hwc(tensor, img_size):
    return _feature_chw(tensor, img_size).permute(1, 2, 0).contiguous()


def _decode_yolov8_headcut(outputs, img_h, img_w, img_size):
    """Decode head-cut YOLOv8/v11 raw feature maps on host (deployment path).

    `outputs` is the 6-output tuple from the head-cut model: 3 scales x
    (box, cls), each ``[1, C, H, W]``. box has ``C = 4*reg_max`` (DFL),
    cls has ``C = nc``. Does DFL softmax + dist2bbox + sigmoid here so the
    decode matches the FP32 host path used at deployment (the NPU only runs
    the quantized backbone+conv, returning these raw maps).

    Returns bboxes (xywh in original-image scale), scores, cls - same
    contract as the other decoders.
    """
    in_h, in_w = img_size
    # group the 6 outputs by (H, W): each scale has a box and a cls map
    feats = {}
    for t in outputs:
        t = _feature_chw(t, img_size)
        feats.setdefault((t.shape[1], t.shape[2]), []).append(t)

    boxes_all, scores_all, cls_all = [], [], []
    for (h, w), maps in sorted(feats.items()):
        stride = in_h // h
        box = max(maps, key=lambda x: x.shape[0])      # C = 4*reg_max
        cls = min(maps, key=lambda x: x.shape[0])      # C = nc
        reg_max = box.shape[0] // 4
        nc = cls.shape[0]
        n = h * w
        # DFL: [4*reg_max, H, W] -> [4, reg_max, N] -> softmax(reg_max) -> weighted sum
        box = box.reshape(4, reg_max, n).softmax(dim=1)
        proj = torch.arange(reg_max, dtype=box.dtype, device=box.device)
        dist = (box * proj.view(1, reg_max, 1)).sum(dim=1)  # [4, N]
        lt, rb = dist[:2], dist[2:]                          # each [2, N]
        # DFL dist is in grid units (0..reg_max-1); anchor-free grid centers
        # in grid coords, then scale box by stride to input pixels.
        gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                                torch.arange(w, dtype=torch.float32), indexing='ij')
        ax = gx + 0.5
        ay = gy + 0.5
        ax = ax.reshape(-1)
        ay = ay.reshape(-1)
        x1 = ax - lt[0]
        y1 = ay - lt[1]
        x2 = ax + rb[0]
        y2 = ay + rb[1]
        bw = x2 - x1
        bh = y2 - y1
        # grid units -> input pixels (left/top + wh, matching other decoders)
        x1 = x1 * stride
        y1 = y1 * stride
        bw = bw * stride
        bh = bh * stride
        boxes_all.append(torch.stack([x1, y1, bw, bh], dim=1))  # [N, 4]
        cls_s = torch.sigmoid(cls.reshape(nc, n))             # [nc, N]
        scores, cls_idx = cls_s.max(dim=0)                     # [N]
        scores_all.append(scores)
        cls_all.append(cls_idx)

        # ------------------------- Debug -------------------------
        '''
        raw = cls.reshape(nc, n)
        max_logit, max_class = raw.max(dim=0)

        zero_mask = max_logit == 0
        all_zero_mask = (raw == 0).all(dim=0)

        print("max logit zero:", zero_mask.sum().item())
        print("all classes zero:", all_zero_mask.sum().item())

        print(
            "winning classes:",
            torch.bincount(
                max_class[zero_mask],
                minlength=nc,
            ).tolist()
        )

        indices = torch.nonzero(max_logit == 0, as_tuple=False).flatten()

        for index in indices[:100]:
            k = index.item()
            y = k // w
            x = k % w

            print(
                "scale:", (h, w),
                "position:", (y, x),
                "class:", max_class[k].item(),
                "raw:", raw[:, k].tolist(),
            )

        raw_cls = cls.reshape(nc, n)

        max_logit, max_class = raw_cls.max(dim=0)

        print(
            "cls shape:", tuple(cls.shape),
            "dtype:", cls.dtype,
            "device:", cls.device,
        )
        print(
            "raw cls:",
            "min =", raw_cls.min().item(),
            "max =", raw_cls.max().item(),
            "mean =", raw_cls.float().mean().item(),
        )
        print(
            "max logit:",
            "min =", max_logit.min().item(),
            "max =", max_logit.max().item(),
        )

        print("max_logit == 0:",
            torch.isclose(max_logit.float(),
                            torch.tensor(0.0, device=max_logit.device),
                            atol=1e-6).sum().item())

        print("max_logit ~= -0.571:",
            torch.isclose(max_logit.float(),
                            torch.tensor(-0.571, device=max_logit.device),
                            atol=1e-3).sum().item())
        '''
        # ---------------------------------------------------------

    bboxes = torch.cat(boxes_all, dim=0)
    scores = torch.cat(scores_all, dim=0)
    cls = torch.cat(cls_all, dim=0)
    # input scale -> original image scale
    sx = float(img_w) / float(in_w)
    sy = float(img_h) / float(in_h)
    bboxes[:, 0] *= sx
    bboxes[:, 2] *= sx
    bboxes[:, 1] *= sy
    bboxes[:, 3] *= sy
    return bboxes, scores, cls.long()


def _decode_yolov5_headcut(outputs, img_h, img_w, img_size, anchors):
    """Decode the three sigmoid NHWC branches used by RK YOLOv5 runtime."""
    if len(anchors) != 3 or any(len(scale) != 6 for scale in anchors):
        raise ValueError('yolov5_headcut requires three groups of six anchors')
    in_h, in_w = img_size
    features = []
    for tensor in outputs:
        feature = _feature_hwc(tensor, img_size)
        stride_h = in_h // feature.shape[0]
        stride_w = in_w // feature.shape[1]
        if stride_h != stride_w:
            raise ValueError('non-square YOLOv5 stride for {}'.format(
                tuple(feature.shape)))
        features.append((stride_h, feature))

    boxes_all, scores_all, classes_all = [], [], []
    for scale_index, (stride, feature) in enumerate(sorted(features)):
        height, width, channels = feature.shape
        if channels % 3:
            raise ValueError('YOLOv5 channels must be divisible by 3: {}'.format(
                channels))
        properties = channels // 3
        num_classes = properties - 5
        if num_classes <= 0:
            raise ValueError('YOLOv5 output has no class channels')
        prediction = feature.reshape(height, width, 3, properties).float()
        anchor = torch.tensor(
            anchors[scale_index], dtype=prediction.dtype,
            device=prediction.device).reshape(1, 1, 3, 2)
        gy, gx = torch.meshgrid(
            torch.arange(height, dtype=prediction.dtype,
                         device=prediction.device),
            torch.arange(width, dtype=prediction.dtype,
                         device=prediction.device), indexing='ij')
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(2)
        center = (prediction[..., 0:2] * 2.0 - 0.5 + grid) * stride
        size = (prediction[..., 2:4] * 2.0).square() * anchor
        top_left = center - size / 2.0
        boxes_all.append(torch.cat((top_left, size), dim=-1).reshape(-1, 4))
        class_scores, class_ids = prediction[..., 5:].max(dim=-1)
        scores_all.append((prediction[..., 4] * class_scores).reshape(-1))
        classes_all.append(class_ids.reshape(-1))

    bboxes = torch.cat(boxes_all, dim=0)
    scores = torch.cat(scores_all, dim=0)
    classes = torch.cat(classes_all, dim=0).long()
    scale_x = float(img_w) / float(in_w)
    scale_y = float(img_h) / float(in_h)
    bboxes[:, 0] *= scale_x
    bboxes[:, 2] *= scale_x
    bboxes[:, 1] *= scale_y
    bboxes[:, 3] *= scale_y
    return bboxes, scores, classes


DECODERS = {
    'yolox': _decode_yolox,
    'yolov5': _decode_yolov5,
    'yolov8': _decode_yolov5,
    'yolov11': _decode_yolov5,
    'post_nms': _decode_post_nms,
    'yolov8_raw': _decode_yolov8_raw,
    'yolov8_headcut': _decode_yolov8_headcut,
    'yolov5_headcut': _decode_yolov5_headcut,
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
                 visualize_only=False, decoder_options=None, class_map=None):
        from pycocotools.coco import COCO
        annfile = _get_absolute_path(annfile)
        self.coco = COCO(annfile)
        self.class_ids = sorted(self.coco.getCatIds())
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        if decode_mode not in DECODERS:
            raise ValueError('unsupported decode_mode {!r}; choose one of {}'.format(
                decode_mode, sorted(DECODERS)))
        self.decode_fn = DECODERS[decode_mode]
        self.decode_mode = decode_mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_dir = _get_absolute_path(img_dir) if img_dir else None
        self.vis_dir = _get_absolute_path(vis_dir) if vis_dir else None
        self.vis_num = int(vis_num)
        self.vis_conf_threshold = float(vis_conf_threshold)
        self.vis_max_boxes = int(vis_max_boxes)
        self.visualize_only = bool(visualize_only)
        self.decoder_options = decoder_options or {}
        self.class_map = ({int(key): int(value)
                           for key, value in class_map.items()}
                          if class_map else None)
        self.data_list = []
        self.img_ids = set()
        self._tmpfile = None

    def reset(self):
        self.data_list = []
        self.img_ids = set()

    def process_output(self, output, img_h, img_w, img_id):
        self.img_ids.add(int(img_id))
        if self.decode_mode in ('yolov8_headcut', 'yolov5_headcut'):
            bboxes, scores, cls = self.decode_fn(
                output, img_h, img_w, self.img_size,
                **self.decoder_options)
        else:
            if not getattr(self, '_debug_printed', False):
                with open('/tmp/bball_debug.log', 'w') as f:
                    f.write(f'output shape: {output.shape}, dtype: {output.dtype}\n')
                    f.write(f'output stats: min={output.min().item():.6f} max={output.max().item():.6f} mean={output.mean().item():.6f}\n')
                    flat = output if output.dim() == 2 else output[0]
                    if flat.shape[0] < flat.shape[1]:
                        flat = flat.transpose(0, 1)
                    f.write(f'first 3 rows: {flat[:3].tolist()}\n')
                    bboxes_part = flat[:, 0:4]
                    cls_part = flat[:, 4:]
                    f.write(f'\nbbox col range: min={bboxes_part.min().item():.4f} max={bboxes_part.max().item():.4f}\n')
                    f.write(f'cls col range: min={cls_part.min().item():.4f} max={cls_part.max().item():.4f}\n')
                    f.write(f'cls sigmoid max per col: {torch.sigmoid(cls_part).max(dim=0).values.tolist()}\n')
                self._debug_printed = True
            bboxes, scores, cls = self.decode_fn(output, img_h, img_w, self.img_size)
            if not getattr(self, '_debug_decoded', False):
                with open('/tmp/bball_debug.log', 'a') as f:
                    f.write(f'\nafter decode:\n')
                    f.write(f'  bboxes shape: {bboxes.shape}\n')
                    f.write(f'  scores: min={scores.min().item():.6f} max={scores.max().item():.6f}\n')
                    f.write(f'  scores > 0.001: {(scores > 0.001).sum().item()} / {len(scores)}\n')
                    f.write(f'  scores > 0.5: {(scores > 0.5).sum().item()}\n')
                    f.write(f'  top 5 scores: {scores.topk(5).values.tolist()}\n')
                    f.write(f'  cls distribution: {torch.bincount(cls, minlength=4).tolist()}\n')
                self._debug_decoded = True

        # confidence filter
        keep = scores > self.conf_threshold
        if self.class_map is not None:
            cls = torch.tensor(
                [self.class_map.get(int(value), -1) for value in cls],
                dtype=torch.long, device=cls.device)
            keep &= cls >= 0
        else:
            cls = torch.tensor(
                [self.class_ids[int(value)]
                 if 0 <= int(value) < len(self.class_ids) else -1
                 for value in cls], dtype=torch.long, device=cls.device)
            keep &= cls >= 0
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
                'category_id': int(cls[ind]),
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

    # 增加评估指标后处理
    @staticmethod
    def _mean_valid(values):
        """计算COCOeval数组中所有非负有效值的均值。"""
        values = np.asarray(values, dtype=np.float64)
        values = values[values >= 0]

        if values.size == 0:
            return 0.0

        return float(values.mean())

    @staticmethod
    def _best_precision_recall(
        precision_curve,
        recall_thresholds,
    ):
        """
        在IoU=0.5的COCO插值PR曲线上选择F1最高的P/R点。

        注意：
        这里的P/R使用COCOeval插值曲线计算，格式与YOLO相同，
        但数值可能与Ultralytics DetMetrics存在少量差异。
        """
        precision_curve = np.asarray(
            precision_curve,
            dtype=np.float64,
        )

        recall_thresholds = np.asarray(
            recall_thresholds,
            dtype=np.float64,
        )

        valid = precision_curve >= 0

        if not valid.any():
            return 0.0, 0.0

        precision = precision_curve[valid]
        recall = recall_thresholds[valid]

        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall + 1e-16)
        )

        best_index = int(np.argmax(f1))

        return (
            float(precision[best_index]),
            float(recall[best_index]),
        )

    @staticmethod
    def _format_metric_value(value):
        """使用接近Ultralytics日志的数字格式。"""
        return "{:.3g}".format(float(value))

    def _build_yolo_style_table(self, coco_eval):
        """
        根据COCOeval结果生成YOLO风格的总指标和分类指标表格。

        COCOeval precision形状：
            [IoU, Recall, Category, Area, MaxDet]
        """
        precision = coco_eval.eval.get("precision")

        if precision is None:
            return "No COCO precision data."

        evaluated_image_ids = set(
            int(image_id)
            for image_id in coco_eval.params.imgIds
        )

        category_ids = list(
            coco_eval.params.catIds
        )

        categories = {
            category["id"]: category["name"]
            for category in self.coco.loadCats(category_ids)
        }

        recall_thresholds = (
            coco_eval.params.recThrs
        )

        # area=all位于索引0。
        area_index = 0

        # 使用最后一个maxDet，通常为100或配置中的最大值。
        max_det_index = -1

        annotations = [
            annotation
            for annotation
            in self.coco.dataset.get("annotations", [])
            if (
                int(annotation["image_id"])
                in evaluated_image_ids
                and not annotation.get("iscrowd", 0)
            )
        ]

        category_instance_count = {
            category_id: 0
            for category_id in category_ids
        }

        category_image_ids = {
            category_id: set()
            for category_id in category_ids
        }

        for annotation in annotations:
            category_id = annotation["category_id"]

            if category_id not in category_instance_count:
                continue

            category_instance_count[category_id] += 1
            category_image_ids[category_id].add(
                int(annotation["image_id"])
            )

        rows = []

        # ----------------------------------------------------
        # all行
        # ----------------------------------------------------

        all_precision = precision[
            :,
            :,
            :,
            area_index,
            max_det_index,
        ]

        all_map50_95 = self._mean_valid(
            all_precision
        )

        all_map50 = self._mean_valid(
            precision[
                0,
                :,
                :,
                area_index,
                max_det_index,
            ]
        )

        # 聚合所有具有GT的类别的IoU=0.5 PR曲线。
        class_precision_curves = []

        for category_index, category_id in enumerate(
            category_ids
        ):
            if category_instance_count[category_id] == 0:
                continue

            curve = precision[
                0,
                :,
                category_index,
                area_index,
                max_det_index,
            ].astype(np.float64)

            # -1表示该点无效，转换为NaN以便跨类别求均值。
            curve[curve < 0] = np.nan
            class_precision_curves.append(curve)

        if class_precision_curves:
            with np.errstate(
                invalid="ignore",
                divide="ignore",
            ):
                mean_precision_curve = np.nanmean(
                    np.stack(
                        class_precision_curves,
                        axis=0,
                    ),
                    axis=0,
                )

            mean_precision_curve = np.nan_to_num(
                mean_precision_curve,
                nan=-1.0,
            )

            all_precision_value, all_recall_value = (
                self._best_precision_recall(
                    mean_precision_curve,
                    recall_thresholds,
                )
            )
        else:
            all_precision_value = 0.0
            all_recall_value = 0.0

        rows.append(
            (
                "all",
                len(evaluated_image_ids),
                len(annotations),
                all_precision_value,
                all_recall_value,
                all_map50,
                all_map50_95,
            )
        )

        # ----------------------------------------------------
        # 各类别行
        # ----------------------------------------------------

        for category_index, category_id in enumerate(
            category_ids
        ):
            category_precision = precision[
                :,
                :,
                category_index,
                area_index,
                max_det_index,
            ]

            category_map50_95 = self._mean_valid(
                category_precision
            )

            category_map50 = self._mean_valid(
                precision[
                    0,
                    :,
                    category_index,
                    area_index,
                    max_det_index,
                ]
            )

            category_pr_curve = precision[
                0,
                :,
                category_index,
                area_index,
                max_det_index,
            ]

            category_p, category_r = (
                self._best_precision_recall(
                    category_pr_curve,
                    recall_thresholds,
                )
            )

            rows.append(
                (
                    categories.get(
                        category_id,
                        str(category_id),
                    ),
                    len(
                        category_image_ids[
                            category_id
                        ]
                    ),
                    category_instance_count[
                        category_id
                    ],
                    category_p,
                    category_r,
                    category_map50,
                    category_map50_95,
                )
            )

        # ----------------------------------------------------
        # 格式化为Ultralytics风格表格
        # ----------------------------------------------------

        header_format = (
            "{:>22}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
        )

        lines = [
            header_format.format(
                "Class",
                "Images",
                "Instances",
                "Box(P",
                "R",
                "mAP50",
                "mAP50-95",
            )
        ]

        for (
            class_name,
            image_count,
            instance_count,
            precision_value,
            recall_value,
            map50,
            map50_95,
        ) in rows:
            lines.append(
                header_format.format(
                    class_name,
                    image_count,
                    instance_count,
                    self._format_metric_value(
                        precision_value
                    ),
                    self._format_metric_value(
                        recall_value
                    ),
                    self._format_metric_value(
                        map50
                    ),
                    self._format_metric_value(
                        map50_95
                    ),
                )
            )

        return "\n".join(lines)

    def _build_empty_yolo_style_table(self):
        """没有任何预测时生成全0的YOLO风格表格。"""
        evaluated_image_ids = set(
            int(image_id)
            for image_id in self.img_ids
        )

        category_ids = sorted(
            self.coco.getCatIds()
        )

        categories = {
            category["id"]: category["name"]
            for category in self.coco.loadCats(
                category_ids
            )
        }

        annotations = [
            annotation
            for annotation
            in self.coco.dataset.get("annotations", [])
            if (
                int(annotation["image_id"])
                in evaluated_image_ids
                and not annotation.get("iscrowd", 0)
            )
        ]

        header_format = (
            "{:>22}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
            "{:>11}"
        )

        lines = [
            header_format.format(
                "Class",
                "Images",
                "Instances",
                "Box(P",
                "R",
                "mAP50",
                "mAP50-95",
            )
        ]

        lines.append(
            header_format.format(
                "all",
                len(evaluated_image_ids),
                len(annotations),
                "0",
                "0",
                "0",
                "0",
            )
        )

        for category_id in category_ids:
            category_annotations = [
                annotation
                for annotation in annotations
                if annotation["category_id"] == category_id
            ]

            category_images = {
                int(annotation["image_id"])
                for annotation
                in category_annotations
            }

            lines.append(
                header_format.format(
                    categories.get(
                        category_id,
                        str(category_id),
                    ),
                    len(category_images),
                    len(category_annotations),
                    "0",
                    "0",
                    "0",
                    "0",
                )
            )

        return "\n".join(lines)

    # 更新
    def compute(self):
        self.save_visualizations()

        if self.visualize_only:
            return (
                0.0,
                0.0,
                "Visualization only; "
                "no ground-truth metrics.",
            )

        if not self.data_list:
            table = self._build_empty_yolo_style_table()

            return (
                0.0,
                0.0,
                table + "\n\nNo predictions.",
            )

        file_descriptor, temporary_path = (
            tempfile.mkstemp(
                suffix=".json"
            )
        )

        os.close(file_descriptor)

        try:
            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as output_file:
                json.dump(
                    self.data_list,
                    output_file,
                )

            coco_dt = self.coco.loadRes(
                temporary_path
            )

            from pycocotools.cocoeval import COCOeval

            coco_eval = COCOeval(
                self.coco,
                coco_dt,
                "bbox",
            )

            coco_eval.params.imgIds = sorted(
                self.img_ids
            )

            # 捕获COCOeval自己的输出，
            # 避免和YOLO表格交叉在一起。
            coco_output = io.StringIO()

            with contextlib.redirect_stdout(
                coco_output
            ):
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

            yolo_table = (
                self._build_yolo_style_table(
                    coco_eval
                )
            )

            summary = (
                yolo_table
                + "\n\n"
                + "COCOeval summary:\n"
                + coco_output.getvalue()
            )

            return (
                float(coco_eval.stats[0]),
                float(coco_eval.stats[1]),
                summary,
            )

        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    '''
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
    '''


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
        if self.eval.decode_mode == 'yolov8_headcut':
            # head-cut model: preds is the 6-output list for a single batch
            # (box,cls per scale); pass all outputs to the host decoder at once.
            img_h = info_imgs[0][0]
            img_w = info_imgs[1][0]
            img_id = ids[0]
            saved = self.eval.data_list
            saved_img_ids = self.eval.img_ids
            self.eval.data_list = self.data_list[id]
            self.eval.img_ids = self.img_ids[id]
            self.eval.process_output(
                [p.cpu() for p in preds], img_h, img_w, img_id)
            self.data_list[id] = self.eval.data_list
            self.img_ids[id] = self.eval.img_ids
            self.eval.data_list = saved
            self.eval.img_ids = saved_img_ids
            return
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

    # 更新
    def get_result_string(self, is_convert_model=True):
        ap50_95, ap50, summary = self.results[
            0 if is_convert_model else 1
        ]

        if summary:
            return summary

        return (
            "AP50_95: {:.6f}, "
            "AP50: {:.6f}"
        ).format(
            ap50_95,
            ap50,
        )

    # def get_result_string(self, is_convert_model=True):
    #     ap50_95, ap50, summary = self.results[0 if is_convert_model else 1]
    #     return f'AP50_95: {ap50_95}, AP50: {ap50}.\nSummary: {summary}'


# ── Standalone demo mode ──────────────────────────────

def demo():
    """
    Standalone COCO eval driven by YAML config.
    Usage: python yolo_coco_metric.py --config modes/demo/configs/vs859/eval.yaml [--num 50] [--conf 0.25]
    """
    import argparse, torch, yaml
    from torchvision import transforms
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Eval yaml config path')
    ap.add_argument('--model', help='Override model path from the eval config')
    ap.add_argument('--num', type=int, default=0,
                    help='Max images; 0 uses the config limit or all images')
    ap.add_argument('--conf', type=float,
                    help='Override the config confidence threshold')
    ap.add_argument('--vis-dir', help='Override visualization output directory')
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
    def size_pair(value):
        if isinstance(value, int):
            return (value, value)
        return tuple(value)

    img_size = size_pair(m['img_size'])
    decode_mode = m.get('decode_mode', 'yolov5')
    configured_num = d.get('num_samples', 5000)
    resize_size = size_pair(d.get('resize_size', img_size))
    crop_size = size_pair(d.get('crop_size', img_size))

    # Load model - prefer onnx2torch, fall back to onnxruntime on ANY conversion/inference error
    backend = 'torch'
    model = None
    sess = None
    import onnx
    from statlas_quant.third_party.onnx2torch.onnx2torch import convert
    try:
        model = convert(onnx.load(model_path))[0]
        model.eval()
        # Warmup forward to detect runtime issues (e.g. FP16 ops on CPU)
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, *img_size)
            model(_dummy)
    except Exception as exc:
        print(f'onnx2torch unusable ({type(exc).__name__}), falling back to onnxruntime')
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        backend = 'onnxruntime'
        model = None

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

    vis_dir = args.vis_dir or m.get('vis_dir')
    if vis_dir and not os.path.isabs(vis_dir):
        vis_dir = os.path.join(cwd, vis_dir)
    inference_conf = (args.conf if args.conf is not None
                      else m.get('conf_threshold', 0.001))
    evaluator = CocoEvalBase(
        annfile, img_size, decode_mode,
        conf_threshold=inference_conf,
        iou_threshold=m.get('iou_threshold', 0.65),
        img_dir=img_dir,
        vis_dir=vis_dir,
        vis_num=m.get('vis_num', 0),
        vis_conf_threshold=m.get('vis_conf_threshold', 0.25),
        vis_max_boxes=m.get('vis_max_boxes', 100),
        visualize_only=m.get('visualize_only', False))
    img_ids = sorted(evaluator.coco.imgs.keys())
    limits = [value for value in (args.num, configured_num) if value > 0]
    num = min(limits) if limits else len(img_ids)
    num = min(num, len(img_ids))
    if num < len(img_ids):
        import random
        random.seed(42)
        img_ids = random.sample(img_ids, num)

    print(f'Model:  {model_path}')
    print(f'Anns:   {annfile}')
    print(f'Images: {len(img_ids)} from {img_dir}')
    print(f'Size:   resize={resize_size} crop={crop_size} decode={decode_mode}')
    print(f'Confidence: {inference_conf}')
    if vis_dir:
        print(f'Visualizations: {vis_dir}')
    print()

    for i, img_id in enumerate(img_ids):
        info = evaluator.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(img_dir, info['file_name'])).convert('RGB')
        w, h = img.size

        inp = transform(img).unsqueeze(0)
        if backend == 'torch':
            with torch.no_grad():
                out = model(inp)
            if isinstance(out, (list, tuple)) and len(out) > 1:
                # multi-output (head-cut) model: keep all outputs as a list
                out = [o if isinstance(o, torch.Tensor) else o[0]
                       for o in out]
            elif isinstance(out, (list, tuple)):
                out = out[0]
        else:
            input_name = sess.get_inputs()[0].name
            outs = sess.run(None, {input_name: inp.numpy()})
            if len(outs) > 1:
                out = [torch.from_numpy(o) for o in outs]
            else:
                out = torch.from_numpy(outs[0])
        if isinstance(out, dict):
            out = list(out.values())[0]
        if isinstance(out, torch.Tensor):
            out = out[0]

        evaluator.process_output(out, h, w, img_id)

        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(img_ids)}] img_id={img_id} {w}x{h} '
                  f'dets={len(evaluator.data_list)}')

    if args.pred_json:
        pred_json = args.pred_json
        if not os.path.isabs(pred_json):
            pred_json = os.path.join(cwd, pred_json)
        os.makedirs(os.path.dirname(pred_json), exist_ok=True)
        with open(pred_json, 'w', encoding='utf-8') as f:
            json.dump(evaluator.data_list, f, indent=2)
        print('Predictions: {}'.format(pred_json))

    print('\n')
    ap50_95, ap50, summary = evaluator.compute()
    if args.skip_metric:
        return
    print(summary)
    print(f'AP50_95={ap50_95:.4f} AP50={ap50:.4f}')


if __name__ == '__main__':
    demo()
