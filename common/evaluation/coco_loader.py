import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _sequence_windows(paths, sequence_length=1, frame_step=1,
                      sequence_stride=1, sample_count=0):
    """Build ordered temporal windows; the first path is the target frame."""
    sequence_length = int(sequence_length)
    frame_step = int(frame_step)
    sequence_stride = int(sequence_stride)
    if sequence_length < 1 or frame_step < 1 or sequence_stride < 1:
        raise ValueError('sequence_length, frame_step and sequence_stride must be positive')
    last_start = len(paths) - 1 - (sequence_length - 1) * frame_step
    if last_start < 0:
        raise ValueError(
            'need at least {} images, found {}'.format(
                1 + (sequence_length - 1) * frame_step, len(paths)))
    windows = [
        paths[start:start + sequence_length * frame_step:frame_step]
        for start in range(0, last_start + 1, sequence_stride)
    ]
    if sample_count:
        sample_count = int(sample_count)
        if sample_count < 1:
            raise ValueError('sample_count/num_samples must be positive when set')
        windows = windows[:sample_count]
    return windows


def _sequence_transform(resize_size, crop_size, normalize_mean,
                        normalize_std):
    import torchvision.transforms as transforms

    return transforms.Compose([
        transforms.Resize(tuple(int(v) for v in resize_size)),
        transforms.CenterCrop(tuple(int(v) for v in crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(float(v) for v in normalize_mean),
                             tuple(float(v) for v in normalize_std)),
    ])


class CocoEvalDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform, num_samples,
                 sequence_length=1, frame_step=1, sequence_stride=1,
                 split_inputs=False):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        all_ids = list(sorted(self.coco.imgs.keys()))
        all_paths = [
            Path(self.img_dir) / self.coco.loadImgs(img_id)[0]['file_name']
            for img_id in all_ids
        ]
        if any(not path.is_file() for path in all_paths):
            missing = next(path for path in all_paths if not path.is_file())
            raise FileNotFoundError('COCO image not found: {}'.format(missing))
        windows = _sequence_windows(
            all_paths, sequence_length, frame_step, sequence_stride)
        self.ids = [all_ids[all_paths.index(window[0])] for window in windows]
        self.windows = windows
        self.transform = transform
        self.split_inputs = bool(split_inputs)

        if num_samples > 0 and num_samples < len(self.ids):
            import random
            random.seed(42)
            selected = random.sample(range(len(self.ids)), num_samples)
            self.ids = [self.ids[index] for index in selected]
            self.windows = [self.windows[index] for index in selected]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        with Image.open(self.windows[idx][0]) as image:
            image = image.convert('RGB')
            img_w, img_h = image.size
        frames = []
        for path in self.windows[idx]:
            with Image.open(path) as image:
                frame = image.convert('RGB')
                frames.append(self.transform(frame) if self.transform else frame)
        images = tuple(frames) if self.split_inputs else torch.cat(frames, dim=0)
        return images, (img_h, img_w, img_id)


def collate_fn(batch):
    images, infos = zip(*batch)
    if isinstance(images[0], (tuple, list)):
        images = tuple(torch.stack(frame_batch, dim=0)
                       for frame_batch in zip(*images))
    else:
        images = torch.stack(images, dim=0)
    img_hs, img_ws, img_ids = zip(*infos)
    return images, (torch.tensor(0),
                    (torch.tensor(list(img_hs)), torch.tensor(list(img_ws))),
                    torch.tensor(list(img_ids)))


def calibration_collate_fn(batch):
    """Stack calibration samples while preserving multiple model inputs."""
    if isinstance(batch[0], (tuple, list)):
        return tuple(torch.stack(frame_batch, dim=0)
                     for frame_batch in zip(*batch))
    return torch.stack(batch, dim=0)


def statlasquant_eval_dataloader(ann_file, img_dir, normalize_mean,
                                  normalize_std, num_samples, batch_size,
                                  num_workers, resize_size, crop_size,
                                  sequence_length=1, frame_step=1,
                                  sequence_stride=1, split_inputs=False):
    transform = _sequence_transform(
        resize_size, crop_size, normalize_mean, normalize_std)

    dataset = CocoEvalDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transform=transform,
        num_samples=num_samples,
        sequence_length=sequence_length,
        frame_step=frame_step,
        sequence_stride=sequence_stride,
        split_inputs=split_inputs,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader


class FolderSequenceDataset(Dataset):
    """Image-folder dataset yielding temporal RGB frames.

    By default frames are concatenated on the channel axis.  ``split_inputs``
    returns one tensor per frame for an ONNX graph with multiple inputs.
    """

    def __init__(self, root, transform, sequence_length=1, frame_step=1,
                 sequence_stride=1, sample_count=0, split_inputs=False):
        paths = sorted(
            path for path in Path(root).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError('no calibration images found under {}'.format(root))
        self.windows = _sequence_windows(
            paths, sequence_length, frame_step, sequence_stride, sample_count)
        self.transform = transform
        self.split_inputs = bool(split_inputs)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        frames = []
        for path in self.windows[index]:
            with Image.open(path) as image:
                frames.append(self.transform(image.convert('RGB')))
        return tuple(frames) if self.split_inputs else torch.cat(frames, dim=0)


def statlasquant_calibrate_dataloader(
        root, resize_size=(1024, 3328), crop_size=(1024, 3328),
        normalize_mean=(0.0, 0.0, 0.0), normalize_std=(1.0, 1.0, 1.0),
        sequence_length=1, frame_step=1, sequence_stride=1,
        calibrate_num_sampler=0, batch_size=1, num_workers=0,
        split_inputs=False, **kwargs):
    """Return calibration tensors with ``3 * sequence_length`` channels."""
    del kwargs
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError('calibration image directory not found: {}'.format(root))
    dataset = FolderSequenceDataset(
        root, _sequence_transform(resize_size, crop_size, normalize_mean,
                                  normalize_std), sequence_length, frame_step,
        sequence_stride, calibrate_num_sampler, split_inputs=split_inputs)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                      num_workers=int(num_workers), pin_memory=False,
                      collate_fn=calibration_collate_fn)
