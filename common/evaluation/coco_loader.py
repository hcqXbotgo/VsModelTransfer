import os
import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image


class CocoEvalDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform, num_samples):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

        if num_samples > 0 and num_samples < len(self.ids):
            import random
            random.seed(42)
            self.ids = random.sample(self.ids, num_samples)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        img_w, img_h = img.size

        if self.transform:
            img = self.transform(img)

        return img, (img_h, img_w, img_id)


def collate_fn(batch):
    images, infos = zip(*batch)
    images = torch.stack(images, dim=0)
    img_hs, img_ws, img_ids = zip(*infos)
    return images, (torch.tensor(0),
                    (torch.tensor(list(img_hs)), torch.tensor(list(img_ws))),
                    torch.tensor(list(img_ids)))


def statlasquant_eval_dataloader(ann_file, img_dir, normalize_mean,
                                  normalize_std, num_samples, batch_size,
                                  num_workers, resize_size, crop_size):
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    dataset = CocoEvalDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        transform=transform,
        num_samples=num_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader
