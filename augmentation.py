"""Create synchronized augmented dermoscopy images and hair masks."""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset with dermoscopic_image/ and hair_mask/")
    parser.add_argument("--out", type=Path, required=True, help="Output folder (source files are not changed)")
    parser.add_argument("--n", type=int, default=10, help="Augmented copies per original image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    images, masks = args.data / "dermoscopic_image", args.data / "hair_mask"
    if not images.is_dir() or not masks.is_dir():
        parser.error("Expected --data/dermoscopic_image and --data/hair_mask")
    out_images, out_masks = args.out / "dermoscopic_image", args.out / "hair_mask"
    out_images.mkdir(parents=True, exist_ok=True); out_masks.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    count = 0
    with (args.out / "augmentation_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["generated_image", "source_image"])
        for path in files:
            mask_path = masks / path.name
            if not mask_path.exists():
                print(f"Skipping {path.name}: matching mask not found"); continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                print(f"Skipping unreadable pair: {path.name}"); continue
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Image/mask size mismatch: {path.name}")
            h, w = mask.shape
            for i in range(args.n):
                aug_i, aug_m = image.copy(), (mask > 127).astype(np.uint8)
                if rng.random() < .5:
                    aug_i, aug_m = cv2.flip(aug_i, 1), cv2.flip(aug_m, 1)
                if rng.random() < .5:
                    aug_i, aug_m = cv2.flip(aug_i, 0), cv2.flip(aug_m, 0)
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-25, 25), rng.uniform(.85, 1.15))
                matrix[:, 2] += (rng.uniform(-.08, .08) * w, rng.uniform(-.08, .08) * h)
                aug_i = cv2.warpAffine(aug_i, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                aug_m = cv2.warpAffine(aug_m, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                aug_i = cv2.convertScaleAbs(aug_i, alpha=rng.uniform(.85, 1.15), beta=rng.uniform(-15, 15))
                name = f"{path.stem}_aug{i + 1:03d}.png"
                cv2.imwrite(str(out_images / name), aug_i)
                cv2.imwrite(str(out_masks / name), (aug_m > 0).astype(np.uint8) * 255)
                writer.writerow([name, path.name]); count += 1
    if not count:
        raise FileNotFoundError("No readable image/mask pairs were found.")
    print(f"Created {count} pairs in {args.out}")


if __name__ == "__main__":
    main()
