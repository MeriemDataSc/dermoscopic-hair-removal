"""Hair localization from HSV-V using the Gabor flowchart in the project."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# One (sigma_x, sigma_y, spatial frequency u) tuple for each orientation
# shown in the user's filter-bank figure, in the same order as ANGLES.
GABOR_BANK = (
    (0,   2.0, 6.0, 0.030),
    (45,  2.0, 4.0, 0.060),
    (90,  2.0, 6.0, 0.065),
    (135, 2.0, 6.0, 0.070),
    (157, 2.0, 6.0, 0.070),
)
ANGLES = tuple(item[0] for item in GABOR_BANK)


def prepare_image(image, variance_threshold=0.01, median_kernel=3, mean_kernel=3):
    """HSV -> V -> median -> conditional mean smoothing.

    Variance is measured globally on the median-filtered V channel normalized
    to [0, 1]. Mean smoothing is applied only when variance > Tv.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    median = cv2.medianBlur(value, median_kernel)
    variance = float(np.var(median.astype(np.float32) / 255.0))
    if variance > variance_threshold:
        prepared = cv2.blur(median, (mean_kernel, mean_kernel))
        smoothed = True
    else:
        prepared = median
        smoothed = False
    return prepared, variance, smoothed


def gabor_responses(prepared):
    """Apply the five anisotropic Gabor kernels specified for this filter bank.

    Convention: sigma_x is the narrow width across a hair; sigma_y is the
    elongated envelope along it. ``u`` is spatial frequency in cycles/pixel,
    so the cosine carrier is cos(2*pi*u*distance_across_hair).
    """
    source = prepared.astype(np.float32) / 255.0
    responses = {}
    for angle, sigma_x, sigma_y, frequency in GABOR_BANK:
        # Include roughly three standard deviations in both directions.
        radius = int(np.ceil(3 * max(sigma_x, sigma_y)))
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        theta = np.deg2rad(angle)
        # x_along follows the hair axis; y_across is perpendicular to it.
        x_along = xx * np.cos(theta) + yy * np.sin(theta)
        y_across = -xx * np.sin(theta) + yy * np.cos(theta)
        envelope = np.exp(-0.5 * ((y_across / sigma_x) ** 2 + (x_along / sigma_y) ** 2))
        kernel = envelope * np.cos(2 * np.pi * frequency * y_across)
        kernel -= kernel.mean()
        kernel /= np.sum(np.abs(kernel)) + 1e-8
        response = np.abs(cv2.filter2D(source, cv2.CV_32F, kernel))
        responses[angle] = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return responses


def threshold_responses(responses, threshold_mode="otsu", fixed_threshold=35):
    """Threshold each orientation and combine their binary hair candidates."""
    masks = {}
    for angle, response in responses.items():
        if threshold_mode == "otsu":
            _, mask = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(response, fixed_threshold, 255, cv2.THRESH_BINARY)
        masks[angle] = mask
    combined = np.maximum.reduce(list(masks.values()))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return masks, combined


def metrics(truth, prediction):
    truth, prediction = truth > 0, prediction > 0
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    tn = int(np.logical_and(~truth, ~prediction).sum())
    return {
        "precision": tp / (tp + fp + 1e-9),
        "recall": tp / (tp + fn + 1e-9),
        "specificity": tn / (tn + fp + 1e-9),
        "dice_f1": 2 * tp / (2 * tp + fp + fn + 1e-9),
        "iou": tp / (tp + fp + fn + 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Dataset folder containing dermoscopic_image/ and hair_mask/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variance-threshold", type=float, default=0.01,
                        help="Tv for global variance of median-filtered V channel, scaled to [0,1]")
    parser.add_argument("--median-kernel", type=int, default=3, help="Odd median-filter size")
    parser.add_argument("--mean-kernel", type=int, default=3, help="Mean-filter size used when variance > Tv")
    parser.add_argument("--threshold", choices=("otsu", "fixed"), default="otsu")
    parser.add_argument("--fixed-threshold", type=int, default=35,
                        help="Threshold 0..255, used only with --threshold fixed")
    args = parser.parse_args()
    if args.median_kernel < 3 or args.median_kernel % 2 == 0:
        parser.error("--median-kernel must be odd and at least 3")
    if args.mean_kernel < 1 or args.fixed_threshold < 0 or args.fixed_threshold > 255:
        parser.error("--mean-kernel must be positive and --fixed-threshold must be 0..255")

    image_dir, mask_dir = args.data / "dermoscopic_image", args.data / "hair_mask"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        parser.error("Expected --data/dermoscopic_image and --data/hair_mask")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for path in sorted(image_dir.iterdir()):
        mask_path = mask_dir / path.name
        if path.suffix.lower() not in valid_ext or not mask_path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or truth is None:
            print(f"Skipping unreadable pair: {path.name}")
            continue
        if image.shape[:2] != truth.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {path.name}")

        prepared, variance, smoothed = prepare_image(
            image, args.variance_threshold, args.median_kernel, args.mean_kernel
        )
        responses = gabor_responses(prepared)
        masks, combined = threshold_responses(responses, args.threshold, args.fixed_threshold)
        binary = (combined > 0).astype(np.uint8)
        rows.append({"image": path.stem, "variance_V": variance, "mean_filter_applied": smoothed,
                     **metrics(truth > 127, binary)})

        cv2.imwrite(str(args.out / f"{path.stem}_prepared_V.png"), prepared)
        for angle in ANGLES:
            cv2.imwrite(str(args.out / f"{path.stem}_gabor_{angle}.png"), responses[angle])
            cv2.imwrite(str(args.out / f"{path.stem}_mask_{angle}.png"), masks[angle])
        cv2.imwrite(str(args.out / f"{path.stem}_mask_combined.png"), binary * 255)
        restored = cv2.inpaint(image, binary * 255, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(str(args.out / f"{path.stem}_removed.png"), restored)

    if not rows:
        raise FileNotFoundError("No matching readable image/mask pairs found.")
    fields = list(rows[0])
    with (args.out / "metrics_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = ("precision", "recall", "specificity", "dice_f1", "iou")
    with (args.out / "metrics_mean.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n_images"] + ["mean_" + n for n in metric_names])
        writer.writeheader()
        writer.writerow({"method": "HSV-V + conditional mean + Gabor", "n_images": len(rows),
                         **{"mean_" + n: float(np.mean([r[n] for r in rows])) for n in metric_names}})
    print(f"Processed {len(rows)} pairs. Outputs and evaluation saved in {args.out}")


if __name__ == "__main__":
    main()
