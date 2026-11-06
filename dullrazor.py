"""DullRazor-style hair detection, inpainting and mask evaluation."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def detect_hairs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    responses = []
    # Grayscale closing estimates the skin under dark hairs; use multiple angles.
    for angle in range(0, 180, 15):
        kernel = np.zeros((17, 17), np.uint8)
        rad = np.deg2rad(angle)
        p1 = (int(8 - 7 * np.cos(rad)), int(8 - 7 * np.sin(rad)))
        p2 = (int(8 + 7 * np.cos(rad)), int(8 + 7 * np.sin(rad)))
        cv2.line(kernel, p1, p2, 1, 1)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        responses.append(cv2.subtract(closed, gray))
    response = np.maximum.reduce(responses)
    _, mask = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def score(truth, pred):
    t, p = truth > 0, pred > 0
    tp = np.logical_and(t, p).sum(); fp = np.logical_and(~t, p).sum()
    fn = np.logical_and(t, ~p).sum(); tn = np.logical_and(~t, ~p).sum()
    return {"precision": tp/(tp+fp+1e-9), "recall": tp/(tp+fn+1e-9),
            "specificity": tn/(tn+fp+1e-9), "dice_f1": 2*tp/(2*tp+fp+fn+1e-9),
            "iou": tp/(tp+fp+fn+1e-9)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); im_dir=args.data/"dermoscopic_image"; gt_dir=args.data/"hair_mask"
    if not im_dir.is_dir() or not gt_dir.is_dir(): ap.error("Expected dermoscopic_image/ and hair_mask/ folders")
    args.out.mkdir(parents=True, exist_ok=True); rows=[]
    for path in sorted(im_dir.iterdir()):
        gt_path=gt_dir/path.name
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} or not gt_path.exists(): continue
        image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        gt=cv2.imread(str(gt_path),cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None: print(f"Skipping unreadable pair: {path.name}"); continue
        if image.shape[:2]!=gt.shape[:2]: raise ValueError(f"Image/mask size mismatch: {path.name}")
        pred=detect_hairs(image)
        binary=(pred>0).astype(np.uint8)
        rows.append({"image":path.stem,**score(gt>127,binary)})
        cv2.imwrite(str(args.out/f"{path.stem}_mask.png"),binary*255)
        cv2.imwrite(str(args.out/f"{path.stem}_removed.png"),cv2.inpaint(image,binary*255,3,cv2.INPAINT_TELEA))
    if not rows: raise FileNotFoundError("No matching image/mask pairs found.")
    fields=list(rows[0]);
    with (args.out/"metrics_per_image.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    keys=fields[1:]
    with (args.out/"metrics_mean.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["method","n_images"]+["mean_"+k for k in keys]); w.writeheader()
        w.writerow({"method":"DullRazor-style","n_images":len(rows),**{"mean_"+k:float(np.mean([r[k] for r in rows])) for k in keys}})
    print(f"Saved masks, restored images and evaluation to {args.out}")


if __name__ == "__main__": main()
