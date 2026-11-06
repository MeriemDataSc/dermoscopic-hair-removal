# Dermoscopic Hair Artifact Preprocessing

This project explores classical image-processing methods for detecting and removing hair artifacts from dermoscopic images. Hair can obscure lesion structures and interfere with later image-analysis steps. The project focuses on preprocessing: locating hair pixels, evaluating the predicted hair masks against annotations, and restoring the detected regions by inpainting.

## Project goals

- Prepare paired dermoscopic images and hair masks through synchronized data augmentation.
- Implement a DullRazor-style baseline for dark-hair detection.
- Implement an oriented Gabor-filter approach, including the HSV-V filter bank in [`my_gabor.py`](my_gabor.py).
- Evaluate predicted hair masks with pixel-level segmentation metrics.

## Data

The supplied dataset contains RGB dermoscopic images and corresponding fine-grained binary hair masks. The images and masks are paired by filename. In the masks, hair pixels are white and the background is black.

Expected input structure:

```text
dataset/
├── dermoscopic_image/
│   ├── ISIC_0000004.png
│   └── ...
└── hair_mask/
    ├── ISIC_0000004.png
    └── ...
```

The dataset itself is not included in this repository. Check the dataset's original terms before redistributing it.

## Methods

### Data augmentation

[`augmentation.py`](augmentation.py) creates additional image-mask pairs using horizontal and vertical flips, rotation, scaling, translation, and brightness/contrast changes. Geometric transformations are applied identically to each image and its mask; photometric changes are applied only to the image. Original files are left unchanged.

Example:

```bash
python augmentation.py --data "path/to/dataset" --out "path/to/augmented_dataset" --n 10
```

### DullRazor-style baseline

[`dullrazor.py`](dullrazor.py) converts each input image to grayscale and applies morphological closing with line-shaped structuring elements at multiple orientations. The difference between the closed image and the original grayscale image highlights dark, line-like structures. Otsu thresholding and morphological cleanup produce the predicted hair mask. OpenCV inpainting then fills the detected regions.

This is a DullRazor-style implementation of the published morphological detection principle; it is not a bit-for-bit reproduction of the original software.

```bash
python dullrazor.py --data "path/to/dataset" --out "predicted_masks_dullrazor"
```

### Gabor filter bank

[`my_gabor.py`](my_gabor.py) follows the project's proposed flowchart:

1. Convert the RGB image to HSV and extract the V channel.
2. Apply a median filter.
3. Measure the global variance of the median-filtered V channel. Apply mean smoothing only when the variance is greater than `Tv`.
4. Apply five anisotropic Gabor kernels at 0°, 45°, 90°, 135°, and 157°.
5. Threshold each orientation response and combine the resulting binary masks.
6. Evaluate the combined mask and inpaint detected regions.

The five filter parameter sets, in orientation order, are `(σx=2, σy=6, u=0.03)`, `(2, 4, 0.06)`, `(2, 6, 0.065)`, `(2, 6, 0.07)`, and `(2, 6, 0.07)`. The code interprets `σx` as the filter width across the hair, `σy` as its extent along the hair, and `u` as spatial frequency in cycles per pixel. The diagram does not define these conventions; verify them against the mathematical definition used in the source material.

Otsu thresholding is the default. A fixed threshold can be selected with `--threshold fixed --fixed-threshold 35`. The default variance threshold is `Tv=0.01` on the V channel normalized to `[0, 1]`.

```bash
python my_gabor.py --data "path/to/dataset" --out "predicted_masks_mygabor"
```

The output folder contains per-orientation responses and masks, the combined predicted mask, the inpainted image, and CSV evaluation files. The Otsu/fixed-threshold choice and `Tv` should be selected using training or validation data and kept fixed for final evaluation.

## Evaluation

Each predicted binary mask is compared pixel by pixel with its ground-truth hair mask. The scripts report per-image values and their arithmetic mean for:

| Metric | Meaning |
|---|---|
| Precision | Fraction of predicted hair pixels that are annotated as hair |
| Recall | Fraction of annotated hair pixels detected by the method |
| Specificity | Fraction of background pixels correctly left unmarked |
| Dice / F1 | Overlap score between predicted and annotated hair pixels |
| IoU | Intersection over union of the predicted and annotated hair regions |

The evaluation scripts save `metrics_per_image.csv` and `metrics_mean.csv` in the selected output folder. Higher precision, recall, Dice/F1, and IoU are better; higher specificity means fewer background pixels are marked as hair.

## Available results

The following values are copied from the available `metrics_mean.csv` files. Scores are means across the 5,000 samples listed in those files.

| Method | Samples | Precision | Recall | Specificity | Dice / F1 | IoU |
|---|---:|---:|---:|---:|---:|---:|
| DullRazor-style | 5,000 | 0.4056 | 0.3957 | 0.9213 | 0.3391 | 0.2284 |
| Gabor baseline (`gabor.py`) | 5,000 | 0.3722 | 0.1155 | 0.9783 | 0.1481 | 0.0834 |
| HSV-V Gabor (`my_gabor.py`) | Pending | — | — | — | — | — |

The evaluated filenames include augmented variants (for example, `_aug001`). Therefore, the 5,000 samples should not be interpreted as 5,000 independent original images. Variants derived from the same original image are correlated. For a stronger estimate of generalization, split the original images before augmentation, keep all variants of each original in the same split, tune parameters only on training/validation data, and report the held-out test scores. A `metrics_mean.csv` for `my_gabor.py` was not present in the supplied results folder, so its result is left pending.

## Requirements

```bash
python -m pip install numpy opencv-python
```

## References

- Lee et al., “DullRazor: A software approach to hair removal from images,” *Computers in Biology and Medicine*, 1997. [https://doi.org/10.1016/S0010-4825(97)00020-6](https://doi.org/10.1016/S0010-4825(97)00020-6)
- Nasonova et al., “Image warping in dermatological image hair removal,” ICIAR 2014. [Article PDF](https://imaging.cs.msu.ru/pub/2014.ICIAR.Nasonova_Krylov.Warping.en.pdf). The paper describes Gabor-based hair detection and PDE reconstruction; `my_gabor.py` uses OpenCV inpainting and does not reproduce the paper's complete reconstruction and warping pipeline.
