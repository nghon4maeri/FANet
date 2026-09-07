# FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation

Reproduction + analysis of:

> Tomar, Jha, Riegler, Johansen, Johansen, Rittscher, Halvorsen, Ali.
> *FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation*,
> IEEE TNNLS 2022. [arXiv:2103.17235](https://arxiv.org/abs/2103.17235)

## Repository structure

```
FANet/
├── configs/                 # experiment configs (yaml)
│   └── kvasir_sessile.yaml
├── src/fanet/               # installable package
│   ├── models/              #   blocks (SELayer, ResidualBlock, MixPool) + FANet
│   ├── data/                #   DATASET, load_data, RLE feedback decoding
│   ├── losses.py            #   DiceLoss, DiceBCELoss
│   ├── metrics.py           #   Dice/IoU/recall/... evaluation metrics
│   ├── config.py            #   yaml config loader
│   └── utils.py             #   seeding, RLE, Otsu init-mask, logging
├── scripts/
│   ├── train.py             # FANet training with cross-epoch feedback masks
│   ├── evaluate.py          # test-time iterative refinement + metrics
│   └── report.py            # params/dataset report + refinement demo
├── analysis/
│   └── grad_flow.py         # gradient-flow check for MixPool's fmask branch
├── notebooks/
│   └── fanet_kaggle.py      # self-contained Kaggle notebook version
├── docs/                    # papers + FANet_Complete_Guide.md + reports/
├── papers/                  # research PDFs (original/ + translated/)
├── tools/
│   ├── pdf_translate.py     # PDF→Vietnamese translation wrapper (VI-Translate)
│   ├── vitranslate/         # VI-Translate submodule (runtime engine)
│   └── tests/               # integration smoke tests
├── data/                    # datasets (git-ignored, download separately)
├── assets/                  # architecture / qualitative figures
├── checkpoints/             # model weights (git-ignored)
├── logs/                    # training & analysis logs
└── results/                 # evaluation outputs (git-ignored)
```

## Setup

```bash
pip install -e .            # installs the fanet package (src/fanet)
```

Requirements: torch, numpy, opencv-python, albumentations, scikit-learn, tqdm, pyyaml.

## Data

Place datasets under `data/`:

```
data/
├── Kvasir-SEG/                # full Kvasir-SEG (1000 images)
└── sessile-main-Kvasir-SEG/   # sessile subset (196 images) with train.txt/val.txt
```

Update `dataset.root` in `configs/kvasir_sessile.yaml` if needed.

## Usage

Train (feedback masks from previous epochs, checkpoint saved on val-loss improvement):

```bash
python scripts/train.py --config configs/kvasir_sessile.yaml
```

Evaluate with iterative test-time refinement:

```bash
python scripts/evaluate.py --config configs/kvasir_sessile.yaml \
    --checkpoint checkpoints/checkpoint.pth --num-iter 10 --save-masks
```

Gradient-flow analysis (checks whether the hard binary gate in MixPool cuts
gradient to the learned fmask branch):

```bash
python analysis/grad_flow.py --config configs/kvasir_sessile.yaml
```

## PDF translation (VI-Translate)

Translate research PDFs into Vietnamese while preserving layout, formulas and
figures. Inputs live under `papers/original/`, output under `papers/translated/`.

```bash
git submodule update --init --recursive     # one-time: fetch VI-Translate
python tools/pdf_translate.py --setup        # one-time: create isolated runtime
python tools/pdf_translate.py papers/original/attention.pdf
# -> papers/translated/attention-vi.pdf
```

Google engine is the default (free, no API key, needs network). See
[`docs/pdf_translate_guide.md`](docs/pdf_translate_guide.md) for options,
engines, OCR, troubleshooting, and the OpenCode skill.

## Key findings so far

- MixPool's learned mask branch `fmask` receives **zero gradient**: the
  `(fmask > 0.5)` binarization is non-differentiable, so its conv weights
  never update (verified in `logs/analysis_grad_log.txt`). Only its
  BatchNorm running stats drift via the forward pass.
- The feedback mask itself is a hard binary mask (prediction thresholded at
  0.5 + RLE round-trip), discarding confidence/uncertainty information of
  the previous prediction.

## Citation

```bibtex
@article{tomar2022fanet,
  title={Fanet: A feedback attention network for improved biomedical image segmentation},
  author={Tomar, Nikhil Kumar and Jha, Debesh and Riegler, Michael A and Johansen, H{\aa}vard D and Johansen, Dag and Rittscher, Jens and Halvorsen, P{\aa}l and Ali, Sharib},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```
