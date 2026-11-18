# FSNA-Net — anonymous review code archive

Anonymized code, configs, and pretrained weights for peer review of the FSNA-Net JEI manuscript.
No author identities are included in this archive.

## What is included

| Path | Role |
|------|------|
| `checkpoints/fsna_ge_lolv1_best.pth` | **Main reported model** (FSNA + learned global exposure) on LOL-v1 |
| `checkpoints/fsna_rf_lolv1_best.pth` | FSNA-on-RF Phase-2 seed used by the main model |
| `third_party/Retinexformer/pretrained_weights/LOL_v1.pth` | Official RetinexFormer LOL-v1 weights (baseline) |
| `code/` | Training / evaluation scripts for Route A (FSNA-RF) and Route B (GE head) |
| `third_party/Retinexformer/` | Minimal vendor arch file (RetinexFormer) + license |

Expected LOL-v1 numbers under the paper protocol (no GT-mean alignment):

- RetinexFormer official: ~25.16 PSNR / 0.860 SSIM
- FSNA-on-RF: ~25.23 PSNR / 0.872 SSIM
- **FSNA+GE (main): ~25.68 PSNR / 0.871 SSIM / LPIPS ~0.125**

## Environment

```bash
pip install -r requirements.txt
```

Requires a public LOL-v1 layout, for example:

```text
<data_root>/lolv1/eval15/low/*.png
<data_root>/lolv1/eval15/high/*.png
```

(or the equivalent split resolved by `code/utils/dataset.py`).

## Evaluate the main checkpoint (FSNA+GE)

```bash
cd code
python eval_main.py --data_root <path-to-datasets> --ckpt ../checkpoints/fsna_ge_lolv1_best.pth --fsna_init ../checkpoints/fsna_rf_lolv1_best.pth
```

Evaluate FSNA-RF only:

```bash
python eval_fsna_rf.py --data_root <path-to-datasets> --dataset lolv1 --model fsna_rf --weights ../checkpoints/fsna_rf_lolv1_best.pth
```

Evaluate official RetinexFormer:

```bash
python eval_retinexformer_pretrained.py --data_root <path-to-datasets>
```

## Train (optional)

FSNA-on-RF (Route A Phase-2 style):

```bash
python train_route_a.py --model fsna_rf --pretrained ../third_party/Retinexformer/pretrained_weights/LOL_v1.pth --official_augs
```

FSNA+GE probe (Route B):

```bash
python train_route_b_gen.py --base fsna --fsna_init ../checkpoints/fsna_rf_lolv1_best.pth --tag review_fsna_ge
```

## Protocol notes

- Metrics are computed without GT-mean / GT-alignment post-processing.
- Datasets are public benchmarks only (LOL-v1 / LOL-v2 / no-reference sets as cited in the manuscript).
- RetinexFormer architecture code is redistributed under its original license (`third_party/Retinexformer/LICENSE.txt`).

## Contact during review

Editors may request clarification via the journal portal. A public non-anonymous release is planned upon acceptance.
