# FineGrainedAlignment

Training and testing code for **Fine Grained Alignment**, a fine-grained vision-language model for ultrasound, MRI and CT images.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10. The pretrained FG-CLIP, DINOv2 and SBERT models are downloaded automatically on first run.

## Data

The datasets are not included. Put them under `data/`:

```text
data/
├── ultrasound_data/          # US training data and test folders
├── mri_data/                 # MRI training data
├── mri_data_retrieval_test/
├── mri_seg_data_test/
├── ct_data/                  # CT training data
├── ct_data_retrieval_test/
└── ct_seg_data_sel/
```

## Training

Set `modality`, `data_path`, `base_image_size` and `checkpoint_dir` in `configs/train_ultraclip.conf`, then run:

```bash
python train_ultraclip.py --conf configs/train_ultraclip.conf
```

| Modality | `modality` | `data_path` | `base_image_size` |
|---|---|---|---|
| Ultrasound | `us` | `data/ultrasound_data` | 512 |
| MRI | `mri` | `data/mri_data` | 336 |
| CT | `ct` | `data/ct_data` | 336 |

Training logs to Weights & Biases. To run without an account, set `WANDB_MODE=disabled`.

## Testing

Set `model_path` and `base_image_size` in `configs/test_ultraclip.conf`, then run:

```bash
python test_ultraclip.py --modality us --conf configs/test_ultraclip.conf --data_root data
```

Use `--modality mri` or `--modality ct` for the other modalities. Results are saved to `results/`.
