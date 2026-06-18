# Soccer Pass Event Detection

End-to-end PyTorch pipeline for **frame-level pass probability** estimation on 30s soccer clips (25 FPS, 750 frames).

## Project Structure

```
soccer-pass-detection/
├── data/
│   ├── videos/          # .mp4 clips (30s @ 25 FPS)
│   └── annotations/     # matching JSON per video
├── dataset/
│   ├── sliding_window.py
│   └── dataset.py
├── models/
│   ├── backbone.py      # timm RegNetY
│   ├── gsm.py           # Gate Shift Module
│   ├── transformer.py   # 6-layer temporal encoder
│   ├── frame_head.py    # per-frame sigmoid head
│   └── model.py
├── train.py
├── infer.py
├── evaluate.py
├── merge_windows.py     # stitch overlapping windows
├── nms.py               # temporal NMS + event extraction
├── utils.py
└── config.yaml
```

## Setup

```bash
pip install -r requirements.txt
```

Place training clips under `data/` using this layout:

```
data/
  clip_4/
    224p.mp4
    label.json
  clip_10/
    224p.mp4
    label.json
  ...
```

Only `observation` PASS events in `label.json` are used as labels (anticipation events are outside the clip).

## GPU Setup (required before training)

Your earlier error means **PyTorch was installed without a working CUDA build** (or the CUDA wheel doesn't match your driver). Fix it on Ubuntu:

```bash
source venv/bin/activate

# 1) Diagnose
python check_gpu.py

# 2) Reinstall PyTorch with CUDA (pick one)
bash setup_gpu.sh          # default: CUDA 12.1 wheel
bash setup_gpu.sh cu118    # older drivers / GPUs

# 3) Confirm GPU works
python check_gpu.py

# 4) Train on GPU
python train.py --require-gpu
```

If `nvidia-smi` fails, install/update the NVIDIA driver first, then reboot:
```bash
sudo apt install nvidia-driver-535
sudo reboot
```

### GPU config (`config.yaml`)

| Setting | Default (GPU) |
|---------|----------------|
| backbone | `regnety_032` (1296-d) |
| batch_size | 4 |
| grad_accum_steps | 2 (effective batch 8) |
| amp | true |
| backbone_chunk_size | 32 |

Increase `batch_size` to 8 on 24GB cards (3090/4090). Use `regnety_064` if you have headroom.

---

## Training

Train with **on-the-fly augmentations** (grayscale, ±5° rotate, hue shift, 1.1× zoom):

```bash
python train.py --require-gpu
```

Augmentation grid (in `config.yaml`): **3 × 2 × (3 + 1) = 24** variants per window  
(color gets hue shifts; grayscale skips hue) → train set ≈ **24× larger**.

**Training runs for 30 epochs** by default (`training.num_epochs` in `config.yaml`).

Disable augmentations (baseline):

```bash
python train.py --require-gpu --no-augment
```

Customize grid in `config.yaml`:

```yaml
augmentation:
  enabled: true
  rotation_deg: [0, -5, 5]    # 3
  zoom: [1.0, 1.1]            # 2
  hue_deg: [0, 15, -15]       # 3 color + 1 gray each → 24× total
```

Then re-infer / evaluate:

```bash
bash bulk_grayscale_infer_plot.sh checkpoints/best.pt
bash bulk_rotated_infer_plot.sh checkpoints/best.pt
python evaluate.py --checkpoint checkpoints/best.pt
```

## Inference

```bash
python infer.py --checkpoint checkpoints/best.pt
```

**Primary output** per video (`outputs/{video_id}_frame_probs.json`):

```json
{
  "frame_probs": [0.01, 0.02, 0.15, ..., 0.88]
}
```

Length = **750** (full 30s clip). Overlapping 7s windows are averaged via `merge_windows.py`.

Event JSON also written to `outputs/{video_id}_events.json`.

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

Reports frame-level AUC/F1 and event-level precision/recall + temporal error.

## Visualize pass probability curve

Plot predicted probability vs time with ground-truth pass frames marked:

```bash
python plot_probs.py --clip clip_4 --checkpoint checkpoints/best.pt
```

Saved to `outputs/plots/clip_4_pass_probs.png`.

### Bulk infer + plot all clips in `data/`

**One command:**

```bash
bash bulk_infer_plot.sh checkpoints/best.pt
```

Skips clips already inferred/plotted (`--skip-existing`). Only **new** clips in `data/` are processed.

Re-run everything from scratch:

```bash
python infer.py --checkpoint checkpoints/best.pt
python plot_probs.py --all
```

**Or step by step:**

```bash
# 1) Infer all clips (default: every data/clip_XXX/)
python infer.py --checkpoint checkpoints/best.pt

# 2) Plot all clips from saved JSONs
python plot_probs.py --all
```

Outputs:
- `outputs/clip_XXX_frame_probs.json` — 750 frame probabilities per clip
- `outputs/plots/clip_XXX_pass_probs.png` — probability curve with GT pass frames

Plot only (skip inference if JSONs already exist):

```bash
python plot_probs.py --all --probs-dir outputs
```

Use an existing inference JSON for one clip:

```bash
python plot_probs.py --clip clip_4 --probs-json outputs/clip_4_frame_probs.json
```

Show interactively (requires display):

```bash
python plot_probs.py --clip clip_4 --show
```

## Create a custom test video

Build a 30s clip from clip_4 starting at 1s, with 1s black at the end (keeps original 398×224 and SAR):

```bash
bash make_test_video.sh

python infer.py --checkpoint checkpoints/best.pt \
  --video data/test_videos/clip_4_from1s_black1s.mp4
```

## Find duplicate / overlapping segments

Search which clips share the same footage as part of another clip:

```bash
python find_matching_segment.py --query-clip clip_235 --start-sec 20 --end-sec 30
```

```bash
python find_matching_segment.py --query-clip clip_235 --start-sec 20 --end-sec 30 --min-score 0.85 --exclude-self
```

## Grayscale (no color) inference experiment

Build grayscale copies, infer all, and plot (separate from color outputs):

```bash
bash bulk_grayscale_infer_plot.sh checkpoints/best.pt
```

Or step by step:

```bash
bash make_grayscale_clips.sh data data_grayscale

python infer.py --checkpoint checkpoints/best.pt \
  --video-dir data_grayscale --output-dir outputs_grayscale

python plot_probs.py --all --data-root data_grayscale \
  --probs-dir outputs_grayscale --output-dir outputs_grayscale/plots
```

## Rotated ±5° inference experiment

Rotate left/right, zoom to remove black corners, keep original resolution, infer and plot:

```bash
FORCE=1 bash bulk_rotated_infer_plot.sh checkpoints/best.pt
```

`FORCE=1` rebuilds rotated videos (needed after fixing the rotation filter).

Or step by step:

```bash
bash make_rotated_clips.sh data data_rotated_left data_rotated_right 5

python infer.py --checkpoint checkpoints/best.pt \
  --video-dir data_rotated_left --output-dir outputs_rotated_left --skip-existing
python plot_probs.py --all --data-root data_rotated_left \
  --probs-dir outputs_rotated_left --output-dir outputs_rotated_left/plots

python infer.py --checkpoint checkpoints/best.pt \
  --video-dir data_rotated_right --output-dir outputs_rotated_right --skip-existing
python plot_probs.py --all --data-root data_rotated_right \
  --probs-dir outputs_rotated_right --output-dir outputs_rotated_right/plots
```

## Model Architecture

1. **Backbone**: `timm` RegNetY (`regnety_064` → 2016-d features per frame)
2. **GSM**: channel-wise temporal shift for motion
3. **Transformer**: 6 layers, d_model=512, 8 heads
4. **Frame head**: Linear → sigmoid → `(B, T, 1)` pass probability per frame

## Annotation Format

```json
{
  "observation": [
    {"label": "PASS", "position": "22280", "team": "home", "visibility": "visible"}
  ],
  "anticipation": []
}
```

`position` is milliseconds within the clip → frame index = `position / 1000 * 25`.

## Config

Edit `config.yaml` for batch size, backbone (`regnety_008` for lighter / `regnety_064` for 2016-dim), thresholds, and NMS window.
