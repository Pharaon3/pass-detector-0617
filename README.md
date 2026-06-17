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

```bash
python train.py --require-gpu
```

On **CPU** (when CUDA unavailable and `--require-gpu` not set), training auto-switches to low-memory CPU settings.

If you still hit OOM on GPU, lower `batch_size` to 2 or `backbone_chunk_size` to 16 in `config.yaml`.

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
