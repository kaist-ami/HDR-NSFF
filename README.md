# HDR-NSFF: High Dynamic Range Neural Scene Flow Fields (ICLR 2026)

### [Project Page](https://shin-dong-yeon.github.io/HDR-NSFF/) | [Paper](https://arxiv.org/abs/2603.08313) | [Data](https://huggingface.co/datasets/SHlNDY/HDR-NSFF)

Official implementation of **HDR-NSFF**, accepted at **ICLR 2026**.

---

## 📢 News
- **[2026.04]** Code released!
- **[2026.04]** HDR-GoPro dataset released on HuggingFace! 🤗

---

## Abstract

We propose **HDR-NSFF**, a method to learn robust **4D HDR Radiance Fields** from monocular videos with alternating exposures. Unlike conventional HDR video approaches that align frames at the pixel level—causing ghosting and temporal inconsistencies—our method models the scene as a continuous function across space and time, explicitly representing HDR radiance, 3D scene flow, geometry, and tone-mapping in a unified framework.

**Key components:**
- **Tone-Mapping Module** — Learnable piecewise camera response function (CRF) with per-channel white balance, converting HDR radiance to LDR observations.
- **Semantic Optical Flow** — DINOv2-based motion estimation robust to exposure variation, where DINO features remain invariant to photometric changes.
- **Generative Prior (Difix)** — Compensates for monocular capture limitations and saturation-induced information loss by generating enhanced novel views as pseudo-labels during training.

---

## Dataset

We introduce the **HDR-GoPro Dataset**, the first real-world HDR benchmark for dynamic scene synthesis, captured with **9 synchronized cameras** across **12 diverse scenes** at multiple exposure levels.

🤗 **Dataset available on HuggingFace:** [SHlNDY/HDR-NSFF](https://huggingface.co/datasets/SHlNDY/HDR-NSFF)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SHlNDY/HDR-NSFF",
    repo_type="dataset",
    allow_patterns="<scene_name>/*",   # e.g. "tumbler/*"
    local_dir="./data/hdr-gopro",
)
```
Place your dataset under `./hdr-nsff/data/`
Expected structure per scene:

```
data/
  hdr-gopro
    <scene_name>/
      dense/
  hdr-dslr
    <scene_name>/
      dense/

```


---

## Environment Setup

Two separate conda environments are required:

```bash
# Main training environment (PyTorch 2.5.1, includes SAM2)
bash nsff_scripts/setup_hdr_nsff.sh

# DINO-tracker semantic flow environment for data preprocessing (PyTorch 2.1.0)
bash nsff_scripts/setup_dino_tracker.sh
```

---


## Training

```bash
conda activate hdr-nsff
cd nsff_exp/
python run_nerf.py --config configs/<config_file.txt>
```

<!-- Key config options:

| Option | Description |
|--------|-------------|
| `disp_model` | `depth-anything` / `midas` / `aligned` |
| `render_tm` | `tm` (tone-mapping) / `wb` (white balance) / `no` |
| `use_difix` | Enable generative prior enhancement |
| `chain_sf` | 5-frame scene flow chain loss |
| `use_percept` | Perceptual (LPIPS) loss | -->

---

## Rendering & Evaluation

```bash
cd nsff_exp/

# Evaluation metrics (PSNR / SSIM / LPIPS)
python evaluation_gopro.py --config configs/<config_file.txt>
python evaluation_dslr.py  --config configs/<config_file.txt>

# Temporal interpolation evaluation
python evaluation_gopro_time.py    --config configs/<config_file.txt>
python evaluation_gopro_time_36.py --config configs/<config_file.txt>

# Novel view rendering
python run_nerf.py --config <config> --render_lockcam --target_idx 2
python run_nerf.py --config <config> --render_bt --target_idx 20
python run_nerf.py --config <config> --render_train
```

---

## Preprocessing Pipeline (Optional)

All preprocessing runs on your dataset organized as:

```
<data_path>/
  images/       ← raw video frames (.jpg or .png)
```

Run the steps in order:

**1. Motion Masks (SAM2) — Interactive**

Open `nsff_scripts/sam2_motion_mask.ipynb` in JupyterLab (`hdr-nsff` env).  
Left-click foreground, right-click background on the first frame, then propagate.

Outputs: `masks/` (for COLMAP), `motion_masks/` (foreground=255).

**2. COLMAP — Camera Poses & Undistortion**

```bash
# Edit data_path inside the script first
bash nsff_scripts/run_colmap_0.sh
```

Outputs: `sparse/0/`, `dense/images/`, `dense/motion_masks/`.

**3. Camera Pose Conversion**

```bash
conda activate hdr-nsff
python nsff_scripts/save_poses_nerf.py --data_path <dense_path>
```

**4. Resize Images**

```bash
python nsff_scripts/resize_images.py --data_path <dense_path> --resize_height <H>
# Output: dense/images_{W}x{H}/
```

**5. Metric Depth Estimation**

```bash
python nsff_exp/Depth-Anything-V2/metric_depth/run.py \
    --encoder vitl \
    --load-from nsff_exp/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
    --max-depth 20 --data-dir <dense_path> --save-numpy
# Output: dense/depth-anything/
```

**6. Semantic Optical Flow (DINO-tracker)**

```bash
# Edit data_path inside the script first
bash nsff_scripts/run_dino-tracker_0.sh
```

Outputs: `dense/dino-tracker/semantic_flow/semantic_flows_i1_{fwd,bwd}.npy`, then `dense/semantic_flow_i1/*.npz`.


---

## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{dong-yeon2026hdr-nsff,
  title     = {HDR-NSFF: High Dynamic Range Neural Scene Flow Fields},
  author    = {Dong-Yeon, Shin and Jun-Seong, Kim and Byung-Ki, Kwon and Oh, Tae-Hyun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---

## Acknowledgements

This project builds upon [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields), [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [DINO-tracker](https://github.com/AssafSinger94/dino-tracker), [SAM2](https://github.com/facebookresearch/sam2), and [Difix3D](https://github.com/nv-tlabs/Difix3D). We thank the authors for their excellent work.
