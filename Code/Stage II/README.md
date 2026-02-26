# Stage II — Discrete Beam Selection

> **CORTEX Stage II** selects optimal TX/RX beam patterns from a 255×255 codebook  
> for THz UAV communications using a causal transformer trained with knowledge distillation.

---

## Contents

```
code/Stage II/
├── CORTEX_Stage2.ipynb   ← main notebook (data generation → training → evaluation → plots)
└── README.md             ← this file
```

---

## Architecture

```
State sequence  [B, T=7, F=17]
      │
      ▼
Conv Front-End         2 × Conv1d (GELU) — downsamples time axis
      │
      ▼
Causal Transformer     5 × (RoPE self-attention + FFN + LayerNorm)
      │
   last token + 0.5 × mean pool  →  query  [B, 1, D]
      │
      ├─ Cross-attention over TX beam codebook  [255, D]  →  TX classifier  →  logits [B, 255]
      ├─ Cross-attention over RX beam codebook  [255, D]  →  RX classifier  →  logits [B, 255]
      └─ Angle Refiner MLP  →  6-D unit vector (TX + RX angles)
```

**Key design choices**

| Choice | Reason |
|---|---|
| Learnable beam codebooks | Maps each of 255 patterns into a latent beam embedding |
| Cross-attention query over codebook | Selects beams by attending over the full codebook |
| Focal loss | Handles class imbalance across 255 beam classes |
| Knowledge distillation (top-K, temperature-annealed) | Transfers rich capacity-score information from the exhaustive teacher |
| Wrapped Cauchy angle loss | Circular-aware auxiliary loss improves directional accuracy |
| Incremental checkpointing | Data generation and evaluation checkpoint every N steps — safe to interrupt |

---

## Two-Stage Pipeline

This notebook is **Stage II** of a two-stage system:

| Stage | Task | Output |
|---|---|---|
| **Stage I** (`code/Stage I/`) | Continuous angle prediction | TX/RX unit vectors |
| **Stage II** (`code/Stage II/`) | Discrete beam selection | TX/RX beam indices from a 255-pattern codebook |

The two stages are independent — Stage II can be run without Stage I.

---

## Input Data

### Antenna Radiation Patterns

A folder of CSV files (`rows_*.csv`), one per beam pattern. Each CSV must contain:

| Column | Description |
|---|---|
| `phi` | Azimuth angle (°) |
| `theta` | Elevation angle (°) |
| `gain` | Realized gain (dBi) |

255 pattern files are expected (configurable via `NUM_TX_PATTERNS` / `NUM_RX_PATTERNS`).

### Mobility Trace

A single CSV file with per-timestep UAV kinematics:

| Column | Description |
|---|---|
| `time` | Timestep (s) |
| `x_tx`, `y_tx`, `z_tx` | TX position (m) |
| `x_rx`, `y_rx`, `z_rx` | RX position (m) |
| `vx_tx`, `vy_tx`, `vz_tx` | TX velocity (m/s) |
| `vx_rx`, `vy_rx`, `vz_rx` | RX velocity (m/s) |

---

## Running the Notebook

### On Google Colab (recommended — data generation is slow without GPU)

1. Open `CORTEX_Stage2.ipynb` in Colab.
2. In **§ 3 · Load Radiation Patterns**, set `PATTERNS_DIR` to your Drive path, or let auto-detect run.
3. Set `MOBILITY_FILE` to your mobility trace path.
4. Run all cells.

All outputs are saved to `CORTEX_experiments_255/` in My Drive.

### Locally

```bash
pip install torch numpy matplotlib pandas scipy scikit-learn einops

# Edit PATTERNS_DIR and MOBILITY_FILE in the notebook, then:
jupyter notebook CORTEX_Stage2.ipynb
```

> ⚠️ **Data generation warning**: The exhaustive search (255×255 per timestep) is slow.  
> For a 200-step trace it takes ~10–30 minutes on GPU.  
> The notebook checkpoints every 50 samples — safe to interrupt and resume.

---

## Hyperparameters

All hyperparameters live in a single `Config` class (**§ 2 · Configuration**):

| Parameter | Default | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | 7 | Input window (time steps) |
| `D_MODEL` | 256 | Transformer hidden dim |
| `N_HEADS` | 8 | Attention heads |
| `N_LAYERS` | 5 | Transformer depth |
| `DIM_FEEDFORWARD` | 512 | FFN inner dim |
| `DROPOUT` | 0.15 | Dropout rate |
| `BEAM_EMBED_DIM` | 128 | Beam codebook embedding dim |
| `LEARNING_RATE` | 2e-4 | AdamW learning rate |
| `BATCH_SIZE` | 16 | Mini-batch size |
| `NUM_EPOCHS` | 250 | Maximum training epochs |
| `DISTILL_TEMPERATURE_MAX` | 8.0 | KD temperature at epoch 0 |
| `DISTILL_TEMPERATURE_MIN` | 3.0 | KD temperature at final epoch |
| `DISTILL_ALPHA_MIN` | 0.2 | Hard-label weight at epoch 0 |
| `DISTILL_ALPHA_MAX` | 0.6 | Hard-label weight at final epoch |
| `TOPK_TX / TOPK_RX` | 6 | Top-K beams for sparse KD targets |
| `FOCAL_GAMMA` | 2.0 | Focal loss focusing parameter |
| `ANGLE_LOSS_WEIGHT` | 0.20 | Weight of the angle-refinement auxiliary loss |

---

## Outputs

| File | Description |
|---|---|
| `checkpoints/ckpt_*_best.pth` | Best model checkpoint (highest joint accuracy) |
| `checkpoints/ckpt_*_latest.pth` | Most recent checkpoint |
| `data_cache/data_*.npz` | Cached training data (skip re-running exhaustive search) |
| `logs/log_*.csv` | Per-epoch: loss, accuracy, LR |
| `results/results_*.csv` | Per-timestep: CORTEX capacity vs. optimal capacity |
| `paper_plots/fig_*.png/.pdf` | IEEE-formatted figures |

### Generated Plots

| Figure | Description |
|---|---|
| `fig_training_dynamics` | Loss curves + joint beam accuracy over epochs |
| `fig_cdf_spectral_efficiency` | CDF of spectral efficiency: CORTEX vs. exhaustive search |
| `fig_rate_tracking` | Capacity time-series: CORTEX vs. optimal |

---

## Resuming Interrupted Runs

The notebook handles interruptions at two levels:

**Data generation** — A partial `.npz` file is saved every 50 samples. On restart, generation resumes from the last saved index automatically.

**Training** — A checkpoint is saved every 10 epochs and whenever a new best validation accuracy is achieved. On restart, the latest checkpoint is detected and training resumes from the correct epoch.

**Evaluation** — An incremental CSV is saved every 100 timesteps. If the file exists when evaluation is re-run, it resumes from where it left off.

---

## Citation

```bibtex
@article{usta2026cortex,
  author  = {Usta, Mahir Burak and Bafarassat, Milad and Erdem, Mikail and
             Gurbuz, Ozgur and Saeed, Akhtar and Tokgoz, Korkut Kaan and Qaraqe, Khalid},
  title   = {Transformer-Driven Beam Control via Reconfigurable Antenna Arrays
             for Terahertz {UAV} Communications},
  journal = {IEEE Open Journal of the Communications Society},
  year    = {2026},
}
```
