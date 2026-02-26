# Stage I — Beam Angle Prediction

> **CORTEX Stage I** predicts optimal TX/RX beam angles for THz UAV communications  
> from mobility time-series features using a causal transformer with circular regression.

---

## Contents

```
code/stage I/
├── CORTEX_Stage1.ipynb   ← main notebook (train + evaluate + save)
└── README.md             ← this file
```

---

## Architecture

```
Input features [B, T, F]
      │
      ▼
Conv Front-End          2 × strided Conv1d (kernel 5→3, stride 2) + GELU + BN
      │
      ▼
Causal Transformer      N × (RoPE multi-head self-attention + FFN + LayerNorm)
      │
 last token + mean pool
      │
      ▼
MLP Head                Linear → LayerNorm → GELU → Linear → LayerNorm → GELU → Linear(6)
      │
      ▼
L2-normalise            TX unit vector (3D)  |  RX unit vector (3D)
```

**Key design choices**

| Choice | Reason |
|---|---|
| 3-D unit-vector targets | Avoids angle-wrap discontinuity in target space |
| Rotary Positional Embeddings (RoPE) | Better length generalisation than learned embeddings |
| Causal attention mask | Enforces temporal causality — no future leakage |
| Strided Conv front-end | Reduces sequence length before attention, lowers compute |
| Wrapped Cauchy loss | Circular-aware loss; penalises angular error correctly |

---

## Input Data

The notebook expects a CSV with these columns:

| Column | Description |
|---|---|
| `time` | Timestep index within a scenario |
| `scenario` | Unique scenario ID (used for stratified splits) |
| `motion_type` | Motion category (`Linear`, `Circular`, `Helical`, …) |
| `distance` | UAV–BS distance (m) |
| `velocity` | UAV speed (m/s) |
| `acceleration` | UAV acceleration (m/s²) |
| `los_angle_tx` | Line-of-sight azimuth to TX (°) |
| `los_angle_rx` | Line-of-sight azimuth to RX (°) |
| `los_ele_tx` | Line-of-sight elevation to TX (°) |
| `los_ele_rx` | Line-of-sight elevation to RX (°) |
| `velocity_angle` | Direction of velocity vector (°) |
| `mobility_factor` | Composite mobility descriptor |
| `phi_tx_optimal` | **Target** TX azimuth (°) |
| `theta_tx_optimal` | **Target** TX elevation (°) |
| `phi_rx_optimal` | **Target** RX azimuth (°) |
| `theta_rx_optimal` | **Target** RX elevation (°) |

---

## Running the Notebook

### On Google Colab (recommended for GPU)

1. Open `CORTEX_Stage1.ipynb` in Colab.
2. Set `CSV_PATH` in **Cell 2 · Load Data** to your Drive path, or let the auto-detect run.
3. Run all cells — outputs are saved to `CORTEX_Stage1/` in My Drive.

### Locally

```bash
pip install torch pandas numpy matplotlib tqdm scikit-learn

# Edit CSV_PATH in the notebook, then:
jupyter notebook CORTEX_Stage1.ipynb
```

---

## Hyperparameters

All key hyperparameters are in a single cell (**§ 8 · Hyperparameters**):

| Parameter | Default | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | 15 | Input window size (time steps) |
| `D_MODEL` | 256 | Transformer hidden dimension |
| `N_HEADS` | 8 | Attention heads |
| `N_LAYERS` | 4 | Transformer depth |
| `DIM_FF` | 512 | FFN inner dimension |
| `DROPOUT` | 0.2 | Dropout rate |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `NUM_EPOCHS` | 100 | Maximum training epochs |
| `PATIENCE` | `None` | Early-stopping patience (`None` = disabled) |

---

## Outputs

| File | Description |
|---|---|
| `models/stage1_best_model.pth` | Best checkpoint (lowest val loss) |
| `models/stage1_model_<ts>.pth` | Full checkpoint with scaler, encoder & metadata |
| `results/stage1_metrics_<ts>.json` | MAE, RMSE, R² per angle + overall |
| `plots/stage1_training_results.png` | Loss curves, scatter plots, error histograms |

---

## Evaluation Metrics

| Metric | Notes |
|---|---|
| **MAE** | Circular mean absolute error — wraps correctly at 360° for φ |
| **RMSE / MSE** | Standard; computed on circular-wrapped differences |
| **R²** | Coefficient of determination per angle |
| **3-D angular distance** | Great-circle error (degrees) between predicted and true unit vectors |

---

## Loading the Model in Stage II

```python
import torch
from CORTEX_Stage1 import CORTEXAnglePredictor   # or copy the class

ckpt  = torch.load("stage1_model_<ts>.pth", map_location="cpu")
hp    = ckpt["hyperparameters"]

model = CORTEXAnglePredictor(
    input_size      = hp["input_size"],
    d_model         = hp["d_model"],
    n_heads         = hp["n_heads"],
    n_layers        = hp["n_layers"],
    dim_feedforward = hp["dim_feedforward"],
    dropout         = hp["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scaler        = ckpt["scaler"]
label_encoder = ckpt["label_encoder"]
```

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
