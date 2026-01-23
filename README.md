# GKI 2026 — Three Machine Learning Projects

This repository contains three complete machine learning projects from the GKI 2026 competition season:

1. **Classification_Project_2** — Icelandic Habitat Classification
2. **TimeSeries_Project_3** — Hot Water Demand Forecasting
3. **NLP_Project_1** — Byte-level Language Modeling (Golden Plate)

Each project is self-contained with its own reproducible pipeline, code, and documentation.

---

## Project Summaries

### 🏞️ Classification_Project_2: Icelandic Habitat Classification

**Task:** Classify 35×35 satellite image patches into one of 71 Icelandic habitat types.

**Input:** 15-channel satellite imagery (12 Sentinel-2 bands + elevation + slope + aspect)

**Approach:** Ensemble of tabular and CNN models
- Tabular models: ExtraTrees, LightGBM, PyTorch MLP (hand-engineered features)
- CNN models: Small CNN, ResNet variants (image-based)
- Final ensemble: 4-model weighted average

**Key Files:**
- `api.py` — FastAPI endpoint for predictions
- `README.md` — full reproducible pipeline and setup instructions

**Location:** [`Classification_Project_2/`](Classification_Project_2/)

---

### 📈 TimeSeries_Project_3: Hot Water Demand Forecasting

**Task:** Forecast 72-hour (3-day) hot water demand for 45 sensors given 672-hour (28-day) history and weather data.

**Input:** Historical sensor readings + weather forecasts + observations

**Approach:** Ensemble of tree-based models with hierarchical reconciliation
- LightGBM (per-sensor, weather-aware)
- ExtraTrees (for diversity)
- Equal weighting, optional hierarchical reconciliation

**Key Files:**
- `api.py` — FastAPI endpoint for real-time forecasts
- `scripts/build_train_full.py` — data pipeline
- `scripts/train_models.py` — model training
- `README.md` — full setup and reproducible pipeline

**Location:** [`TimeSeries_Project_3/`](TimeSeries_Project_3/)

---

### 📝 NLP_Project_1: Byte-level Language Modeling

**Task:** Train a tiny byte-level (0–255) next-byte prediction model under strict constraints.

**Constraints:** 
- ≤ 1 MB submission size
- CPU-only inference
- Metric: bits-per-byte (BPB)

**Approach:** Quantized GRU language model
- 1-2 layer GRU with embeddings and linear output
- Int8 weight quantization + fp16 scaling for size efficiency
- Achieves competitive BPB on held-out validation data

**Key Files:**
- `submission/model.py` — evaluator's entry point
- `scripts/train_tiny_gru.py` — training script
- `submission/gru_config.json` — model configuration
- `README.md` — setup and training instructions

**Location:** [`NLP_Project_1/`](NLP_Project_1/)

---

## Getting Started

Each project is independent. To work on a specific project:

```bash
cd Classification_Project_2
# or
cd TimeSeries_Project_3
# or
cd NLP_Project_1
```

Then see the project's **README.md** for detailed setup, data requirements, and reproducible pipeline instructions.

---

## Repository Structure

```
GKI2026/
├── Classification_Project_2/          # Habitat classification
│   ├── README.md                       # Full documentation
│   ├── api.py, model.py               # API and inference
│   ├── scripts/                        # Training scripts
│   ├── configs/                        # Model configurations
│   ├── requirements.txt                # Python dependencies
│   └── .gitignore
│
├── TimeSeries_Project_3/              # Demand forecasting
│   ├── README.md                       # Full documentation
│   ├── api.py, model.py               # API and inference
│   ├── scripts/                        # Data prep and training
│   ├── data/                           # Raw zip files
│   ├── requirements.txt                # Python dependencies
│   └── .gitignore
│
├── NLP_Project_1/                     # Byte-level LM
│   ├── README.md                       # Full documentation
│   ├── submission/                     # Evaluator submission
│   ├── scripts/                        # Training script
│   ├── requirements.txt                # Python dependencies
│   └── .gitignore
│
└── README.md                           # This file
```

---

## Key Takeaways

| Aspect | Classification | TimeSeries | NLP |
|--------|---|---|---|
| **Type** | Image Classification | Time Series Forecasting | Language Modeling |
| **Ensemble** | 4 heterogeneous models | 2 tree-based models | Single quantized model |
| **Main Challenge** | Feature engineering + model diversity | Weather integration + hierarchical structure | Size constraint (≤1 MB) |
| **Framework** | Scikit-learn, PyTorch, timm | LightGBM, ExtraTrees, scikit-learn | PyTorch (custom quantization) |

---

## Notes

- Each project has a `.gitignore` that excludes large artifacts (data, trained models) but tracks code, configs, and submission files
- All three projects use FastAPI for inference endpoints
- Complete reproducible pipelines are documented in each project's README
- Experimental/auxiliary scripts are noted and explained in project-specific READMEs

For detailed information on any project, see the respective **README.md** file.
