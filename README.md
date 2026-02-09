<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" />
</p>

# PolymerID — Real-Time Polymer Classification from NIR Spectra

**ML-powered identification of 11 polymer types from near-infrared spectroscopy data, enabling automated sorting for industrial recycling.**

> Only 9% of plastic ever produced has been recycled. Contamination from misidentified polymers is a leading cause of recycling stream failure. PolymerID uses machine learning to classify polymers at the point of sorting — in under 50ms per sample.

---

## The Problem

Mixed-plastic waste streams are worth ~$0 when contaminated. A single misidentified polymer (e.g., PVC in a PET bale) can ruin an entire batch. Current manual sorting is slow, expensive, and error-prone. NIR spectroscopy can fingerprint polymers non-destructively, but raw spectra require expert interpretation.

## Our Approach

PolymerID replaces manual spectral interpretation with a trained ML pipeline that takes a raw 128-point NIR spectrum (1550–1950 nm) and returns a polymer classification with calibrated confidence — **95.0% accuracy** on held-out test data using SMOTE-balanced kNN.

### Pipeline Architecture

```
Raw NIR Spectrum (128 channels, 1550-1950 nm)
    │
    ▼
┌─────────────────────────────────────┐
│  1. INGEST & VALIDATE               │
│     • Parse PlasTell CSV format      │
│     • Detect NaN / non-numeric →     │
│       median imputation              │
│     • SHA-256 deduplication          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. OUTLIER DETECTION                │
│     • Pearson correlation to         │
│       group mean spectrum            │
│     • Z-score flagging (z < -2)      │
│     • Negative-correlation filter    │
│     • 51 anomalous spectra removed   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. FEATURE ENGINEERING              │
│     • StandardScaler (μ=0, σ=1)      │
│     • PCA: 99.6% variance in 10 PCs │
│     • SMOTE oversampling for class   │
│       balance across 11 polymer      │
│       types                          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. CLASSIFICATION                   │
│     • Distance-weighted kNN (k=5)    │
│     • Certainty threshold @ 0.60     │
│     • "Uncertain" fallback prevents  │
│       contamination                  │
└──────────────┬──────────────────────┘
               │
               ▼
   Polymer ID + Confidence Score
```

### Supported Polymers

| Polymer | Full Name | Recycling Code | Use Case |
|---------|-----------|:--------------:|----------|
| **PET** | Polyethylene terephthalate | 1 | Bottles, food packaging |
| **HDPE** | High-density polyethylene | 2 | Milk jugs, pipes |
| **LDPE** | Low-density polyethylene | 4 | Film, bags |
| **PP** | Polypropylene | 5 | Containers, automotive |
| **PS** | Polystyrene | 6 | Packaging foam |
| **PVC** | Polyvinyl chloride | 3 | Pipes, window frames |
| **PC** | Polycarbonate | 7 | Electronics, lenses |
| **PMMA** | Polymethyl methacrylate | 7 | Displays, lighting |
| **PA6** | Polyamide 6 (Nylon 6) | 7 | Textiles, gears |
| **PA66** | Polyamide 66 (Nylon 66) | 7 | Engineering parts |
| **ABS** | Acrylonitrile butadiene styrene | 7 | Electronics housings |

---

## Results

### Model Benchmarks (50/50 stratified train/test split, n=282 cleaned spectra)

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|:--------:|:--------:|:-----------:|
| kNN (k=5, baseline) | 87.2% | 0.764 | 0.863 |
| Distance-weighted kNN | 91.0% | 0.820 | 0.905 |
| **kNN + SMOTE** | **95.0%** | **0.944** | **0.952** |
| Class-weighted kNN | 90.1% | 0.880 | 0.901 |
| Optimised kNN (grid search, k=2) | 96.0% | 0.950 | — |
| SVM (RBF kernel) | 84.0% | 0.700 | — |

### Key Findings

- **SMOTE oversampling** delivered the highest macro F1 (0.94), resolving minority-class collapse observed in the baseline
- **Grid search** identified k=2 with uniform weights as optimal (CV F1 = 0.866), achieving 96% test accuracy
- **Outlier removal** (51 spectra via Pearson correlation screening) improved weighted kNN from 0.82 → 0.94 macro F1
- **PCA** confirms separability: 99.6% of spectral variance captured in 10 principal components; clear cluster separation in PC1-PC2 space
- **Certainty thresholding** (>0.60) flags ambiguous samples as "Uncertain" rather than risking contamination — critical for recycling applications

---

## Quick Start

```bash
git clone https://github.com/ViktorSmirnov71/ML-Classification-NIR-PolymerID.git
cd ML-Classification-NIR-PolymerID

pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn scipy

# Run the full pipeline
jupyter notebook ID1_Section_B.ipynb
```

### Project Structure

```
ML-Classification-NIR-PolymerID/
├── ID1_Section_B.ipynb              # Full pipeline (clean notebook)
├── ID1_Section_B_executed.ipynb     # Executed with all outputs & plots
├── data/
│   ├── synthetic_data.csv           # PlasTell spectrophotometer export (35 spectra)
│   └── data_source2.csv             # Lab reference spectra (335 spectra)
└── output/
    ├── 2.02_filtered-data.csv       # Label + spectrum columns only
    ├── 2.03_transformed-data.csv    # Transposed wavelength × spectra matrix
    ├── 2.04_combined-data.csv       # Merged datasets (370 spectra)
    ├── 2.06_duplicate-checked-data.csv
    ├── 2.07_checked-data.csv        # NaN/invalid values imputed
    ├── 3.01_averaged-data.csv       # Per-polymer mean ± std spectra
    └── 3.02_outlier-checked-data.csv # Final cleaned dataset (282 spectra)
```

---

## Technical Details

### Data Acquisition

Spectra are captured using the **PlasTell NIR spectrophotometer** — a portable device that measures diffuse reflectance across 128 channels spanning 1550–1950 nm. Each measurement takes <1 second. The wavelength range targets the first and second overtone C–H, O–H, and N–H absorption bands that differentiate polymer backbone structures.

### Preprocessing

| Step | Method | Rationale |
|------|--------|-----------|
| Invalid value handling | Median imputation | Robust to outliers (vs. mean) |
| Deduplication | SHA-256 spectral hashing | O(n) duplicate detection across merged datasets |
| Outlier detection | Pearson r to group centroid, z-score < -2 | Removes sensor artifacts, mislabelled samples, contaminated reads |
| Feature scaling | StandardScaler (z-normalization) | Required for distance-based classifiers (kNN) |
| Class balancing | SMOTE (k=3 neighbors) | Addresses 24:1 class imbalance (PMMA vs ABS) |

### Dimensionality Analysis (PCA)

| Component | Variance Explained | Cumulative |
|:---------:|:-----------------:|:----------:|
| PC1 | ~70% | ~70% |
| PC2 | ~15% | ~85% |
| PC3 | ~8% | ~93% |
| PC1–10 | — | 99.6% |

The high variance concentration in 2–3 components confirms that polymer identity is primarily encoded in a small number of spectral absorption features — consistent with the known chemistry of C–H and O–H overtone bands in the 1550–1950 nm window.

---

## Roadmap

- [ ] **Real-time inference API** — Flask/FastAPI endpoint for PlasTell hardware integration
- [ ] **Expanded polymer library** — Add PLA, PBT, PEEK, and recycled-grade variants
- [ ] **Convolutional feature extraction** — 1D-CNN on raw spectra to learn discriminative features end-to-end
- [ ] **Edge deployment** — ONNX export for Raspberry Pi / Jetson Nano at the sorting line
- [ ] **Confidence calibration** — Platt scaling for well-calibrated probability outputs
- [ ] **Multi-sensor fusion** — Combine NIR with Raman or LIBS for improved discrimination of spectrally similar pairs (e.g., PA6 vs PA66)

---

## License

MIT

---

<p align="center">
  <sub>Built for the circular economy. Every correctly sorted polymer keeps plastic out of landfill.</sub>
</p>
