# NIR Polymer Classification — AI-Aided Identification for Chemical Recycling

**Imperial College London — ID1: Informing Chemical Recycling Strategies using AI-Aided Polymer Identification**

Machine learning pipeline for identifying 15 polymer types from near-infrared (NIR) spectral data collected using a PlasTell Desktop spectrophotometer. Achieves **97% test accuracy** with optimised kNN (k=1) on cleaned data, and **95.9% accuracy** when addressing class imbalance with SMOTE oversampling.

---

## Context

Only ~9% of plastic ever produced has been recycled. A major barrier is **sorting accuracy** — contamination from misidentified polymers (e.g., PVC in a PET recycling stream) can ruin entire batches. NIR spectroscopy provides non-destructive polymer fingerprinting in under a second, but raw spectra require systematic processing and trained classifiers to be useful at scale.

This project builds the complete pipeline from raw spectral data to polymer identification, including data cleaning, dark-plastic separation, model training, and unknown sample classification.

---

## How the PlasTell Spectrometer Works

The PlasTell Desktop uses **transmittance-reflectance (transflectance) NIR spectroscopy** operating at 1550–1950 nm with 128 wavelength channels:

- **Two tungsten halogen lamps** (upper for reflectance, lower for transmittance) and an **InGaAs detector**
- NIR light is absorbed by molecular bond vibrations — specifically **C–H, O–H, and N–H overtone and combination bands** — producing polymer-specific absorption fingerprints
- The **Beer-Lambert law** (A = ε × c × l) governs absorption: band depth depends on the material's absorptivity (ε), concentration (c), and optical path length (l = sample thickness)
- For transparent/thin samples, the device uses transflectance mode (light passes through, reflects, and returns), effectively doubling the path length

**Limitation — material thickness:** Thin films (< 1 mm) produce weaker spectral features due to shorter path lengths. Very thin samples (< 20 μm) can produce flat, featureless spectra indistinguishable from dark/opaque plastics. Mitigation: fold or stack thin films before scanning.

---

## Pipeline Overview

```
Raw PlasTell CSV (334 spectra) + Lab Reference CSV (335 spectra)
    │
    ├── Activity 2: Data Processing
    │   ├── 2.01  Explore class distribution (15 polymers, PET max=37, LDPE min=6)
    │   ├── 2.02  Filter to label + spectrum columns
    │   ├── 2.03  Transpose to wavelength × spectra matrix (128 × 334)
    │   ├── 2.04  Combine both sources → 669 spectra
    │   ├── 2.05  Check class balance (PMMA=143, TPU=15, ~10:1 ratio)
    │   ├── 2.06  SHA-256 duplicate detection → 666 unique spectra
    │   └── 2.07  Repair invalid values via linear interpolation → 666 clean spectra
    │
    ├── Activity 3: Spectral Exploration & Outlier Detection
    │   ├── 3.01  Mean ± std spectra per polymer (666 spectra)
    │   └── 3.02  Colour-based dark/light separation → 86 dark removed, 580 light retained
    │
    ├── Activity 4: Classification
    │   ├── 4.01  Standard kNN (k=5) → 93% accuracy
    │   ├── 4.02  Distance-weighted kNN + certainty threshold → 93% accuracy, 12 uncertain
    │   ├── 4.03  Grid search (k=1 optimal) → 97% accuracy; SVM comparison → 86%
    │   └── 4.04  Before/after comparison: dark removal yields +5.6–8.0 pp accuracy gain
    │
    ├── Activity 5: Unknown Identification
    │   └── 23 unknown samples classified via majority voting (14 high, 5 moderate, 4 low confidence)
    │
    └── Extensions
        ├── PCA: first 3 PCs capture 84.9% variance (PC1=43.8%, PC2=29.5%, PC3=11.6%)
        ├── Class imbalance: SMOTE → 95.9% accuracy (macro F1=0.951)
        └── Two-stage dark classifier: Stage 1 light/dark detection 98.3%, Stage 2b dark polymer ID 49.9%
```

---

## Results Summary

### Classification Performance (580 light spectra, 50/50 stratified split)

| Model | Accuracy | Macro F1 | Notes |
|-------|:--------:|:--------:|-------|
| Standard kNN (k=5) | 93.4% | 0.902 | Baseline; LDPE F1=0.59, PLA F1=0.67 |
| Distance-weighted kNN (k=5) | 93.4% | 0.927 | ABS F1: 0.82→0.95; 12 uncertain flagged |
| **Optimised kNN (k=1, grid search)** | **96.9%** | **0.963** | **7 classes at F1=1.00; LDPE→0.78, PLA→0.89** |
| SVM (RBF, default params) | 85.5% | 0.810 | LDPE F1=0.00, PP F1=0.29; needs tuning |
| kNN + SMOTE oversampling | 95.9% | 0.951 | Best macro F1; LDPE F1: 0.59→0.86 |
| kNN + class weighting | 95.9% | 0.949 | Matches SMOTE without synthetic data |

### Impact of Dark-Spectrum Removal

| Model | Accuracy (before → after) | Change |
|-------|:-------------------------:|:------:|
| kNN (k=5) | 87.1% → 93.4% | +6.3 pp |
| Weighted kNN (k=5) | 89.2% → 94.8% | +5.6 pp |
| SVM (RBF) | 77.5% → 85.5% | +8.0 pp |

Removing 86 dark/pigmented spectra (12.9% of data) where carbon black absorption suppresses polymer-specific features yields consistent 6–8 percentage-point accuracy gains across all models.

### Key Findings

- **k=1 is optimal** after dark-spectrum removal — polymer clusters are tight enough that the single nearest neighbour is the most reliable predictor
- **LDPE/HDPE confusion** is a hardware limitation (shared polyethylene backbone), not a model limitation — resolving it requires wider spectral range or supplementary measurements
- **Precision matters most for recycling** — contamination (e.g., PVC in PET) ruins batches. Weighted kNN's certainty threshold trades 4.1% uncertain predictions for substantially fewer misclassifications
- **Dark plastics retain partial identifiability** — kNN achieves 49.9% on dark polymer classification (3.5× above chance), but practical deployment requires more training data

---

## Supported Polymers (15 types)

| Polymer | Full Name | Recycling Code | Spectra (light) | Test F1 (k=1) |
|---------|-----------|:--------------:|:---------------:|:--------------:|
| ABS | Acrylonitrile butadiene styrene | 7 | 21 | 1.00 |
| HDPE | High-density polyethylene | 2 | 38 | 0.93 |
| LDPE | Low-density polyethylene | 4 | 21 | 0.78 |
| PA6 | Polyamide 6 (Nylon 6) | 7 | 27 | 1.00 |
| PA66 | Polyamide 66 (Nylon 66) | 7 | 31 | 1.00 |
| PC | Polycarbonate | 7 | 47 | 0.98 |
| PET | Polyethylene terephthalate | 1 | 46 | 0.96 |
| PETG | PET glycol-modified | 1 | 26 | 1.00 |
| PLA | Polylactic acid | 7 | 17 | 0.89 |
| PMMA | Polymethyl methacrylate | 7 | 119 | 0.97 |
| POM | Polyoxymethylene (Delrin) | 7 | 16 | 1.00 |
| PP | Polypropylene | 5 | 48 | 0.94 |
| PS | Polystyrene | 6 | 31 | 1.00 |
| PVC | Polyvinyl chloride | 3 | 77 | 0.99 |
| TPU | Thermoplastic polyurethane | 7 | 15 | 1.00 |

---

## Project Structure

```
ML-Classification-NIR-PolymerID/
│
├── ID1_Section_B.ipynb                  # Source notebook (all code + analytical markdown)
├── ID1_Section_B_executed_new.ipynb     # Executed notebook with all outputs and plots
├── ID1_Section_B_executed.ipynb         # Earlier executed version (pre-PMMA fix)
├── 2025 - ID1 Experiment Guidance.pdf   # Imperial College lab guidance document
├── README.md
├── .gitignore
│
├── data/
│   ├── matoha-data_3.csv               # PlasTell export: 334 spectra, 15 polymers
│   ├── data_source2.csv                # Lab reference spectra: 335 spectra
│   ├── matoha-data-unknown.csv         # Unknown samples for identification
│   ├── plastell_data.csv               # Additional PlasTell data
│   └── synthetic_data.csv              # Synthetic/supplementary spectra
│
└── output/                             # Intermediate pipeline outputs
    ├── 2.02_filtered-data.csv          # Label + spectrum columns only
    ├── 2.03_transformed-data.csv       # Transposed wavelength × spectra matrix
    ├── 2.04_combined-data.csv          # Merged datasets (669 spectra)
    ├── 2.06_duplicate-checked-data.csv # After SHA-256 deduplication (666)
    ├── 2.07_checked-data.csv           # After invalid value repair (666)
    ├── 3.01_averaged-data.csv          # Per-polymer mean spectra (pre-separation)
    ├── 3.02_averaged-data.csv          # Per-polymer mean spectra (post-separation)
    └── 3.02_outlier-checked-data.csv   # Final cleaned dataset (580 light spectra)
```

---

## Quick Start

### Requirements

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/ViktorSmirnov71/ML-Classification-NIR-PolymerID.git
cd ML-Classification-NIR-PolymerID

pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn scipy pino
```

### Running

```bash
jupyter notebook ID1_Section_B.ipynb
```

Run all cells sequentially. The notebook is self-contained — all data files are in `data/` and intermediate outputs are saved to `output/` automatically.

To view pre-computed results without re-running, open `ID1_Section_B_executed_new.ipynb`.

---

## Methodology Notes

### Dark/Light Colour Separation (Task 3.02)

Rather than applying blind statistical outlier detection, we use **physically motivated colour-based separation**. Carbon black pigment in dark/black plastics absorbs broadband across 1550–1950 nm, suppressing the polymer-specific absorption features that kNN relies on. These spectra are systematically different from light-coloured spectra of the same polymer and degrade classifier performance if included in training.

**86 dark spectra** were identified and separated via:
- Automatic detection from PlasTell colour metadata (black-labelled samples)
- Manual heatmap inspection for data_source2 spectra lacking metadata (62 overrides)
- 1 measurement outlier (PET.12 — sensor/placement error)

The separated dark spectra are not discarded — they are analysed separately and used in the two-stage dark-plastic classification extension.

### kNN vs Weighted kNN: Choosing the Right Metric

Both models achieve 93% accuracy, but their error profiles differ:

| Metric | When it matters | Model preference |
|--------|----------------|-----------------|
| **Precision** | Contamination is costly (PVC in PET stream) | Weighted kNN (0.97 vs 0.93) |
| **Recall** | Losing recyclable material is costly | Standard kNN (slightly higher for PET) |
| **Macro F1** | All polymer classes matter equally | Weighted kNN (excluding Uncertain class) |

**Recommendation:** Weighted kNN for deployment — the certainty threshold (0.6) converts low-confidence wrong predictions into "Uncertain" flags routed to manual inspection, reducing contamination risk.

---

## Acknowledgements

**Imperial College London** — Dr Benji Fenech-Salerno & Dr Rebecca L. Jones (ID1 Lab)

**PlasTell** by Matoha Instrumentation — NIR spectrophotometer hardware

---

## License

MIT
