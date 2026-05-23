# Battery Health Prediction, Professional ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Three-task ML pipeline for battery health monitoring: State of Health (SOH) regression, Remaining Useful Life (RUL) regression, and Fault Classification, applicable to EV manufacturing, Battery Management Systems, and grid storage.

---

## Problem Statement

Battery degradation causes EV range anxiety, unexpected failures, and costly premature replacements. Accurate health prediction enables proactive replacement scheduling, warranty management, and BMS optimisation.

---

## Three ML Tasks

| Task | Type | Target | Use Case |
|------|------|--------|---------|
| SOH Prediction | Regression | State of Health % | Current battery capacity |
| RUL Prediction | Regression | Remaining cycles | Replacement scheduling |
| Fault Detection | Classification | Fault type | BMS safety alerts |

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-Model Comparison | Random Forest, Gradient Boosting, Ridge, Decision Tree |
| SMOTE Balancing | Imbalanced fault class handling |
| Model Serialisation | joblib persistence for all trained models |
| EDA Visualisations | Degradation curves, feature correlations, fault distribution |
| Prediction Script | Single-battery health scoring |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn (RF, GBM, Ridge), imbalanced-learn |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Serialisation | joblib |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/battery-health-prediction.git
cd battery-health-prediction
pip install -r requirements.txt  # or: pip install pandas scikit-learn imbalanced-learn matplotlib seaborn joblib
python data_generator.py
python eda.py
python train.py
python predict.py
```

---

## Author

**Momah Moses**, Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
