# 🚦 VisionZero — Accident Severity Predictor

> Urban Infrastructure & Public Safety | Hackathon Project 

Traffic accidents are rarely caused by a single factor — they are the tragic result of intersecting variables ranging from road conditions and lighting to driver experience and vehicle defects. **VisionZero** is a robust predictive system that forecasts accident severity from post-crash report data, navigating complex, messy, real-world categorical data to find the signal in the noise.

---

## Problem Statement

Build a data-driven system that:

1. **Analyzes** the primary catalysts for severe accidents through deep-dive categorical analysis
2. **Predicts** accident severity (`Slight Injury`, `Serious Injury`, `Fatal Injury`) using a multi-class classification model
3. **Recommends** the top 3 infrastructural or policy changes for local traffic authorities to reduce serious injuries

---

## Real-World Impact

- Empowers **smart-city infrastructure grids** to dynamically allocate emergency response resources
- Helps **policymakers prioritize** specific road redesigns based on empirical hazard data
- Bridges the gap between granular event logs and high-level public safety decisions

---

## Dataset

- **File:** `Road.csv`
- **Size:** 12,316 rows × 32 columns
- **Target Variable:** `Accident_severity`

| Class | Count | Distribution |
|---|---|---|
| Slight Injury | 10,415 | 84.56% |
| Serious Injury | 1,743 | 14.15% |
| Fatal Injury | 158 | 1.28% |

> ⚠️ Severe class imbalance — raw accuracy is misleading. Evaluation focuses on **Macro/Micro F1-Score** and **Confusion Matrix**.

### Key Features

| Feature | Description |
|---|---|
| `Time` / `Day_of_week` | Temporal context of the accident |
| `Age_band_of_driver` / `Sex_of_driver` | Driver demographics |
| `Driving_experience` | Years of driving experience |
| `Type_of_vehicle` | Vehicle type involved |
| `Weather_conditions` | Weather at time of accident |
| `Road_surface_type` / `Road_alignment` | Road infrastructure factors |
| `Number_of_vehicles_involved` | Count of vehicles |
| `Number_of_casualties` | Count of casualties |
| `Cause_of_accident` | Primary cause (20 unique categories) |
| `Casualty_severity` / `Casualty_class` | Casualty-level details |
| `Pedestrian_movement` | Pedestrian involvement |

---

## ML Pipeline

```
Data Loading → EDA → Preprocessing → Feature Engineering
    → Advanced Encoding → SMOTE Resampling → XGBoost Training
        → Evaluation (F1 + Confusion Matrix) → Policy Insights
```

### 1. Exploratory Data Analysis
- Class imbalance visualization
- Distribution analysis across categorical features
- Correlation between accident causes and severity

### 2. Preprocessing
- Missing value imputation
- Feature scaling with `StandardScaler`

### 3. Advanced Categorical Encoding
Handling high-cardinality text-based features:
- **Target Encoding** — for high-cardinality columns (e.g., `Cause_of_accident`, `Vehicle_type`)
- **Label Encoding** — for ordinal features (e.g., `Driving_experience`, `Age_band`)
- **One-Hot Encoding** — for low-cardinality nominals

### 4. Feature Engineering
- Derived temporal and contextual features
- Dataset expanded: 32 → 47 columns after engineering

### 5. Imbalance Handling — SMOTE
- Applied **SMOTE** (Synthetic Minority Oversampling Technique)
- Resampled from 2,889 → 7,398 rows (balanced across all 3 classes)
- Fallback strategy: `sampling_strategy='not majority'`

### 6. Model — XGBoost Classifier (Advanced Optimized)

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 7 |
| `colsample_bytree` | 0.85 |
| `random_state` | 42 |

Train/Test Split: **80/20** → 5,918 training / 1,480 test samples

---

## Results

| Metric | Score |
|---|---|
| Overall Accuracy | **95.95%** |
| Weighted F1-Score | **0.9595** |
| Macro F1-Score | **0.9595** |
| Micro F1-Score | **0.9595** |

- Full confusion matrix with heatmap visualization
- Per-class precision, recall, and F1 breakdown
- Top-15 feature importance chart

---

## Data Science Components

| Component | Technique Used |
|---|---|
| Categorical Encoding | Target Encoding, Label Encoding, One-Hot Encoding |
| Imbalance Handling | SMOTE oversampling |
| Evaluation | Macro F1, Micro F1, Confusion Matrix |
| Model | XGBoost Classifier |
| Scaling | StandardScaler |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, metrics, model evaluation |
| `xgboost` | Gradient boosted classifier |
| `imbalanced-learn` | SMOTE oversampling |
| `category_encoders` | Target / ordinal encoding |

---

## Getting Started

### Run on Google Colab (Recommended)

1. Open `VisionZero_Colab.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `Road.csv` when prompted
3. Run all cells in order

### Run Locally

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn category_encoders
jupyter notebook VisionZero_Colab.ipynb
```

> Make sure `Road.csv` is in the same directory, or update the file path in the data loading cell.

---

## Learning Outcomes

- Mastering **advanced preprocessing** for high-cardinality text columns
- Combating **severe class imbalance** in real-world operational data
- Deriving **high-level policy and infrastructure insights** from granular event logs

---

## License

This project is open-source and available under the [MIT License](LICENSE).
