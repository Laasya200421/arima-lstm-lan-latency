# 📡 Hybrid ARIMA-LSTM Framework for LAN Latency Prediction with Stability Classification and Anomaly Detection

> A hybrid time-series forecasting system combining **ARIMA** (linear patterns) and **LSTM** (nonlinear deep learning) for LAN network latency prediction — with multi-model anomaly detection and stability classification. Published at **16th IEEE ICCCNT 2025, IIT Indore**.

---

## 📌 Project Description

Network latency in Local Area Networks (LANs) is highly dynamic, influenced by both linear trends and complex nonlinear behavioral patterns. Traditional models like ARIMA capture linear temporal dependencies well but fail on nonlinear spikes and irregular patterns. Deep learning models like LSTM capture nonlinearity but lack interpretability on trend components.

This project proposes a **hybrid ARIMA-LSTM framework** that:
- Uses **ARIMA** to model linear latency trends and extract residuals
- Feeds residuals into an **LSTM** to capture nonlinear patterns
- Combines both into a final hybrid forecast
- Applies **three anomaly detection models** (Isolation Forest, One-Class SVM, Autoencoder) to flag network anomalies
- Classifies network stability based on latency behavior

---

## 📄 Publication

**A Hybrid ARIMA-LSTM Framework for LAN Latency Prediction with Stability Classification and Anomaly Detection**  
*Accepted and Presented at the 16th IEEE ICCCNT 2025, IIT Indore*

---

## 🗂️ Dataset

| Property | Details |
|----------|---------|
| Domain | Network / LAN latency time-series |
| Target Variable | `local_avg` (local average latency in ms) |
| Format | Time-series CSV with timestamp index |
| Train / Test Split | 80% / 20% |

### Features Engineered

| Feature | Description |
|---------|-------------|
| `hour`, `day_of_week`, `weekend` | Time-based features |
| `rolling_mean`, `rolling_std` | Rolling window statistics (window=5, 30) |
| `rolling_min`, `rolling_max` | Rolling extremes |
| `skewness`, `kurtosis` | Statistical distribution features |
| `latency_slope` | Rate of change in latency |
| `latency_pct_change` | Percentage change in latency |
| `fft_dominant_freq` | Dominant frequency via Fast Fourier Transform |
| `arima_pred` | ARIMA predictions used as hybrid feature for LSTM |

---

## ⚙️ Methodology

### Pipeline Overview

```
Raw LAN Data
     │
     ▼
Data Preprocessing
(outlier removal, missing value imputation, feature engineering)
     │
     ▼
     ├──► ARIMA Model (linear trend forecasting)
     │         │
     │         └──► Residuals (nonlinear patterns)
     │                    │
     │                    ▼
     └──► Bidirectional LSTM (learns residual patterns)
                    │
                    ▼
          Hybrid Forecast = ARIMA + LSTM Residuals
                    │
                    ▼
          Anomaly Detection (3 models in parallel)
          + Stability Classification
```

### Step 1 — Data Preprocessing
- Removed duplicate timestamps and extreme outliers (±3 standard deviations)
- Forward-filled missing values
- Engineered rolling statistics, FFT features, slope and acceleration features
- Normalized using `MinMaxScaler`
- Applied **SMOTE** to handle class imbalance for anomaly classification

### Step 2 — ARIMA Forecasting
- Checked stationarity using the **Augmented Dickey-Fuller (ADF) test**
- Applied differencing where required
- Fitted `SARIMAX(5, 1, 2)` on the latency series
- Extracted residuals for LSTM input

### Step 3 — Bidirectional LSTM
- Architecture: `Bidirectional LSTM (64) → Bidirectional LSTM (32) → Dense (1)`
- Custom **weighted MSE loss** (penalizes larger errors / latency spikes more)
- Sequence length: 30 time steps
- Trained with Early Stopping (patience=5)
- ARIMA predictions concatenated as an additional feature channel

### Step 4 — Hybrid Combination
```
Hybrid Forecast = ARIMA Forecast + LSTM Predicted Residuals
```

### Step 5 — Anomaly Detection (3 Models)
| Model | Approach | Anomalies Detected |
|-------|----------|--------------------|
| **Isolation Forest** | Unsupervised tree-based | ~1,585 |
| **One-Class SVM** | Kernel-based boundary | ~1,590 |
| **Autoencoder** | Reconstruction error threshold (95th percentile) | ~500 |

Anomalies flagged consistently by both Isolation Forest and One-Class SVM were marked as **high-confidence anomalies**.

---

## 📊 Results

### Model Comparison — RMSE

| Model | RMSE |
|-------|------|
| ARIMA (standalone) | 0.0123 |
| LSTM (standalone) | 0.0091 |
| **Hybrid ARIMA + LSTM** | **0.0051** |

> The hybrid model reduces RMSE by **58.5% vs ARIMA** and **44% vs standalone LSTM**.

### Hybrid Model Evaluation Metrics

| Metric | Result |
|--------|--------|
| RMSE | 0.0051 |
| MAE | (computed per run) |
| R² Score | (computed per run) |
| Custom Accuracy (within 5% error threshold) | Computed via `mean(abs((y_true - y_pred) / y_true) < 0.05) × 100` |

### Anomaly Detection Summary

| Detection Type | Count |
|----------------|-------|
| Detected by both models (high confidence) | Consistent overlap of Isolation Forest + One-Class SVM |
| Isolation Forest only | Subset unique to IF |
| One-Class SVM only | Subset unique to OC-SVM |
| Autoencoder (95th percentile threshold) | ~500 flagged |

---

## 🔍 Key Findings

- **Hybrid outperforms standalone models** — ARIMA alone misses nonlinear spikes; LSTM alone lacks interpretability on trend components; the combination achieves the lowest RMSE.
- **Intensity-based rolling features** (mean, std, slope) are the most important predictors of latency behavior.
- **Isolation Forest and One-Class SVM** show strong agreement on anomaly flagging, making their intersection a reliable high-confidence anomaly signal.
- **Autoencoder** is more conservative, flagging only the most severe reconstruction anomalies.
- FFT-based frequency features add complementary signal for detecting periodic latency spikes.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep Learning | TensorFlow, Keras (Bidirectional LSTM, Autoencoder) |
| Time-Series | Statsmodels (ARIMA, SARIMAX), pmdarima (auto_arima) |
| Anomaly Detection | Scikit-learn (Isolation Forest, One-Class SVM) |
| Data Processing | Pandas, NumPy, MinMaxScaler, SMOTE (imbalanced-learn) |
| Signal Processing | NumPy FFT |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Laasya200421/arima-lstm-lan-latency.git
cd arima-lstm-lan-latency

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline notebook
jupyter notebook notebooks/LAN.ipynb
```

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
statsmodels
pmdarima
tensorflow
keras
scipy
```

