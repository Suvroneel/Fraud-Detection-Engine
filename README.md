# RaptorX Fraud Detection Pipeline

A production-grade fraud detection system that ingests raw financial transaction data, engineers entity-level features from behavioral patterns, velocity anomalies, and graph-relationship signals, trains a fraud classification model, and exposes a real-time scoring API with explainability output.

---

## Architecture Overview

```
data/                   Raw CSV files (IEEE-CIS + Elliptic)
notebooks/              EDA, feature engineering, model training
features/               Feature engineering modules
models/                 Trained model artifacts (.pkl, .pt, .json)
api/
  main.py               FastAPI scoring service
  drift_detection.py    PSI and KS drift monitoring
tests/
  benchmark_latency.py  p50/p95/p99 latency benchmarking
reports/                Evaluation plots and analysis
logs/                   SQLite prediction log (auto-created)
Dockerfile
docker-compose.yml
requirements.txt
```

---

## Datasets

### IEEE-CIS Fraud Detection (Primary)
Source: https://www.kaggle.com/c/ieee-fraud-detection/data

- 590K transactions across train_transaction.csv and train_identity.csv
- 434 features after merging on TransactionID
- Binary target: isFraud (3.5% positive rate)
- Features split across transaction-level (393 cols) and identity-level (41 cols)

### Elliptic Bitcoin Transaction Dataset (Graph Signals)
Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

- 203K Bitcoin transactions with 166 anonymized features
- 234K directed edges representing fund flows between transactions
- Labels: 1 (illicit), 2 (licit), unknown (unlabeled)
- Used to derive graph centrality, community fraud density, and network risk signals

---

## Setup and Running

### Prerequisites

- Docker and Docker Compose installed
- Trained model artifacts in the models/ directory (run the notebook first)

### Step 1: Download datasets

Place the following files in the data/ directory:

```
data/
  train_transaction.csv
  train_identity.csv
  test_transaction.csv
  test_identity.csv
  elliptic_txs_classes.csv
  elliptic_txs_edgelist.csv
  elliptic_txs_features.csv
```

### Step 2: Train the model

Open and run the notebook end-to-end:

```bash
cd notebooks
jupyter notebook eda_fraud_detection.ipynb
```

This generates all artifacts in models/:
- xgboost_model.pkl
- nn_model_best.pt
- feature_scaler.pkl
- fill_values.pkl
- encoding_maps.pkl
- model_metadata.json
- elliptic_graph_features.csv
- global_graph_stats.pkl

### Step 3: Start the API

```bash
docker-compose up --build
```

The API will be available at http://localhost:8000

---

## API Endpoints

### POST /score

Score a single transaction for fraud risk.

Request body:
```json
{
  "TransactionAmt": 250.00,
  "ProductCD": "W",
  "card1": 12345,
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "DeviceType": "desktop"
}
```

Response:
```json
{
  "transaction_id": null,
  "risk_score": 0.034521,
  "risk_level": "MINIMAL",
  "top_contributing_factors": [
    {"factor": "No specific high-risk signals detected", "weight": "low"}
  ],
  "model_version": "v1.0.0",
  "latency_ms": 12.4
}
```

Risk levels: MINIMAL, LOW, MEDIUM, HIGH, CRITICAL

### GET /health

Returns API health status and model load state.

### GET /model/info

Returns model metadata including version, training date, and test AUC scores.

### GET /monitoring/stats

Returns recent prediction statistics including mean latency, risk distribution, and p95/p99 latency.

---

## Latency Benchmarking

After starting the API, run:

```bash
python tests/benchmark_latency.py
```

Targets: p50 under 250ms, p95 under 500ms, p99 under 500ms.

---

## Feature Engineering Summary

### Behavioral Aggregation Features (13 features)

Entity-level (card1) historical statistics computed without future leakage using expanding windows:

- card_tx_count_hist: cumulative transaction count per card
- card_amt_mean_hist: expanding mean transaction amount per card
- card_amt_std_hist: expanding standard deviation of transaction amounts
- card_amt_max_hist: expanding maximum transaction amount
- card_amt_zscore: z-score of current amount vs card's historical mean
- tx_hour_of_day: hour of day the transaction occurred
- tx_day_of_week: day of week
- card_mean_hour_hist: card's historical average transaction hour
- card_hour_deviation: absolute deviation from card's normal transaction hour
- card_unique_products_hist: number of unique products per card historically
- card_tx_count_24hr: approximate 24-hour transaction count
- card_amt_ratio_to_mean: ratio of current amount to historical mean
- log_transaction_amt: log-transformed transaction amount

### Velocity Anomaly Features (8 features)

Designed to detect sudden changes in transaction behavior:

- time_since_last_tx: seconds elapsed since the card's previous transaction
- is_after_dormancy: flag for transactions after 30+ day inactivity
- amount_velocity: ratio of current amount to previous transaction amount
- tx_frequency_accel: acceleration in transaction frequency
- addr1_changed: flag for billing address change vs previous transaction
- device_unique_cards: number of unique cards sharing the same device
- device_multi_card_flag: flag when more than 3 cards share a device
- amount_spike_flag: flag for transactions exceeding 3x the card's historical max

### Graph-Relationship Features (5 features)

Derived from entity relationship modeling on the IEEE-CIS dataset and Elliptic Bitcoin graph:

- graph_device_shared_card_count: unique card count per device
- graph_email_domain_fraud_rate: historical fraud rate of the sender's email domain
- graph_card_linkage_degree: total transaction count per card entity
- graph_addr_shared_card_count: unique card count per billing address
- graph_network_risk_score: composite network risk signal combining device, email, and address signals

Elliptic-derived graph signals (PageRank, community fraud density, in/out-degree) are used during training to validate graph feature importance and inform the network risk score design.

### Composite Risk Scores (3 features)

- composite_behavioral_risk: normalized combination of amount z-score and hour deviation
- composite_velocity_risk: normalized combination of dormancy, spike, velocity, and address change signals
- composite_network_risk: normalized graph network risk score

---

## Class Imbalance Handling

Two strategies were implemented and compared:

Strategy 1: scale_pos_weight in XGBoost (ratio of negative to positive class, approximately 28:1). No data augmentation, directly adjusts the loss function.

Strategy 2: SMOTE (Synthetic Minority Oversampling Technique) applied to training set with a target sampling strategy of 10% minority class. Generates synthetic fraud samples in feature space.

XGBoost with scale_pos_weight was selected as the primary strategy based on validation AUC-PR. SMOTE was used for the neural network training comparison.

---

## Temporal Split Strategy

Transactions are sorted by TransactionDT and split strictly by time:

- Train: first 70% of the time range
- Validation: next 15%
- Test: final 15%

No shuffling is applied before splitting. This prevents any future data leakage, which would invalidate the evaluation.

---

## Missing Value Strategy

C features (C1-C14): filled with 0. These are count features where missing means no history.

D features (D1-D15): filled with -1 as a sentinel value indicating the event never occurred. Binary indicator columns are added to preserve the missingness signal.

M features (M1-M9): missing treated as a third category 'Unknown' and label-encoded.

V features (V1-V339): median imputation per feature. Columns with more than 90% missing are dropped entirely.

Identity features (id_01 to id_38): numeric fields get median imputation, categorical fields get 'Unknown' category. Columns with more than 95% missing are dropped.

---

## Model Performance

XGBoost with scale_pos_weight and 500 estimators with early stopping on validation AUC.

Neural network is a 4-layer feedforward network [512, 256, 128, 64] with BatchNorm, Dropout, and Focal Loss to address class imbalance without data augmentation.

Evaluation metrics reported: AUC-ROC, AUC-PR, Precision at 5% FPR, Recall at 5% FPR, F1 at optimal threshold.

See reports/ for full plots including ROC curves, PR curves, SHAP summary, and training loss curves.

---

## Monitoring and Retraining

Every API prediction is logged to logs/predictions.db (SQLite) with:
- Input transaction amount
- Risk score and level
- Latency
- Timestamp

Drift detection runs PSI (Population Stability Index) and KS statistic on incoming feature distributions vs training distributions. See api/drift_detection.py for full criteria.

Retraining is triggered when:
- PSI on risk scores exceeds 0.25
- KS test p-value drops below 0.05
- Model AUC drops below 0.88 on a weekly labeled sample
- False positive rate spikes above 2x baseline
- 30-day scheduled retraining cycle

---

## Known Limitations

- Real-time entity history (rolling 1hr/6hr/24hr window features) is simulated in the API using reasonable defaults. Production deployment would require a Redis entity store updated via a streaming pipeline.
- The Elliptic Bitcoin graph features are cross-dataset signals. They are used to derive and validate graph feature design patterns rather than direct feature transfer, since the two datasets represent different transaction networks.
- The neural network training is compute-intensive. On CPU it may take 15-30 minutes. A GPU runtime (Google Colab or local) is recommended for the notebook.

---

## What This Is Not

This project is not a Kaggle leaderboard submission. The goal is production thinking: documented decisions, temporal integrity, explainability, and deployable infrastructure. Model ensembling and hyperparameter tuning beyond reasonable bounds were intentionally avoided in favor of clean, understandable architecture.
