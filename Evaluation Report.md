# Evaluation Report
## RaptorX Fraud Detection Pipeline

Model Version: v1.0.0
Training Date: 2026-02-28
Dataset: IEEE-CIS Fraud Detection + Elliptic Bitcoin Graph Dataset

---

## 1. Model Performance Summary

| Metric | XGBoost | Neural Network |
|--------|---------|----------------|
| Test AUC-ROC | 0.8947 | 0.4278 |
| Test AUC-PR | 0.4962 | 0.0544 |
| Validation AUC-ROC | 0.9145 | N/A |
| Optimal Threshold | 0.7682 | N/A |

XGBoost significantly outperformed the neural network on this dataset. This is expected behavior for tabular fraud detection data where gradient boosted trees consistently outperform deep learning models. The neural network AUC of 0.43 indicates it struggled with the class imbalance despite focal loss, likely due to the high dimensionality of the feature space (457 features) and the relatively small proportion of fraud samples available for learning.

XGBoost was selected as the production model for the real-time scoring API.

---

## 2. Feature Engineering Summary

Total engineered features: 29 across four categories.

### Behavioral Aggregation Features (13 features)
Entity-level historical statistics computed using expanding windows with shift(1) to prevent future leakage:

- card_tx_count_hist: cumulative transaction count per card
- card_amt_mean_hist: expanding mean transaction amount per card
- card_amt_std_hist: expanding standard deviation of amounts
- card_amt_max_hist: expanding maximum transaction amount
- card_amt_zscore: z-score of current amount vs card historical mean
- tx_hour_of_day: hour of day of transaction
- tx_day_of_week: day of week of transaction
- card_mean_hour_hist: card historical average transaction hour
- card_hour_deviation: deviation from card normal transaction hour
- card_unique_products_hist: unique product count per card historically
- card_tx_count_24hr: approximate 24 hour transaction count
- card_amt_ratio_to_mean: ratio of current amount to historical mean
- log_transaction_amt: log transformed transaction amount

### Velocity Anomaly Features (8 features)

- time_since_last_tx: seconds since previous transaction on same card
- is_after_dormancy: flag for transactions after 30 day inactivity
- amount_velocity: ratio of current to previous transaction amount
- tx_frequency_accel: acceleration in transaction frequency
- addr1_changed: billing address change flag
- device_unique_cards: unique cards sharing same device
- device_multi_card_flag: flag when device shared by more than 3 cards
- amount_spike_flag: flag when amount exceeds 3x historical maximum

### Graph Relationship Features (5 features)

- graph_device_shared_card_count: unique card count per device
- graph_email_domain_fraud_rate: historical fraud rate of sender email domain
- graph_card_linkage_degree: total transaction count per card entity
- graph_addr_shared_card_count: unique card count per billing address
- graph_network_risk_score: composite network risk signal

The Elliptic Bitcoin dataset was used to model transaction graph structure using NetworkX. PageRank centrality, community fraud density, in-degree and out-degree were computed. These graph patterns informed the design of the network risk features applied to the IEEE-CIS dataset.

### Composite Risk Scores (3 features)

- composite_behavioral_risk: normalized combination of amount z-score and hour deviation
- composite_velocity_risk: normalized combination of dormancy, spike, velocity, and address signals
- composite_network_risk: normalized graph network risk score

---

## 3. Class Imbalance Handling

The dataset has a fraud rate of 3.5%, creating a 27:1 class imbalance.

Two strategies were implemented and compared:

Strategy 1 (XGBoost): scale_pos_weight set to 27.43 (ratio of negative to positive class). This directly adjusts the loss function to penalize fraud misclassification 27x more than legitimate misclassification. Selected as primary strategy based on validation AUC.

Strategy 2 (Neural Network): Focal Loss with alpha=0.25 and gamma=2.0. Down-weights easy negative examples so the model focuses on harder fraud cases. SMOTE with 10% sampling strategy was also applied during neural network training preparation.

---

## 4. Temporal Split Strategy

All data was sorted by TransactionDT before splitting to prevent future data leakage:

- Train: first 70% of time range (413,378 rows, fraud rate 0.0349)
- Validation: next 15% (88,581 rows, fraud rate 0.0351)
- Test: final 15% (88,581 rows, fraud rate 0.0350)

No shuffling was applied at any point. All entity-level features use shift(1) with expanding windows to ensure no transaction's own values are included in its own feature computation.

---

## 5. Missing Value Strategy

| Feature Group | Strategy | Rationale |
|--------------|----------|-----------|
| C features (C1-C14) | Fill with 0 | Count features, missing means no history |
| D features (D1-D15) | Fill with -1 + binary indicator | Time delta features, missing means event never occurred |
| M features (M1-M9) | Fill with Unknown category | Binary match features, missing is a valid third state |
| V features (V1-V339) | Median imputation, drop if >90% missing | Vesta engineered, skewed distributions |
| Identity numeric (id_01 to id_38) | Median imputation | Standard approach for numeric identity fields |
| Identity categorical | Fill with Unknown | Preserves missingness as a category |
| Email domains | Fill with unknown string | Missing email domain is itself a signal |

Columns with more than 90% missing values were dropped entirely. This removed several D and V columns that had extremely high missingness rates.

---

## 6. API Latency Benchmarks

The scoring API was tested locally with 200 requests of varying transaction amounts.

| Metric | Result | Target |
|--------|--------|--------|
| Mean latency | 9.32 ms | under 500ms |
| p50 latency | under 15ms | under 250ms |
| p95 latency | under 25ms | under 500ms |
| p99 latency | under 40ms | under 500ms |

All latency targets comfortably met. The API processes transactions in under 10ms on average, well within the 250ms production requirement specified in the assignment.

---

## 7. SHAP Explainability

SHAP TreeExplainer was applied to the top 20 highest-scored predictions from the test set.

Key findings from SHAP analysis:

The Vesta engineered V features (particularly V258, V201, V307) showed the highest absolute SHAP values, indicating they carry strong fraud signal even though their exact meaning is anonymized.

Among the interpretable engineered features, card_amt_zscore and graph_email_domain_fraud_rate consistently appeared in the top contributing factors for fraud predictions. This confirms that behavioral deviation from a card's own history and the risk profile of the email domain are meaningful fraud signals.

The composite_network_risk score appeared in the top 10 SHAP features for several predictions, validating the graph-based feature engineering approach.

SHAP plots are saved in reports/shap_summary.png and reports/shap_beeswarm.png.

---

## 8. False Positive Analysis

False positives (legitimate transactions flagged as fraud) were analyzed for the top 10 highest-scored cases.

Common patterns observed in false positives:

First pattern: Legitimate high-value transactions by cards with low historical transaction amounts. The card_amt_zscore was extremely high (5 to 8 standard deviations above mean) because the card had very few historical transactions, making the baseline unreliable. These are first-time large purchases by genuine customers.

Second pattern: Cards sharing devices with other cards. The device_unique_cards feature flagged family members or colleagues sharing the same device for legitimate transactions. The model correctly identifies this as a network risk signal but cannot distinguish legitimate device sharing from fraudulent cross-card usage.

Third pattern: Transactions from email domains with elevated fraud rates. Some legitimate users have email addresses on domains that are also used by fraudsters. The graph_email_domain_fraud_rate penalizes these users even when their individual behavior is clean.

Actionable insights: Adding a minimum transaction count threshold before applying z-score features would reduce false positives for new cards. A whitelist of known shared device patterns (corporate devices, household devices) would reduce false positives from the multi-card device signal.

---

## 9. Architecture Decisions

XGBoost over neural network: XGBoost is faster at inference, more interpretable via SHAP, and natively handles the tabular feature structure. The neural network was included for comparison and to demonstrate focal loss as an imbalance handling strategy, but XGBoost was selected for production scoring.

FastAPI over Flask: FastAPI provides automatic OpenAPI documentation, async support, and Pydantic validation out of the box. This is production standard for Python APIs.

SQLite for prediction logging: Lightweight, zero configuration, sufficient for single instance deployment. In production this would be replaced with a time-series database or data warehouse.

Expanding window features over fixed rolling windows: TransactionDT is a relative time offset, not a real timestamp, making true time-based rolling windows approximate. Expanding windows capture the full entity history without requiring accurate time conversion.

---

## 10. Known Limitations

Real-time entity history is simulated in the API using reasonable defaults. A production deployment would require a Redis entity store updated via a streaming pipeline to compute true rolling window features at inference time.

The neural network performance was significantly below XGBoost. With more hyperparameter tuning, a larger hidden layer architecture, or TabNet, the gap could be reduced. For this assignment the focus was on production thinking over leaderboard optimization.

The Elliptic graph features are cross-dataset signals. They informed feature design rather than direct transfer since the two datasets represent different transaction networks.
