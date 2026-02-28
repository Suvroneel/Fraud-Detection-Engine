"""
RaptorX Fraud Detection Scoring API
Real-time transaction risk scoring with explainability output.
"""

import os
import json
import time
import logging
import sqlite3
import numpy as np
import pandas as pd
import joblib
import torch

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
LOG_DB    = BASE_DIR / 'logs' / 'predictions.db'
LOG_DB.parent.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title='RaptorX Fraud Detection API',
    description='Real-time transaction fraud scoring with behavioral, velocity, and graph-based risk signals.',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ---- Pydantic Models ----

class TransactionRequest(BaseModel):
    TransactionID   : Optional[int]   = Field(None,  description='Unique transaction identifier')
    TransactionAmt  : float            = Field(...,   description='Transaction amount in USD')
    ProductCD       : Optional[str]    = Field('W',   description='Product category code')
    card1           : Optional[int]    = Field(None,  description='Card identifier')
    card2           : Optional[float]  = Field(None,  description='Card attribute 2')
    card3           : Optional[float]  = Field(None,  description='Card attribute 3')
    card4           : Optional[str]    = Field(None,  description='Card network (visa, mastercard, etc)')
    card5           : Optional[float]  = Field(None,  description='Card attribute 5')
    card6           : Optional[str]    = Field(None,  description='Card type (debit/credit)')
    addr1           : Optional[float]  = Field(None,  description='Billing address region')
    addr2           : Optional[float]  = Field(None,  description='Billing address country')
    P_emaildomain   : Optional[str]    = Field(None,  description='Purchaser email domain')
    R_emaildomain   : Optional[str]    = Field(None,  description='Recipient email domain')
    DeviceType      : Optional[str]    = Field(None,  description='Device type (mobile/desktop)')
    DeviceInfo      : Optional[str]    = Field(None,  description='Device information string')
    TransactionDT   : Optional[int]    = Field(None,  description='Transaction datetime offset (seconds)')


class ScoreResponse(BaseModel):
    transaction_id          : Optional[int]
    risk_score              : float
    risk_level              : str
    top_contributing_factors: list
    model_version           : str
    latency_ms              : float


# ---- Model Loading ----

class ModelStore:
    """Holds all loaded models and artifacts in memory for fast inference."""

    def __init__(self):
        self.xgb_model       = None
        self.nn_model        = None
        self.scaler          = None
        self.fill_values     = None
        self.encoding_maps   = None
        self.metadata        = None
        self.feature_cols    = None
        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._loaded         = False

    def load(self):
        if self._loaded:
            return
        logger.info('Loading models from %s', MODEL_DIR)
        try:
            self.xgb_model     = joblib.load(MODEL_DIR / 'xgboost_model.pkl')
            self.scaler        = joblib.load(MODEL_DIR / 'feature_scaler.pkl')
            self.fill_values   = joblib.load(MODEL_DIR / 'fill_values.pkl')
            self.encoding_maps = joblib.load(MODEL_DIR / 'encoding_maps.pkl')

            with open(MODEL_DIR / 'model_metadata.json') as f:
                self.metadata = json.load(f)

            self.feature_cols = self.metadata['feature_cols']
            self._loaded = True
            logger.info('Models loaded successfully. Version: %s', self.metadata.get('model_version'))
        except Exception as e:
            logger.error('Failed to load models: %s', str(e))
            raise RuntimeError(f'Model loading failed: {str(e)}')


model_store = ModelStore()


@app.on_event('startup')
def startup_event():
    """Load models on startup and initialize prediction database."""
    model_store.load()
    _init_db()
    logger.info('API startup complete')


def _init_db():
    """Initialize SQLite database for prediction logging."""
    conn = sqlite3.connect(LOG_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            transaction_id  INTEGER,
            transaction_amt REAL,
            risk_score      REAL,
            risk_level      TEXT,
            latency_ms      REAL,
            input_features  TEXT
        )
    ''')
    conn.commit()
    conn.close()


def _log_prediction(transaction_id, transaction_amt, risk_score, risk_level, latency_ms, features_json):
    """Log every prediction to SQLite for monitoring and drift detection."""
    try:
        conn = sqlite3.connect(LOG_DB)
        conn.execute(
            '''INSERT INTO predictions
               (timestamp, transaction_id, transaction_amt, risk_score, risk_level, latency_ms, input_features)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.utcnow().isoformat(),
                transaction_id,
                transaction_amt,
                risk_score,
                risk_level,
                latency_ms,
                features_json,
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning('Prediction logging failed: %s', str(e))


# ---- Feature Engineering (Real-Time) ----

def compute_realtime_features(tx: TransactionRequest) -> Dict[str, float]:
    """
    Compute features for a single incoming transaction.
    In production this would query a Redis entity store for historical aggregates.
    Here we simulate with reasonable defaults and available fields.
    """
    enc = model_store.encoding_maps
    fv  = model_store.fill_values

    tx_dt = tx.TransactionDT or int(time.time())

    # Base features
    features = {
        'TransactionAmt'     : tx.TransactionAmt,
        'log_transaction_amt': np.log1p(tx.TransactionAmt),
        'tx_hour_of_day'     : (tx_dt // 3600) % 24,
        'tx_day_of_week'     : (tx_dt // 86400) % 7,
        'card1'              : tx.card1 or 0,
        'card2'              : tx.card2 or fv.get('card2', 0),
        'card3'              : tx.card3 or fv.get('card3', 0),
        'card5'              : tx.card5 or fv.get('card5', 0),
        'addr1'              : tx.addr1 or fv.get('addr1', 0),
        'addr2'              : tx.addr2 or fv.get('addr2', 0),
    }

    # Frequency encoded fields
    freq_maps = {
        'P_emaildomain': enc.get('freq_P_emaildomain', {}),
        'R_emaildomain': enc.get('freq_R_emaildomain', {}),
        'DeviceInfo'   : enc.get('freq_DeviceInfo', {}),
    }
    field_vals = {
        'P_emaildomain': tx.P_emaildomain or 'unknown',
        'R_emaildomain': tx.R_emaildomain or 'unknown',
        'DeviceInfo'   : tx.DeviceInfo or 'unknown',
    }
    for col, fmap in freq_maps.items():
        features[f'{col}_freq'] = fmap.get(field_vals[col], 0.0)

    # Target encoded fields
    target_maps = {
        'ProductCD' : enc.get('target_ProductCD', {}),
        'card4'     : enc.get('target_card4', {}),
        'card6'     : enc.get('target_card6', {}),
        'DeviceType': enc.get('target_DeviceType', {}),
    }
    target_vals = {
        'ProductCD' : tx.ProductCD or 'W',
        'card4'     : tx.card4 or 'unknown',
        'card6'     : tx.card6 or 'unknown',
        'DeviceType': tx.DeviceType or 'unknown',
    }
    global_mean = 0.035
    for col, tmap in target_maps.items():
        features[f'{col}_target_enc'] = tmap.get(target_vals[col], global_mean)

    # Behavioral features (simulated from entity history defaults)
    # In production: query Redis for card1 entity stats
    features['card_tx_count_hist']      = 5.0   # Assume moderate history
    features['card_amt_mean_hist']       = tx.TransactionAmt * 0.9
    features['card_amt_std_hist']        = tx.TransactionAmt * 0.3
    features['card_amt_max_hist']        = tx.TransactionAmt * 1.2
    features['card_amt_zscore']          = 0.0
    features['card_mean_hour_hist']      = 12.0
    features['card_hour_deviation']      = abs(features['tx_hour_of_day'] - 12.0)
    features['card_unique_products_hist']= 2.0
    features['card_tx_count_24hr']       = 3.0
    features['card_amt_ratio_to_mean']   = 1.0

    # Velocity features
    features['time_since_last_tx']  = 86400.0   # Assume 1 day since last tx
    features['is_after_dormancy']   = 0
    features['amount_velocity']     = 1.0
    features['tx_frequency_accel']  = 0.5
    features['addr1_changed']       = 0
    features['device_unique_cards'] = 1.0
    features['device_multi_card_flag'] = 0
    features['amount_spike_flag']   = int(tx.TransactionAmt > 1000)

    # Graph features
    features['graph_device_shared_card_count'] = 1.0
    features['graph_email_domain_fraud_rate']   = freq_maps['P_emaildomain'].get(
        field_vals['P_emaildomain'], global_mean
    )
    features['graph_card_linkage_degree']        = 10.0
    features['graph_addr_shared_card_count']     = 1.0
    features['graph_network_risk_score']         = (
        features['graph_email_domain_fraud_rate'] * 0.4 +
        min(features['graph_device_shared_card_count'] / 100, 1.0) * 0.4 +
        min(features['graph_addr_shared_card_count'] / 50, 1.0) * 0.2
    )

    # Composite risk scores
    features['composite_behavioral_risk'] = min(abs(features['card_amt_zscore']) / 5.0, 1.0)
    features['composite_velocity_risk']   = (
        features['is_after_dormancy'] * 0.3 +
        features['amount_spike_flag'] * 0.4 +
        min(features['amount_velocity'], 5.0) / 5.0 * 0.3
    )
    features['composite_network_risk']    = features['graph_network_risk_score']

    return features


def build_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Align features to the exact order expected by the trained model."""
    vector = []
    for col in model_store.feature_cols:
        vector.append(features.get(col, 0.0))
    return np.array(vector, dtype=np.float32).reshape(1, -1)


def get_risk_level(score: float) -> str:
    if score >= 0.80:
        return 'CRITICAL'
    elif score >= 0.60:
        return 'HIGH'
    elif score >= 0.40:
        return 'MEDIUM'
    elif score >= 0.20:
        return 'LOW'
    else:
        return 'MINIMAL'


def get_top_contributing_factors(features: Dict[str, float], score: float) -> list:
    """
    Return the top contributing risk factors in human-readable form.
    In production this would use SHAP values per transaction.
    Here we use rule-based explanation from feature values.
    """
    factors = []

    if features.get('amount_spike_flag', 0) == 1:
        factors.append({'factor': 'Unusually large transaction amount', 'weight': 'high'})

    if features.get('is_after_dormancy', 0) == 1:
        factors.append({'factor': 'Transaction after account dormancy', 'weight': 'high'})

    if features.get('device_multi_card_flag', 0) == 1:
        factors.append({'factor': 'Device shared across multiple cards', 'weight': 'high'})

    if features.get('addr1_changed', 0) == 1:
        factors.append({'factor': 'Billing address changed from previous transaction', 'weight': 'medium'})

    if features.get('graph_email_domain_fraud_rate', 0) > 0.10:
        factors.append({'factor': 'High-risk email domain detected', 'weight': 'medium'})

    if features.get('card_amt_zscore', 0) > 2.5:
        factors.append({'factor': 'Transaction amount significantly above card average', 'weight': 'medium'})

    if features.get('composite_velocity_risk', 0) > 0.6:
        factors.append({'factor': 'High transaction velocity detected', 'weight': 'medium'})

    if not factors:
        factors.append({'factor': 'No specific high-risk signals detected', 'weight': 'low'})

    return factors[:5]


# ---- API Endpoints ----

@app.get('/health')
def health_check():
    """Health check endpoint."""
    return {
        'status'       : 'healthy',
        'model_loaded' : model_store._loaded,
        'timestamp'    : datetime.utcnow().isoformat(),
    }


@app.get('/model/info')
def model_info():
    """Model metadata endpoint."""
    if not model_store._loaded:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return {
        'model_version'  : model_store.metadata.get('model_version'),
        'training_date'  : model_store.metadata.get('training_date'),
        'n_features'     : model_store.metadata.get('n_features'),
        'xgb_test_auc'   : model_store.metadata.get('xgb_test_auc'),
        'nn_test_auc'    : model_store.metadata.get('nn_test_auc'),
        'framework'      : 'XGBoost + PyTorch FeedForward',
        'dataset'        : 'IEEE-CIS Fraud Detection + Elliptic Bitcoin Graph',
    }


@app.post('/score', response_model=ScoreResponse)
def score_transaction(tx: TransactionRequest):
    """
    Score a single transaction for fraud risk.
    Returns risk_score (0-1), risk_level, and top contributing factors.
    Response time target: under 500ms.
    """
    if not model_store._loaded:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    start_time = time.perf_counter()

    try:
        # Compute features
        features   = compute_realtime_features(tx)
        feat_vector = build_feature_vector(features)

        # Handle any NaN values
        feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Score with XGBoost (primary model)
        risk_score = float(model_store.xgb_model.predict_proba(feat_vector)[0, 1])

        # Get risk level and contributing factors
        risk_level   = get_risk_level(risk_score)
        top_factors  = get_top_contributing_factors(features, risk_score)

        latency_ms   = (time.perf_counter() - start_time) * 1000

        # Log prediction
        _log_prediction(
            transaction_id  = tx.TransactionID,
            transaction_amt = tx.TransactionAmt,
            risk_score      = risk_score,
            risk_level      = risk_level,
            latency_ms      = latency_ms,
            features_json   = json.dumps({k: float(v) for k, v in features.items() if isinstance(v, (int, float))})
        )

        logger.info(
            'Scored transaction %s | score=%.4f | level=%s | latency=%.1fms',
            tx.TransactionID, risk_score, risk_level, latency_ms
        )

        return ScoreResponse(
            transaction_id           = tx.TransactionID,
            risk_score               = round(risk_score, 6),
            risk_level               = risk_level,
            top_contributing_factors = top_factors,
            model_version            = model_store.metadata.get('model_version', 'v1.0.0'),
            latency_ms               = round(latency_ms, 2),
        )

    except Exception as e:
        logger.error('Scoring failed: %s', str(e))
        raise HTTPException(status_code=500, detail=f'Scoring error: {str(e)}')


@app.get('/monitoring/stats')
def monitoring_stats():
    """
    Basic monitoring endpoint: recent prediction distribution and drift signals.
    """
    try:
        conn    = sqlite3.connect(LOG_DB)
        df_logs = pd.read_sql('SELECT * FROM predictions ORDER BY id DESC LIMIT 1000', conn)
        conn.close()

        if df_logs.empty:
            return {'message': 'No predictions logged yet'}

        stats = {
            'total_predictions'  : len(df_logs),
            'mean_risk_score'    : float(df_logs['risk_score'].mean()),
            'mean_latency_ms'    : float(df_logs['latency_ms'].mean()),
            'p95_latency_ms'     : float(df_logs['latency_ms'].quantile(0.95)),
            'p99_latency_ms'     : float(df_logs['latency_ms'].quantile(0.99)),
            'risk_distribution'  : df_logs['risk_level'].value_counts().to_dict(),
            'high_risk_rate'     : float((df_logs['risk_score'] > 0.6).mean()),
        }
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
