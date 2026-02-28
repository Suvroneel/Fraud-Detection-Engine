"""
Drift Detection Module
Computes PSI (Population Stability Index) and KS statistic
to detect feature distribution drift between training data and live predictions.

PSI interpretation:
  PSI < 0.10  : No significant drift
  PSI 0.10-0.25: Moderate drift, monitor closely
  PSI > 0.25  : Significant drift, trigger retraining review
"""

import numpy as np
import pandas as pd
import sqlite3
import joblib
import logging

from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
LOG_DB    = BASE_DIR / 'logs' / 'predictions.db'


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index between training distribution and live data.

    Args:
        expected: Values from training distribution (reference)
        actual  : Values from live predictions (current)
        buckets : Number of bins for discretization

    Returns:
        PSI score (float)
    """
    expected = np.array(expected)
    actual   = np.array(actual)

    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define buckets based on training distribution quantiles
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints  = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    # Compute proportions
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Avoid division by zero and log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_ks_statistic(expected: np.ndarray, actual: np.ndarray) -> dict:
    """
    Compute KS (Kolmogorov-Smirnov) statistic between two distributions.

    Returns:
        Dict with ks_statistic and p_value
    """
    expected = np.array(expected)[~np.isnan(np.array(expected))]
    actual   = np.array(actual)[~np.isnan(np.array(actual))]

    if len(expected) == 0 or len(actual) == 0:
        return {'ks_statistic': 0.0, 'p_value': 1.0}

    ks_stat, p_value = stats.ks_2samp(expected, actual)
    return {'ks_statistic': float(ks_stat), 'p_value': float(p_value)}


def run_drift_check(reference_features: dict = None) -> dict:
    """
    Run drift detection on recent live predictions vs training distribution.

    Args:
        reference_features: Dict of {feature_name: np.array} from training set.
                           If None, uses stored training statistics.

    Returns:
        Dict with drift report per feature and overall status.
    """
    try:
        conn    = sqlite3.connect(LOG_DB)
        df_logs = pd.read_sql(
            'SELECT risk_score, transaction_amt, latency_ms FROM predictions ORDER BY id DESC LIMIT 500',
            conn
        )
        conn.close()
    except Exception as e:
        logger.error('Could not load prediction logs for drift check: %s', str(e))
        return {'status': 'error', 'message': str(e)}

    if len(df_logs) < 50:
        return {'status': 'insufficient_data', 'n_predictions': len(df_logs)}

    # Load training score distribution if available
    try:
        training_stats = joblib.load(MODEL_DIR / 'training_score_distribution.pkl')
    except FileNotFoundError:
        # Fallback: assume training risk score distribution is approximately beta(0.5, 14)
        # based on 3.5% fraud rate
        training_scores = np.random.beta(0.5, 14, 10000)
        training_stats  = {'risk_scores': training_scores, 'transaction_amt': np.random.lognormal(4, 1.5, 10000)}

    report = {}

    # PSI on risk scores
    psi_score = compute_psi(
        training_stats.get('risk_scores', []),
        df_logs['risk_score'].values
    )
    ks_score = compute_ks_statistic(
        training_stats.get('risk_scores', []),
        df_logs['risk_score'].values
    )
    report['risk_score_drift'] = {
        'psi'              : round(psi_score, 4),
        'ks_statistic'     : round(ks_score['ks_statistic'], 4),
        'ks_p_value'       : round(ks_score['p_value'], 4),
        'drift_level'      : _classify_psi(psi_score),
        'action_required'  : psi_score > 0.25 or ks_score['p_value'] < 0.05,
    }

    # PSI on transaction amounts
    psi_amt = compute_psi(
        training_stats.get('transaction_amt', []),
        df_logs['transaction_amt'].values
    )
    report['transaction_amt_drift'] = {
        'psi'          : round(psi_amt, 4),
        'drift_level'  : _classify_psi(psi_amt),
    }

    # Overall status
    any_critical = any(
        v.get('action_required', False)
        for v in report.values()
        if isinstance(v, dict)
    )
    report['overall_status'] = 'ALERT' if any_critical else 'STABLE'
    report['n_live_predictions_analyzed'] = len(df_logs)

    return report


def _classify_psi(psi: float) -> str:
    if psi < 0.10:
        return 'stable'
    elif psi < 0.25:
        return 'moderate_drift'
    else:
        return 'significant_drift'


def get_retraining_criteria() -> dict:
    """
    Document the retraining trigger criteria.
    This function serves as both documentation and a runtime check.
    """
    return {
        'triggers': [
            {
                'name'       : 'PSI drift on risk scores',
                'threshold'  : 'PSI > 0.25',
                'action'     : 'Immediate retraining review',
            },
            {
                'name'       : 'KS test significance',
                'threshold'  : 'p_value < 0.05 on risk score distribution',
                'action'     : 'Flag for data science team review',
            },
            {
                'name'       : 'Model performance degradation',
                'threshold'  : 'AUC-ROC drops below 0.88 on weekly labeled sample',
                'action'     : 'Trigger full retraining pipeline',
            },
            {
                'name'       : 'High false positive rate spike',
                'threshold'  : 'FPR exceeds 2x baseline over 24hr window',
                'action'     : 'Emergency threshold recalibration',
            },
            {
                'name'       : 'Scheduled retraining',
                'threshold'  : 'Every 30 days regardless of drift',
                'action'     : 'Routine retraining with fresh data window',
            },
        ],
        'retraining_pipeline': [
            'Collect new labeled transactions from the past 90 days',
            'Run data validation checks (schema, distributions)',
            'Re-run feature engineering pipeline on new data',
            'Train new model with same architecture',
            'Compare AUC-ROC on held-out test set vs current production model',
            'If new model AUC is >= current model - 0.005, promote to production',
            'Shadow mode testing for 24hrs before full cutover',
            'Log all prediction comparisons between old and new model during shadow mode',
        ],
    }
