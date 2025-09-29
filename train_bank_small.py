#!/usr/bin/env python3
import os
import json
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

DATA_DIR = os.path.join(os.path.dirname(__file__), 'bank_small', 'bank_small')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing file: {path}')
    return pd.read_csv(path)


def load_data(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = read_csv_safely(os.path.join(data_dir, 'transactions.csv'))
    accounts = read_csv_safely(os.path.join(data_dir, 'accounts.csv'))
    alert_transactions = read_csv_safely(os.path.join(data_dir, 'alert_transactions.csv'))
    return transactions, accounts, alert_transactions


def build_features(transactions: pd.DataFrame, accounts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = transactions.copy()
    df['is_sar'] = df['is_sar'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)
    df['amount'] = pd.to_numeric(df['base_amt'], errors='coerce')

    out_counts = df.groupby('orig_acct')['tran_id'].count().rename('orig_tx_count')
    in_counts = df.groupby('bene_acct')['tran_id'].count().rename('bene_tx_count')

    df = df.merge(out_counts, how='left', left_on='orig_acct', right_index=True)
    df = df.merge(in_counts, how='left', left_on='bene_acct', right_index=True)

    amount_by_orig = df.groupby('orig_acct')['amount']
    amount_by_bene = df.groupby('bene_acct')['amount']
    df = df.merge(amount_by_orig.mean().rename('orig_amt_mean'), left_on='orig_acct', right_index=True, how='left')
    df = df.merge(amount_by_orig.std().rename('orig_amt_std'), left_on='orig_acct', right_index=True, how='left')
    df = df.merge(amount_by_bene.mean().rename('bene_amt_mean'), left_on='bene_acct', right_index=True, how='left')
    df = df.merge(amount_by_bene.std().rename('bene_amt_std'), left_on='bene_acct', right_index=True, how='left')

    acct_cols = ['acct_id', 'acct_rptng_crncy', 'prior_sar_count', 'branch_id', 'bank_id', 'lon', 'lat']
    acc = accounts[acct_cols].copy()
    for c in ['prior_sar_count', 'branch_id', 'lon', 'lat']:
        if c in acc.columns:
            acc[c] = pd.to_numeric(acc[c], errors='coerce')

    df = df.merge(acc.add_prefix('orig_'), left_on='orig_acct', right_on='orig_acct_id', how='left')
    df = df.merge(acc.add_prefix('bene_'), left_on='bene_acct', right_on='bene_acct_id', how='left')

    df['tx_type'] = df['tx_type'].astype(str)
    df['orig_bank_id'] = df['orig_bank_id'].astype(str)
    df['bene_bank_id'] = df['bene_bank_id'].astype(str)
    df['same_bank'] = (df['orig_bank_id'] == df['bene_bank_id']).astype(int)

    feature_cols_num = [
        'amount', 'orig_tx_count', 'bene_tx_count',
        'orig_amt_mean', 'orig_amt_std', 'bene_amt_mean', 'bene_amt_std',
        'orig_prior_sar_count', 'orig_branch_id', 'orig_lon', 'orig_lat',
        'bene_prior_sar_count', 'bene_branch_id', 'bene_lon', 'bene_lat',
        'same_bank'
    ]
    feature_cols_cat = ['tx_type', 'orig_acct_rptng_crncy', 'bene_acct_rptng_crncy']

    y = df['is_sar']
    X = df[feature_cols_num + feature_cols_cat].copy()

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor


def _logreg_pipeline(X: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X)
    model = LogisticRegression(max_iter=200, class_weight='balanced')
    return Pipeline(steps=[('pre', pre), ('clf', model)])


def _xgb_pipeline(X: pd.DataFrame) -> Pipeline:
    if not _HAS_XGB:
        raise RuntimeError('xgboost not installed')
    pre = build_preprocessor(X)
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='binary:logistic',
        tree_method='hist',
        random_state=42,
        eval_metric='auc'
    )
    return Pipeline(steps=[('pre', pre), ('clf', clf)])


def train_logistic_regression(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    pipe = _logreg_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'average_precision': float(average_precision_score(y_test, y_proba)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return pipe, metrics


def train_xgboost(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    if not _HAS_XGB:
        return None, {'error': 'xgboost not installed'}

    pipe = _xgb_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'average_precision': float(average_precision_score(y_test, y_proba)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return pipe, metrics


def cross_validate_model(model_name: str, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    if model_name == 'logreg':
        estimator = _logreg_pipeline(X)
    elif model_name == 'xgboost':
        if not _HAS_XGB:
            return {'error': 'xgboost not installed'}
        estimator = _xgb_pipeline(X)
    else:
        raise ValueError('Unknown model for CV')

    # cross_val_predict will clone and fit per fold
    y_proba = cross_val_predict(estimator, X, y, cv=skf, method='predict_proba')[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        'roc_auc': float(roc_auc_score(y, y_proba)),
        'average_precision': float(average_precision_score(y, y_proba)),
        'classification_report': classification_report(y, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist()
    }
    return metrics


def save_artifacts(model_name: str, pipe, metrics: dict):
    model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    metrics_path = os.path.join(REPORTS_DIR, f'{model_name}_metrics.json')

    if pipe is not None:
        joblib.dump(pipe, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--cv', type=int, default=0, help='Run Stratified K-fold CV with given folds (0=off)')
    args = parser.parse_args()

    transactions, accounts, _ = load_data(args.data_dir)
    X, y = build_features(transactions, accounts)

    # Standard train/test
    logreg_model, logreg_metrics = train_logistic_regression(X, y)
    save_artifacts('bank_small_logreg', logreg_model, logreg_metrics)

    xgb_model, xgb_metrics = train_xgboost(X, y)
    save_artifacts('bank_small_xgboost', xgb_model, xgb_metrics)

    print('LogReg AUC:', logreg_metrics.get('roc_auc'))
    print('XGBoost AUC:', xgb_metrics.get('roc_auc'))

    # Optional cross-validation
    if args.cv and args.cv > 1:
        logreg_cv = cross_validate_model('logreg', X, y, n_splits=args.cv)
        with open(os.path.join(REPORTS_DIR, f'bank_small_logreg_cv{args.cv}.json'), 'w') as f:
            json.dump(logreg_cv, f, indent=2)
        print(f'LogReg {args.cv}-fold CV AUC:', logreg_cv.get('roc_auc'))

        xgb_cv = cross_validate_model('xgboost', X, y, n_splits=args.cv)
        with open(os.path.join(REPORTS_DIR, f'bank_small_xgboost_cv{args.cv}.json'), 'w') as f:
            json.dump(xgb_cv, f, indent=2)
        print(f'XGBoost {args.cv}-fold CV AUC:', xgb_cv.get('roc_auc'))


if __name__ == '__main__':
    main()
