#!/usr/bin/env python3
"""
Simple Federated Learning Implementation for Bank Fraud Detection
Trains models locally on each bank dataset and aggregates them into a global model
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

class FederatedBankModel:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.local_models = {}
        self.global_model = None
        self.bank_datasets = {
            'bank_small': 'bank_small/bank_small',
            'bank_medium': 'bank_medium/bank_medium', 
            'bank_large': 'bank_large'
        }
        self.results = {}
        
    def load_bank_data(self, bank_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data for a specific bank"""
        data_dir = self.bank_datasets[bank_name]
        
        transactions = pd.read_csv(os.path.join(data_dir, 'transactions.csv'))
        accounts = pd.read_csv(os.path.join(data_dir, 'accounts.csv'))
        
        # Build features (same as previous implementations)
        df = transactions.copy()
        df['is_sar'] = df['is_sar'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)
        df['amount'] = pd.to_numeric(df['base_amt'], errors='coerce')

        # Account degree features
        out_counts = df.groupby('orig_acct')['tran_id'].count().rename('orig_tx_count')
        in_counts = df.groupby('bene_acct')['tran_id'].count().rename('bene_tx_count')

        df = df.merge(out_counts, how='left', left_on='orig_acct', right_index=True)
        df = df.merge(in_counts, how='left', left_on='bene_acct', right_index=True)

        # Amount statistics
        amount_by_orig = df.groupby('orig_acct')['amount']
        amount_by_bene = df.groupby('bene_acct')['amount']
        df = df.merge(amount_by_orig.mean().rename('orig_amt_mean'), left_on='orig_acct', right_index=True, how='left')
        df = df.merge(amount_by_orig.std().rename('orig_amt_std'), left_on='orig_acct', right_index=True, how='left')
        df = df.merge(amount_by_bene.mean().rename('bene_amt_mean'), left_on='bene_acct', right_index=True, how='left')
        df = df.merge(amount_by_bene.std().rename('bene_amt_std'), left_on='bene_acct', right_index=True, how='left')

        # Account features
        acct_cols = ['acct_id', 'acct_rptng_crncy', 'prior_sar_count', 'branch_id', 'bank_id', 'lon', 'lat']
        acc = accounts[acct_cols].copy()
        for c in ['prior_sar_count', 'branch_id', 'lon', 'lat']:
            if c in acc.columns:
                acc[c] = pd.to_numeric(acc[c], errors='coerce')

        df = df.merge(acc.add_prefix('orig_'), left_on='orig_acct', right_on='orig_acct_id', how='left')
        df = df.merge(acc.add_prefix('bene_'), left_on='bene_acct', right_on='bene_acct_id', how='left')

        # Categorical features
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

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline"""
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

    def create_model_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create model pipeline based on model type"""
        preprocessor = self.build_preprocessor(X)
        
        if self.model_type == 'logistic':
            model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
        elif self.model_type == 'xgboost' and _HAS_XGB:
            model = xgb.XGBClassifier(
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
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    def train_local_models(self):
        """Train models locally on each bank dataset"""
        print("=== Training Local Models ===")
        
        for bank_name in self.bank_datasets.keys():
            print(f"\nTraining {bank_name}...")
            
            try:
                X, y = self.load_bank_data(bank_name)
                print(f"  Dataset size: {len(X)} transactions")
                print(f"  Suspicious transactions: {y.sum()} ({y.mean()*100:.2f}%)")
                
                # Create and train model
                model = self.create_model_pipeline(X)
                model.fit(X, y)
                
                # Evaluate on local data
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1]
                
                local_metrics = {
                    'roc_auc': float(roc_auc_score(y, y_proba)),
                    'average_precision': float(average_precision_score(y, y_proba)),
                    'classification_report': classification_report(y, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                    'dataset_size': len(X),
                    'suspicious_count': int(y.sum()),
                    'suspicious_percentage': float(y.mean() * 100)
                }
                
                self.local_models[bank_name] = {
                    'model': model,
                    'metrics': local_metrics,
                    'data_size': len(X)
                }
                
                print(f"  Local ROC AUC: {local_metrics['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"  Error training {bank_name}: {str(e)}")
                continue

    def aggregate_models_federated_averaging(self):
        """Aggregate local models using federated averaging"""
        print("\n=== Federated Model Aggregation ===")
        
        if not self.local_models:
            print("No local models to aggregate!")
            return
            
        # For Logistic Regression, we can average the coefficients
        if self.model_type == 'logistic':
            self._aggregate_logistic_models()
        else:
            print(f"Federated averaging not implemented for {self.model_type}")
            return

    def _aggregate_logistic_models(self):
        """Aggregate Logistic Regression models using weighted averaging"""
        print("Aggregating Logistic Regression models...")
        
        # Get the first model as template
        first_bank = list(self.local_models.keys())[0]
        template_model = self.local_models[first_bank]['model']
        
        # Calculate weights based on dataset size
        total_size = sum(model_info['data_size'] for model_info in self.local_models.values())
        weights = {bank: model_info['data_size'] / total_size 
                 for bank, model_info in self.local_models.items()}
        
        print("Bank weights:", weights)
        
        # Extract coefficients from each model
        coefficients = []
        intercepts = []
        
        for bank_name, model_info in self.local_models.items():
            model = model_info['model']
            classifier = model.named_steps['classifier']
            
            coefficients.append(classifier.coef_[0])
            intercepts.append(classifier.intercept_[0])
        
        # Weighted average of coefficients
        weighted_coef = np.average(coefficients, axis=0, weights=list(weights.values()))
        weighted_intercept = np.average(intercepts, axis=0, weights=list(weights.values()))
        
        # Create global model with averaged parameters
        global_model = template_model
        global_classifier = global_model.named_steps['classifier']
        
        # Set averaged parameters
        global_classifier.coef_ = weighted_coef.reshape(1, -1)
        global_classifier.intercept_ = weighted_intercept
        
        self.global_model = global_model
        print("Global model created with federated averaging")

    def evaluate_global_model(self):
        """Evaluate global model on each bank's data"""
        print("\n=== Global Model Evaluation ===")
        
        if self.global_model is None:
            print("No global model to evaluate!")
            return
            
        for bank_name in self.bank_datasets.keys():
            print(f"\nEvaluating global model on {bank_name}...")
            
            try:
                X, y = self.load_bank_data(bank_name)
                
                # Evaluate global model
                y_pred = self.global_model.predict(X)
                y_proba = self.global_model.predict_proba(X)[:, 1]
                
                global_metrics = {
                    'roc_auc': float(roc_auc_score(y, y_proba)),
                    'average_precision': float(average_precision_score(y, y_proba)),
                    'classification_report': classification_report(y, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y, y_pred).tolist()
                }
                
                # Compare with local model
                local_metrics = self.local_models[bank_name]['metrics']
                
                improvement = global_metrics['roc_auc'] - local_metrics['roc_auc']
                
                print(f"  Local ROC AUC:  {local_metrics['roc_auc']:.4f}")
                print(f"  Global ROC AUC: {global_metrics['roc_auc']:.4f}")
                print(f"  Improvement:    {improvement:+.4f}")
                
                self.results[bank_name] = {
                    'local_metrics': local_metrics,
                    'global_metrics': global_metrics,
                    'improvement': improvement
                }
                
            except Exception as e:
                print(f"  Error evaluating {bank_name}: {str(e)}")
                continue

    def cross_validate_models(self):
        """Perform cross-validation on each bank dataset"""
        print("\n=== Cross-Validation Analysis ===")
        
        for bank_name in self.bank_datasets.keys():
            print(f"\nCross-validation on {bank_name}...")
            
            try:
                X, y = self.load_bank_data(bank_name)
                
                # Local model CV
                local_model = self.create_model_pipeline(X)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                from sklearn.model_selection import cross_val_score
                local_cv_scores = cross_val_score(local_model, X, y, cv=skf, scoring='roc_auc')
                
                print(f"  Local CV AUC: {local_cv_scores.mean():.4f} (+/- {local_cv_scores.std() * 2:.4f})")
                
                # Global model CV (if available)
                if self.global_model is not None:
                    global_cv_scores = cross_val_score(self.global_model, X, y, cv=skf, scoring='roc_auc')
                    print(f"  Global CV AUC: {global_cv_scores.mean():.4f} (+/- {global_cv_scores.std() * 2:.4f})")
                    
                    cv_improvement = global_cv_scores.mean() - local_cv_scores.mean()
                    print(f"  CV Improvement: {cv_improvement:+.4f}")
                
            except Exception as e:
                print(f"  Error in CV for {bank_name}: {str(e)}")
                continue

    def generate_summary_report(self):
        """Generate summary report of federated learning results"""
        print("\n" + "="*60)
        print("FEDERATED LEARNING SUMMARY REPORT")
        print("="*60)
        
        print(f"\nModel Type: {self.model_type.upper()}")
        print(f"Number of Banks: {len(self.local_models)}")
        
        print("\nDataset Overview:")
        for bank_name, model_info in self.local_models.items():
            metrics = model_info['metrics']
            print(f"  {bank_name}:")
            print(f"    Transactions: {metrics['dataset_size']:,}")
            print(f"    Suspicious: {metrics['suspicious_count']} ({metrics['suspicious_percentage']:.2f}%)")
            print(f"    Local AUC: {metrics['roc_auc']:.4f}")
        
        if self.results:
            print("\nFederated Learning Results:")
            total_improvement = 0
            for bank_name, result in self.results.items():
                improvement = result['improvement']
                total_improvement += improvement
                print(f"  {bank_name}:")
                print(f"    Local AUC:  {result['local_metrics']['roc_auc']:.4f}")
                print(f"    Global AUC: {result['global_metrics']['roc_auc']:.4f}")
                print(f"    Change:     {improvement:+.4f}")
            
            avg_improvement = total_improvement / len(self.results)
            print(f"\nAverage Improvement: {avg_improvement:+.4f}")
            
            if avg_improvement > 0:
                print("✅ Federated learning shows positive improvement!")
            else:
                print("❌ Federated learning shows negative impact")
        
        print("\nKey Insights:")
        print("- Federated learning allows banks to collaborate without sharing raw data")
        print("- Global model benefits from diverse patterns across all banks")
        print("- Performance depends on data distribution and model compatibility")
        print("- Privacy-preserving approach for sensitive financial data")

    def save_results(self, filename='federated_results.json'):
        """Save results to JSON file"""
        results_to_save = {
            'model_type': self.model_type,
            'local_models': {},
            'global_results': self.results,
            'summary': {}
        }
        
        # Save local model metrics
        for bank_name, model_info in self.local_models.items():
            results_to_save['local_models'][bank_name] = model_info['metrics']
        
        # Calculate summary statistics
        if self.results:
            improvements = [result['improvement'] for result in self.results.values()]
            results_to_save['summary'] = {
                'average_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'min_improvement': np.min(improvements),
                'max_improvement': np.max(improvements),
                'total_banks': len(self.results)
            }
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to {filename}")

def main():
    print("Federated Learning for Bank Fraud Detection")
    print("="*50)
    
    # Initialize federated learning system
    federated_system = FederatedBankModel(model_type='logistic')
    
    # Train local models on each bank
    federated_system.train_local_models()
    
    # Aggregate models using federated averaging
    federated_system.aggregate_models_federated_averaging()
    
    # Evaluate global model
    federated_system.evaluate_global_model()
    
    # Cross-validation analysis
    federated_system.cross_validate_models()
    
    # Generate summary report
    federated_system.generate_summary_report()
    
    # Save results
    federated_system.save_results()
    
    print("\nFederated learning analysis complete!")

if __name__ == "__main__":
    main()
