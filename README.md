# Bank Fraud Detection - Local, Federated, and Reporting

This repository contains datasets and code to:
- Train and evaluate fraud detection models on three bank datasets (small, medium, large)
- Run federated learning across the three banks (privacy-preserving, parameter aggregation)
- Generate investor-ready PDF reports and a comprehensive cross-bank report

## 1) Quickstart

### Prerequisites
- Python 3.10+ (tested on 3.13)
- macOS users (for XGBoost): `brew install libomp`
- Git (for source control)

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If on macOS for XGBoost runtime:
export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:${DYLD_LIBRARY_PATH:-}"
```

## 2) Train Models per Dataset

Run on bank_small:
```bash
python train_bank_small.py --cv 5
```

Run on bank_medium:
```bash
python train_bank_medium.py --cv 5
```

Run on bank_large:
```bash
python train_bank_large.py --cv 5
```

- Metrics JSONs are saved in `reports/`
- Trained models are saved in `models/`

## 3) Generate Reports

- Small dataset investor report:
```bash
python generate_investor_report.py
```
- Medium dataset investor report:
```bash
python generate_medium_investor_report.py
```
- Large dataset investor report:
```bash
python generate_large_investor_report.py
```
- Federated learning report:
```bash
python generate_federated_report.py
```
- Comprehensive cross-bank report:
```bash
python generate_comprehensive_report.py
```

PDFs will be generated in the project root.

## 4) Federated Learning (Simple FedAvg)

Run federated learning across Small/Medium/Large banks:
```bash
python federated_learning.py
```
This will:
- Train local models on each bank dataset
- Aggregate parameters (weighted by dataset size)
- Evaluate the global model vs local models
- Save `federated_results.json` and generate a federated PDF report (see above)

## 5) Datasets

- `bank_small/bank_small`: ~5k transactions, ~1.24% suspicious
- `bank_medium/bank_medium`: ~15k transactions, ~0.57% suspicious
- `bank_large`: ~23k transactions, ~0.24% suspicious

Each bank folder includes:
- `transactions.csv`, `accounts.csv`, `alert_transactions.csv`, `sar_accounts.csv`, logs, and related CSVs.

## 6) Notes & Troubleshooting
- macOS + XGBoost: ensure `libomp` is installed and `DYLD_LIBRARY_PATH` set (see setup).
- If PDFs fail to render icons, it’s harmless (font glyph warnings).
- Large datasets and reports may take a few minutes to run/generate.

## 7) License
This project is for demonstration and evaluation purposes.
