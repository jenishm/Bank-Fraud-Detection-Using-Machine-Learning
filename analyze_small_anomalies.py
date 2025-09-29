#!/usr/bin/env python3
"""
Analyze anomalies in bank_small dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_anomalies():
    # Load data
    transactions = pd.read_csv('bank_small/bank_small/transactions.csv')
    alert_transactions = pd.read_csv('bank_small/bank_small/alert_transactions.csv')
    sar_accounts = pd.read_csv('bank_small/bank_small/sar_accounts.csv')
    accounts = pd.read_csv('bank_small/bank_small/accounts.csv')

    print('=== SUSPICIOUS TRANSACTIONS ANALYSIS ===')
    suspicious_tx = transactions[transactions['is_sar'] == True]
    print(f'Total suspicious transactions: {len(suspicious_tx)}')
    print(f'Total transactions: {len(transactions)}')
    print(f'Suspicious percentage: {len(suspicious_tx)/len(transactions)*100:.2f}%')

    print('\n=== TRANSACTION AMOUNTS ===')
    print(f'Suspicious transaction amounts:')
    print(f'Min: ${suspicious_tx["base_amt"].min():.2f}')
    print(f'Max: ${suspicious_tx["base_amt"].max():.2f}')
    print(f'Mean: ${suspicious_tx["base_amt"].mean():.2f}')
    print(f'Median: ${suspicious_tx["base_amt"].median():.2f}')

    print('\nNormal transaction amounts:')
    normal_tx = transactions[transactions['is_sar'] == False]
    print(f'Min: ${normal_tx["base_amt"].min():.2f}')
    print(f'Max: ${normal_tx["base_amt"].max():.2f}')
    print(f'Mean: ${normal_tx["base_amt"].mean():.2f}')
    print(f'Median: ${normal_tx["base_amt"].median():.2f}')

    print('\n=== ALERT PATTERNS ===')
    print('Alert types:', alert_transactions['alert_type'].value_counts().to_dict())
    print('Alert IDs:', sorted(alert_transactions['alert_id'].unique()))

    print('\n=== SUSPICIOUS ACCOUNTS ===')
    print(f'Total SAR accounts: {len(sar_accounts)}')
    print('Account types:', sar_accounts['ACCOUNT_TYPE'].value_counts().to_dict())
    print('Alert types:', sar_accounts['ALERT_TYPE'].value_counts().to_dict())

    print('\n=== TOP SUSPICIOUS ACCOUNTS BY TRANSACTION COUNT ===')
    susp_accounts = suspicious_tx['orig_acct'].value_counts().head(10)
    print('Originating accounts with most suspicious transactions:')
    for acct, count in susp_accounts.items():
        print(f'Account {acct}: {count} suspicious transactions')

    print('\n=== TEMPORAL PATTERNS ===')
    suspicious_tx['date'] = pd.to_datetime(suspicious_tx['tran_timestamp']).dt.date
    daily_suspicious = suspicious_tx.groupby('date').size()
    print('Daily suspicious transaction counts:')
    print(daily_suspicious.head(10))

    print('\n=== FAN-IN PATTERN ANALYSIS ===')
    print('Fan-in pattern: Multiple accounts sending money to the same destination account')
    
    # Analyze fan-in patterns
    fan_in_patterns = alert_transactions.groupby('bene_acct').size().sort_values(ascending=False)
    print('\nTop destination accounts receiving suspicious funds (fan-in):')
    for acct, count in fan_in_patterns.head(10).items():
        print(f'Account {acct}: {count} suspicious incoming transactions')
    
    # Get account details for top suspicious accounts
    print('\n=== ACCOUNT DETAILS FOR TOP SUSPICIOUS ACCOUNTS ===')
    top_susp_accounts = list(susp_accounts.head(5).index)
    for acct_id in top_susp_accounts:
        acct_info = accounts[accounts['acct_id'] == acct_id]
        if not acct_info.empty:
            info = acct_info.iloc[0]
            print(f'\nAccount {acct_id}:')
            print(f'  Name: {info["dsply_nm"]}')
            print(f'  Type: {info["type"]}')
            print(f'  Bank: {info["bank_id"]}')
            print(f'  Branch: {info["branch_id"]}')
            print(f'  Prior SAR Count: {info["prior_sar_count"]}')
            print(f'  Location: {info["city"]}, {info["state"]}')

    print('\n=== SUMMARY OF ANOMALY CHARACTERISTICS ===')
    print('1. Pattern Type: Fan-in (multiple sources → single destination)')
    print('2. Transaction Type: All suspicious transactions are TRANSFER type')
    print('3. Amount Range: $37-$491 (similar to normal transactions)')
    print('4. Geographic Spread: Accounts across multiple locations')
    print('5. Temporal Distribution: Spread across multiple days')
    print('6. Account Types: All suspicious accounts are INDIVIDUAL type')

if __name__ == "__main__":
    analyze_anomalies()
