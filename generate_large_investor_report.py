#!/usr/bin/env python3
"""
Generate professional PDF report for investors showing ML model results for bank_large
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import numpy as np
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_metrics():
    """Load all metrics from JSON files"""
    reports_dir = 'reports'
    
    metrics = {}
    files = [
        'bank_large_logreg_metrics.json',
        'bank_large_xgboost_metrics.json', 
        'bank_large_logreg_cv5.json',
        'bank_large_xgboost_cv5.json'
    ]
    
    for file in files:
        path = os.path.join(reports_dir, file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                key = file.replace('.json', '').replace('bank_large_', '')
                metrics[key] = json.load(f)
    
    return metrics

def create_confusion_matrix_plot(metrics, model_name, cv=False):
    """Create confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = np.array(metrics['confusion_matrix'])
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Suspicious (1)'],
                yticklabels=['Normal (0)', 'Suspicious (1)'],
                ax=ax)
    
    ax.set_title(f'{model_name} Confusion Matrix {"(5-Fold CV)" if cv else "(Train/Test)"}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'confusion_matrix_{model_name}_{"cv" if cv else "test"}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_roc_comparison_plot(metrics):
    """Create ROC AUC comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Logistic Regression', 'XGBoost']
    test_aucs = [
        metrics['logreg_metrics']['roc_auc'],
        metrics['xgboost_metrics']['roc_auc']
    ]
    cv_aucs = [
        metrics['logreg_cv5']['roc_auc'],
        metrics['xgboost_cv5']['roc_auc']
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_aucs, width, label='Train/Test Split', alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_aucs, width, label='5-Fold Cross-Validation', alpha=0.8)
    
    ax.set_ylabel('ROC AUC Score', fontsize=12)
    ax.set_title('Model Performance Comparison - Bank Large Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0.6, 1.0)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'roc_comparison.png'

def create_precision_recall_plot(metrics):
    """Create precision-recall comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test results
    logreg_test = metrics['logreg_metrics']['classification_report']['1']
    xgb_test = metrics['xgboost_metrics']['classification_report']['1']
    
    # CV results  
    logreg_cv = metrics['logreg_cv5']['classification_report']['1']
    xgb_cv = metrics['xgboost_cv5']['classification_report']['1']
    
    categories = ['Precision', 'Recall', 'F1-Score']
    
    # Test results
    logreg_test_scores = [logreg_test['precision'], logreg_test['recall'], logreg_test['f1-score']]
    xgb_test_scores = [xgb_test['precision'], xgb_test['recall'], xgb_test['f1-score']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, logreg_test_scores, width, label='Logistic Regression', alpha=0.8)
    ax1.bar(x + width/2, xgb_test_scores, width, label='XGBoost', alpha=0.8)
    ax1.set_title('Train/Test Split Results', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # CV results
    logreg_cv_scores = [logreg_cv['precision'], logreg_cv['recall'], logreg_cv['f1-score']]
    xgb_cv_scores = [xgb_cv['precision'], xgb_cv['recall'], xgb_cv['f1-score']]
    
    ax2.bar(x - width/2, logreg_cv_scores, width, label='Logistic Regression', alpha=0.8)
    ax2.bar(x + width/2, xgb_cv_scores, width, label='XGBoost', alpha=0.8)
    ax2.set_title('5-Fold Cross-Validation Results', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.suptitle('Precision, Recall, and F1-Score Comparison - Bank Large', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'precision_recall_comparison.png'

def create_dataset_comparison_plot():
    """Create comparison chart across all three datasets"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    datasets = ['Bank Small', 'Bank Medium', 'Bank Large']
    total_transactions = [4993, 15482, 23428]
    suspicious_counts = [62, 89, 56]
    suspicious_percentages = [1.24, 0.57, 0.24]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    bars1 = ax.bar(x - width, total_transactions, width, label='Total Transactions', alpha=0.8)
    bars2 = ax.bar(x, suspicious_counts, width, label='Suspicious Transactions', alpha=0.8)
    
    ax2 = ax.twinx()
    bars3 = ax2.bar(x + width, suspicious_percentages, width, label='Suspicious %', alpha=0.8, color='red')
    
    ax.set_ylabel('Transaction Count', fontsize=12)
    ax2.set_ylabel('Suspicious Percentage (%)', fontsize=12)
    ax.set_title('Dataset Comparison Across Bank Small, Medium, and Large', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'dataset_comparison.png'

def generate_pdf_report():
    """Generate the complete PDF report"""
    # Load metrics
    metrics = load_metrics()
    
    # Create visualizations
    print("Creating visualizations...")
    cm_logreg_test = create_confusion_matrix_plot(metrics['logreg_metrics'], 'Logistic Regression', cv=False)
    cm_xgb_test = create_confusion_matrix_plot(metrics['xgboost_metrics'], 'XGBoost', cv=False)
    cm_logreg_cv = create_confusion_matrix_plot(metrics['logreg_cv5'], 'Logistic Regression', cv=True)
    cm_xgb_cv = create_confusion_matrix_plot(metrics['xgboost_cv5'], 'XGBoost', cv=True)
    roc_plot = create_roc_comparison_plot(metrics)
    pr_plot = create_precision_recall_plot(metrics)
    dataset_plot = create_dataset_comparison_plot()
    
    # Create PDF
    doc = SimpleDocTemplate("Bank_Large_ML_Model_Results_Report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Build PDF content
    story = []
    
    # Title page
    story.append(Paragraph("Bank Transaction Fraud Detection", title_style))
    story.append(Paragraph("Machine Learning Model Performance Report - Large Dataset", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_text = """
    This report presents the results of machine learning models developed to detect suspicious 
    transactions in the Bank Large dataset. This represents the most challenging production-scale 
    scenario with only 0.24% suspicious transactions across 23,428 total transactions. Two models 
    were evaluated: Logistic Regression and XGBoost. The severe class imbalance presents significant 
    challenges, with Logistic Regression showing better resilience (CV AUC: 0.911) compared to 
    XGBoost (CV AUC: 0.785). This analysis provides critical insights for production deployment 
    in realistic banking environments.
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 20))
    
    # Dataset Comparison
    story.append(Paragraph("Dataset Scale Comparison", heading_style))
    story.append(Image(dataset_plot, width=7.2*inch, height=3.6*inch))
    story.append(Spacer(1, 20))
    
    # Dataset Information
    story.append(Paragraph("Dataset Information", heading_style))
    dataset_text = """
    <b>Dataset:</b> Bank Large Dataset<br/>
    <b>Total Transactions:</b> 23,428<br/>
    <b>Suspicious Transactions:</b> 56 (0.24%)<br/>
    <b>Normal Transactions:</b> 23,372 (99.76%)<br/>
    <b>Features:</b> Transaction amount, account activity patterns, geographic data, 
    transaction types, and account metadata<br/>
    <b>Evaluation Method:</b> Train/Test Split (80/20) + 5-Fold Cross-Validation<br/>
    <b>Class Imbalance:</b> Most severe across all datasets (0.24%)<br/>
    <b>Production Scale:</b> Represents realistic banking environment
    """
    story.append(Paragraph(dataset_text, body_style))
    story.append(Spacer(1, 20))
    
    # Model Parameters
    story.append(Paragraph("Model Parameters", heading_style))
    
    # Logistic Regression parameters
    logreg_params = """
    <b>Logistic Regression:</b><br/>
    • Max iterations: 500 (increased for large dataset)<br/>
    • Class weight: balanced<br/>
    • Preprocessing: StandardScaler, OneHotEncoder<br/>
    • Regularization: L2 (default)
    """
    story.append(Paragraph(logreg_params, body_style))
    
    # XGBoost parameters
    xgb_params = """
    <b>XGBoost:</b><br/>
    • N estimators: 500 (increased for large dataset)<br/>
    • Max depth: 8 (increased depth)<br/>
    • Learning rate: 0.05 (reduced for stability)<br/>
    • Subsample: 0.8<br/>
    • Colsample by tree: 0.8<br/>
    • Lambda regularization: 1.0<br/>
    • Tree method: hist
    """
    story.append(Paragraph(xgb_params, body_style))
    story.append(Spacer(1, 20))
    
    # Performance Results
    story.append(Paragraph("Performance Results", heading_style))
    
    # Create results table
    results_data = [
        ['Model', 'Method', 'ROC AUC', 'Precision', 'Recall', 'F1-Score'],
        ['Logistic Regression', 'Train/Test', f"{metrics['logreg_metrics']['roc_auc']:.4f}", 
         f"{metrics['logreg_metrics']['classification_report']['1']['precision']:.3f}",
         f"{metrics['logreg_metrics']['classification_report']['1']['recall']:.3f}",
         f"{metrics['logreg_metrics']['classification_report']['1']['f1-score']:.3f}"],
        ['Logistic Regression', '5-Fold CV', f"{metrics['logreg_cv5']['roc_auc']:.4f}",
         f"{metrics['logreg_cv5']['classification_report']['1']['precision']:.3f}",
         f"{metrics['logreg_cv5']['classification_report']['1']['recall']:.3f}",
         f"{metrics['logreg_cv5']['classification_report']['1']['f1-score']:.3f}"],
        ['XGBoost', 'Train/Test', f"{metrics['xgboost_metrics']['roc_auc']:.4f}",
         f"{metrics['xgboost_metrics']['classification_report']['1']['precision']:.3f}",
         f"{metrics['xgboost_metrics']['classification_report']['1']['recall']:.3f}",
         f"{metrics['xgboost_metrics']['classification_report']['1']['f1-score']:.3f}"],
        ['XGBoost', '5-Fold CV', f"{metrics['xgboost_cv5']['roc_auc']:.4f}",
         f"{metrics['xgboost_cv5']['classification_report']['1']['precision']:.3f}",
         f"{metrics['xgboost_cv5']['classification_report']['1']['recall']:.3f}",
         f"{metrics['xgboost_cv5']['classification_report']['1']['f1-score']:.3f}"]
    ]
    
    results_table = Table(results_data)
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # ROC Comparison Chart
    story.append(Paragraph("Model Performance Comparison", heading_style))
    story.append(Image(roc_plot, width=6*inch, height=3.6*inch))
    story.append(Spacer(1, 20))
    
    # Precision-Recall Chart
    story.append(Paragraph("Precision-Recall Analysis", heading_style))
    story.append(Image(pr_plot, width=7.5*inch, height=3*inch))
    story.append(Spacer(1, 20))
    
    # Confusion Matrices
    story.append(Paragraph("Confusion Matrices", heading_style))
    
    # Train/Test confusion matrices
    story.append(Paragraph("Train/Test Split Results", styles['Heading3']))
    story.append(Image(cm_logreg_test, width=4*inch, height=3*inch))
    story.append(Spacer(1, 10))
    story.append(Image(cm_xgb_test, width=4*inch, height=3*inch))
    story.append(Spacer(1, 20))
    
    # CV confusion matrices
    story.append(Paragraph("5-Fold Cross-Validation Results", styles['Heading3']))
    story.append(Image(cm_logreg_cv, width=4*inch, height=3*inch))
    story.append(Spacer(1, 10))
    story.append(Image(cm_xgb_cv, width=4*inch, height=3*inch))
    story.append(Spacer(1, 20))
    
    # Detailed Anomaly Analysis
    story.append(Paragraph("Detailed Anomaly Analysis", heading_style))
    
    # Anomaly Pattern Breakdown
    story.append(Paragraph("Suspicious Transaction Patterns", styles['Heading3']))
    pattern_text = """
    <b>Pattern Type Breakdown:</b><br/>
    • Fan-in Pattern: 22 transactions (39.3%) - Multiple accounts sending money to same destination<br/>
    • Cycle Pattern: 34 transactions (60.7%) - Circular money flow patterns<br/><br/>
    
    <b>Top Destination Accounts (Fan-in Pattern):</b><br/>
    • Account 43: 3 suspicious incoming transactions<br/>
    • Account 40: 3 suspicious incoming transactions<br/>
    • Account 77: 3 suspicious incoming transactions<br/>
    • Account 47: 2 suspicious incoming transactions<br/>
    • Account 56: 2 suspicious incoming transactions
    """
    story.append(Paragraph(pattern_text, body_style))
    story.append(Spacer(1, 15))
    
    # Geographic and Temporal Analysis
    story.append(Paragraph("Geographic and Temporal Distribution", styles['Heading3']))
    geo_temporal_text = """
    <b>Geographic Spread:</b><br/>
    Suspicious accounts distributed across multiple states:<br/>
    • Marshall Islands (MH) - North Brian<br/>
    • North Carolina (NC) - West Jacquelinemouth<br/>
    • Arizona (AZ) - Osbornetown<br/>
    • American Samoa (AS) - Herringstad<br/>
    • Mississippi (MS) - East Jasminfort<br/><br/>
    
    <b>Temporal Patterns:</b><br/>
    • Peak Activity: January 4, 2017 (4 suspicious transactions)<br/>
    • Distribution: 1-4 transactions per day throughout 2017<br/>
    • Strategy: Temporal spreading to avoid bulk detection<br/><br/>
    
    <b>Bank Distribution:</b><br/>
    • Bank A: All suspicious accounts<br/>
    • Concentrated bank activity pattern
    """
    story.append(Paragraph(geo_temporal_text, body_style))
    story.append(Spacer(1, 15))
    
    # Transaction Amount Analysis
    story.append(Paragraph("Transaction Amount Analysis", styles['Heading3']))
    amount_text = """
    <b>Suspicious Transaction Amounts:</b><br/>
    • Range: $60.83 - $1,955.28<br/>
    • Mean: $919.68<br/>
    • Median: $845.62<br/><br/>
    
    <b>Normal Transaction Amounts (for comparison):</b><br/>
    • Range: $0.18 - $1,999.90<br/>
    • Mean: $913.33<br/>
    • Median: $858.37<br/><br/>
    
    <b>Key Insight:</b> Suspicious amounts are very similar to normal transactions, 
    indicating highly sophisticated evasion tactics to avoid detection thresholds.
    """
    story.append(Paragraph(amount_text, body_style))
    story.append(Spacer(1, 15))
    
    # Account Characteristics
    story.append(Paragraph("Suspicious Account Characteristics", styles['Heading3']))
    account_text = """
    <b>Account Profile:</b><br/>
    • Total SAR Accounts: 68 flagged accounts<br/>
    • Account Type: 100% Individual accounts<br/>
    • Prior SAR History: All accounts have previous suspicious activity flags<br/><br/>
    
    <b>Account Naming Pattern:</b><br/>
    • Generic customer names (C_XXX format)<br/>
    • Systematic naming convention suggests coordinated activity<br/><br/>
    
    <b>Branch Distribution:</b><br/>
    • All accounts in Branch 1<br/>
    • Highly concentrated branch activity pattern<br/><br/>
    
    <b>Risk Indicators:</b><br/>
    • Prior SAR flags: 100% correlation with previous suspicious activity<br/>
    • Geographic dispersion: Coordinated activity across multiple locations<br/>
    • Amount patterns: Nearly identical to normal transactions<br/>
    • Temporal spreading: Activity distributed to avoid detection
    """
    story.append(Paragraph(account_text, body_style))
    story.append(Spacer(1, 20))
    
    # Key Findings
    story.append(Paragraph("Key Findings", heading_style))
    findings_text = """
    <b>1. Extreme Class Imbalance Challenge:</b> The large dataset presents the most realistic 
    production scenario with only 0.24% suspicious transactions, representing the ultimate 
    challenge for ML models in banking environments.<br/><br/>
    
    <b>2. Model Performance Under Severe Imbalance:</b><br/>
    • <b>Logistic Regression:</b> Demonstrates resilience with CV AUC of 0.911, showing 
    better adaptation to severe class imbalance<br/>
    • <b>XGBoost:</b> Struggles significantly with CV AUC of 0.785, highlighting the 
    challenges of tree-based methods in extreme imbalance scenarios<br/><br/>
    
    <b>3. Sophisticated Anomaly Patterns Detected:</b><br/>
    • <b>Fan-in Pattern (39.3%):</b> Multiple accounts funneling money to specific destinations<br/>
    • <b>Cycle Pattern (60.7%):</b> Circular money flows to obscure transaction trails<br/>
    • <b>Geographic Coordination:</b> Suspicious activity across multiple states and territories<br/>
    • <b>Temporal Evasion:</b> Activity spread across time to avoid bulk detection<br/><br/>
    
    <b>4. High-Risk Account Identification:</b><br/>
    • 68 SAR-flagged accounts with 100% prior suspicious activity history<br/>
    • Top destination accounts receiving 2-3 suspicious transactions each<br/>
    • Systematic naming patterns suggesting coordinated criminal activity<br/><br/>
    
    <b>5. Production-Scale Insights:</b><br/>
    • Represents realistic banking environment with 23K+ transactions<br/>
    • Demonstrates the challenges of real-world fraud detection<br/>
    • Highlights the need for specialized approaches in production systems<br/><br/>
    
    <b>6. Amount Evasion Sophistication:</b> Suspicious transactions are nearly identical 
    to normal transactions in amount distribution, indicating highly sophisticated 
    evasion tactics.
    """
    story.append(Paragraph(findings_text, body_style))
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    recommendations_text = """
    <b>1. Model Selection for Production:</b><br/>
    • Prioritize <b>Logistic Regression</b> for severe class imbalance scenarios<br/>
    • Consider ensemble methods combining multiple approaches<br/>
    • Implement specialized anomaly detection techniques<br/><br/>
    
    <b>2. Enhanced Monitoring for High-Risk Accounts:</b><br/>
    • Implement real-time monitoring for top destination accounts (43, 40, 77, 47, 56)<br/>
    • Flag all accounts with prior SAR history for enhanced scrutiny<br/>
    • Monitor concentrated branch activity patterns<br/><br/>
    
    <b>3. Advanced Techniques for Extreme Imbalance:</b><br/>
    • Implement SMOTE, ADASYN, or other advanced oversampling techniques<br/>
    • Use focal loss, class-weighted approaches, or cost-sensitive learning<br/>
    • Consider anomaly detection methods (Isolation Forest, One-Class SVM)<br/>
    • Explore deep learning approaches with specialized architectures<br/><br/>
    
    <b>4. Pattern-Based Detection Rules:</b><br/>
    • Develop specific rules for fan-in pattern detection (multiple → single destination)<br/>
    • Implement cycle detection algorithms for circular money flows<br/>
    • Monitor geographic clustering of suspicious activity<br/>
    • Track temporal patterns and burst detection<br/><br/>
    
    <b>5. Production Deployment Strategy:</b><br/>
    • Implement robust monitoring and alerting systems<br/>
    • Use ensemble methods combining multiple approaches<br/>
    • Regular model retraining with updated data<br/>
    • Implement feedback loops for continuous improvement<br/><br/>
    
    <b>6. Regulatory Compliance:</b><br/>
    • Ensure models meet regulatory requirements for suspicious activity reporting<br/>
    • Implement audit trails and explainability features<br/>
    • Regular validation and testing protocols
    """
    story.append(Paragraph(recommendations_text, body_style))
    story.append(Spacer(1, 20))
    
    # Conclusion
    story.append(Paragraph("Conclusion", heading_style))
    conclusion_text = """
    The Bank Large dataset represents the ultimate test of machine learning models in realistic 
    production banking environments. With only 0.24% suspicious transactions across 23,428 
    total transactions, this analysis reveals the severe challenges of real-world fraud detection. 
    Logistic Regression demonstrates superior resilience to extreme class imbalance, while 
    XGBoost requires significant optimization for such scenarios. The sophisticated criminal 
    patterns detected, including fan-in and cycle structures with geographic coordination, 
    provide critical insights for production deployment. The identification of 68 high-risk 
    accounts with systematic patterns offers actionable intelligence for enhanced monitoring 
    and regulatory compliance. This analysis underscores the critical need for specialized 
    approaches, advanced techniques, and robust production strategies in modern banking 
    fraud detection systems.
    """
    story.append(Paragraph(conclusion_text, body_style))
    
    # Build PDF
    print("Generating PDF report...")
    doc.build(story)
    print("PDF report generated: Bank_Large_ML_Model_Results_Report.pdf")
    
    # Clean up temporary image files
    temp_files = [cm_logreg_test, cm_xgb_test, cm_logreg_cv, cm_xgb_cv, roc_plot, pr_plot, dataset_plot]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
    print("Temporary files cleaned up.")

if __name__ == "__main__":
    generate_pdf_report()
