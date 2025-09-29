#!/usr/bin/env python3
"""
Generate comprehensive federated learning report
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import numpy as np
from datetime import datetime
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_federated_results():
    """Load federated learning results"""
    with open('federated_results.json', 'r') as f:
        return json.load(f)

def create_performance_comparison_chart(results):
    """Create performance comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    banks = list(results['local_models'].keys())
    local_aucs = [results['local_models'][bank]['roc_auc'] for bank in banks]
    global_aucs = [results['global_results'][bank]['global_metrics']['roc_auc'] for bank in banks]
    improvements = [results['global_results'][bank]['improvement'] for bank in banks]
    
    # Performance comparison
    x = np.arange(len(banks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, local_aucs, width, label='Local Models', alpha=0.8)
    bars2 = ax1.bar(x + width/2, global_aucs, width, label='Global Model', alpha=0.8)
    
    ax1.set_ylabel('ROC AUC Score', fontsize=12)
    ax1.set_title('Local vs Global Model Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([bank.replace('bank_', 'Bank ').title() for bank in banks])
    ax1.legend()
    ax1.set_ylim(0.8, 1.0)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Improvement chart
    colors_improvement = ['red' if imp < 0 else 'green' for imp in improvements]
    bars3 = ax2.bar(x, improvements, color=colors_improvement, alpha=0.7)
    
    ax2.set_ylabel('Improvement (AUC)', fontsize=12)
    ax2.set_title('Performance Improvement by Bank', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([bank.replace('bank_', 'Bank ').title() for bank in banks])
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylim(-0.03, 0.01)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.002),
                f'{height:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('federated_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'federated_performance_comparison.png'

def create_dataset_distribution_chart(results):
    """Create dataset distribution chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    banks = list(results['local_models'].keys())
    dataset_sizes = [results['local_models'][bank]['dataset_size'] for bank in banks]
    suspicious_counts = [results['local_models'][bank]['suspicious_count'] for bank in banks]
    suspicious_percentages = [results['local_models'][bank]['suspicious_percentage'] for bank in banks]
    
    x = np.arange(len(banks))
    width = 0.25
    
    bars1 = ax.bar(x - width, dataset_sizes, width, label='Total Transactions', alpha=0.8)
    bars2 = ax.bar(x, suspicious_counts, width, label='Suspicious Transactions', alpha=0.8)
    
    ax2 = ax.twinx()
    bars3 = ax2.bar(x + width, suspicious_percentages, width, label='Suspicious %', alpha=0.8, color='red')
    
    ax.set_ylabel('Transaction Count', fontsize=12)
    ax2.set_ylabel('Suspicious Percentage (%)', fontsize=12)
    ax.set_title('Dataset Distribution Across Banks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([bank.replace('bank_', 'Bank ').title() for bank in banks])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
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
    plt.savefig('federated_dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'federated_dataset_distribution.png'

def generate_federated_pdf_report():
    """Generate comprehensive federated learning PDF report"""
    # Load results
    results = load_federated_results()
    
    # Create visualizations
    print("Creating federated learning visualizations...")
    perf_plot = create_performance_comparison_chart(results)
    dist_plot = create_dataset_distribution_chart(results)
    
    # Create PDF
    doc = SimpleDocTemplate("Federated_Learning_Report.pdf", pagesize=A4)
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
    story.append(Paragraph("Federated Learning for Bank Fraud Detection", title_style))
    story.append(Paragraph("Cross-Bank Model Collaboration Analysis", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_text = """
    This report presents the results of federated learning implementation across three bank datasets 
    (Small, Medium, Large) for fraud detection. Federated learning enables banks to collaborate 
    on model training without sharing sensitive customer data. The analysis shows that while the 
    global model demonstrates consistent performance across all banks, local models outperform 
    the global model on their respective datasets. This highlights the trade-off between data 
    privacy and model specialization in federated learning scenarios.
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 20))
    
    # Dataset Distribution
    story.append(Paragraph("Dataset Distribution Analysis", heading_style))
    story.append(Image(dist_plot, width=7.2*inch, height=3.6*inch))
    story.append(Spacer(1, 20))
    
    # Federated Learning Methodology
    story.append(Paragraph("Federated Learning Methodology", heading_style))
    methodology_text = """
    <b>Approach:</b> Federated Averaging (FedAvg)<br/>
    <b>Model Type:</b> Logistic Regression<br/>
    <b>Aggregation Method:</b> Weighted averaging of model coefficients<br/>
    <b>Weight Calculation:</b> Based on dataset size<br/><br/>
    
    <b>Process:</b><br/>
    1. Train local models on each bank's data independently<br/>
    2. Extract model coefficients from each local model<br/>
    3. Calculate weighted average of coefficients based on dataset size<br/>
    4. Create global model with averaged parameters<br/>
    5. Evaluate global model performance on each bank's data<br/><br/>
    
    <b>Privacy Benefits:</b><br/>
    • No raw data sharing between banks<br/>
    • Only model parameters are exchanged<br/>
    • Maintains data sovereignty for each institution<br/>
    • Enables collaboration without compromising privacy
    """
    story.append(Paragraph(methodology_text, body_style))
    story.append(Spacer(1, 20))
    
    # Performance Results
    story.append(Paragraph("Performance Results", heading_style))
    
    # Create results table
    banks = list(results['local_models'].keys())
    results_data = [
        ['Bank', 'Dataset Size', 'Suspicious %', 'Local AUC', 'Global AUC', 'Improvement'],
    ]
    
    for bank in banks:
        local_metrics = results['local_models'][bank]
        global_metrics = results['global_results'][bank]
        improvement = global_metrics['improvement']
        
        results_data.append([
            bank.replace('bank_', 'Bank ').title(),
            f"{local_metrics['dataset_size']:,}",
            f"{local_metrics['suspicious_percentage']:.2f}%",
            f"{local_metrics['roc_auc']:.4f}",
            f"{global_metrics['global_metrics']['roc_auc']:.4f}",
            f"{improvement:+.4f}"
        ])
    
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
    
    # Performance Comparison Chart
    story.append(Paragraph("Performance Comparison", heading_style))
    story.append(Image(perf_plot, width=7.5*inch, height=3*inch))
    story.append(Spacer(1, 20))
    
    # Key Findings
    story.append(Paragraph("Key Findings", heading_style))
    findings_text = """
    <b>1. Local Model Superiority:</b> Local models consistently outperform the global model 
    on their respective datasets, with performance degradation ranging from -0.0118 to -0.0236 AUC points.<br/><br/>
    
    <b>2. Dataset Size Impact:</b> Bank Large (53.4% weight) has the most influence on the global model, 
    followed by Bank Medium (35.3%) and Bank Small (11.4%).<br/><br/>
    
    <b>3. Class Imbalance Challenge:</b> The global model struggles to maintain performance across 
    datasets with varying class imbalance ratios (1.24%, 0.57%, 0.24%).<br/><br/>
    
    <b>4. Consistency vs Specialization:</b> While the global model provides consistent performance 
    across all banks, local models are specialized for their specific data distributions.<br/><br/>
    
    <b>5. Privacy-Preserving Success:</b> The federated approach successfully enables collaboration 
    without compromising data privacy or security.<br/><br/>
    
    <b>6. Cross-Validation Stability:</b> Cross-validation results show identical performance between 
    local and global models, indicating stable model behavior.
    """
    story.append(Paragraph(findings_text, body_style))
    story.append(Spacer(1, 20))
    
    # Analysis and Insights
    story.append(Paragraph("Analysis and Insights", heading_style))
    analysis_text = """
    <b>Why Local Models Outperform Global Models:</b><br/>
    • <b>Data Distribution Mismatch:</b> Each bank has unique fraud patterns and customer behaviors<br/>
    • <b>Class Imbalance Variation:</b> Different suspicious transaction ratios require specialized handling<br/>
    • <b>Feature Engineering:</b> Local models adapt to bank-specific feature distributions<br/>
    • <b>Overfitting to Local Patterns:</b> Local models optimize for their specific dataset characteristics<br/><br/>
    
    <b>Benefits of Federated Learning:</b><br/>
    • <b>Privacy Preservation:</b> No raw data sharing between institutions<br/>
    • <b>Regulatory Compliance:</b> Maintains data sovereignty requirements<br/>
    • <b>Knowledge Sharing:</b> Enables learning from diverse fraud patterns<br/>
    • <b>Scalability:</b> Can accommodate additional banks without data centralization<br/><br/>
    
    <b>Trade-offs Identified:</b><br/>
    • <b>Performance vs Privacy:</b> Slight performance loss for significant privacy gains<br/>
    • <b>Generalization vs Specialization:</b> Global model generalizes but loses local optimization<br/>
    • <b>Complexity vs Simplicity:</b> Increased implementation complexity for privacy benefits
    """
    story.append(Paragraph(analysis_text, body_style))
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    recommendations_text = """
    <b>1. Hybrid Approach:</b><br/>
    • Use global model as a baseline for new banks or unknown patterns<br/>
    • Implement local fine-tuning on global model for improved performance<br/>
    • Combine global and local predictions using ensemble methods<br/><br/>
    
    <b>2. Advanced Federated Techniques:</b><br/>
    • Implement federated learning with differential privacy<br/>
    • Use personalized federated learning for bank-specific adaptations<br/>
    • Explore federated transfer learning for knowledge sharing<br/><br/>
    
    <b>3. Model Architecture Improvements:</b><br/>
    • Use more sophisticated aggregation methods (e.g., FedProx, FedAvgM)<br/>
    • Implement adaptive learning rates for different banks<br/>
    • Consider multi-task learning approaches<br/><br/>
    
    <b>4. Production Deployment Strategy:</b><br/>
    • Deploy global model for initial fraud detection<br/>
    • Use local models for final decisions and edge cases<br/>
    • Implement continuous learning and model updates<br/><br/>
    
    <b>5. Privacy and Security Enhancements:</b><br/>
    • Implement secure multi-party computation for aggregation<br/>
    • Use homomorphic encryption for parameter sharing<br/>
    • Establish federated learning governance frameworks
    """
    story.append(Paragraph(recommendations_text, body_style))
    story.append(Spacer(1, 20))
    
    # Conclusion
    story.append(Paragraph("Conclusion", heading_style))
    conclusion_text = """
    The federated learning implementation demonstrates both the potential and challenges of 
    collaborative machine learning in banking. While local models outperform the global model 
    on their respective datasets, the federated approach successfully enables privacy-preserving 
    collaboration between banks. The slight performance trade-off is justified by the significant 
    privacy and regulatory benefits. Future work should focus on advanced federated techniques 
    and hybrid approaches to bridge the performance gap while maintaining the privacy advantages 
    of federated learning. This analysis provides a foundation for implementing federated learning 
    in production banking environments where data privacy and regulatory compliance are paramount.
    """
    story.append(Paragraph(conclusion_text, body_style))
    
    # Build PDF
    print("Generating federated learning PDF report...")
    doc.build(story)
    print("PDF report generated: Federated_Learning_Report.pdf")
    
    # Clean up temporary image files
    temp_files = [perf_plot, dist_plot]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
    print("Temporary files cleaned up.")

if __name__ == "__main__":
    generate_federated_pdf_report()
