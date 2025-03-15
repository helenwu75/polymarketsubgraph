#!/usr/bin/env python3
"""
Streamlined Summary Statistics for Polymarket Election Markets Analysis

This script provides a streamlined approach to generating summary statistics
for Polymarket election markets analysis, including target variable distribution, prediction quality metrics, feature importance, and market outcome analysis.

The script is designed to be run in a Jupyter notebook or as a standalone Python script.

Author: Helen Wu
Last updated: 2025-03-15
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure visualization
plt.style.use('ggplot')
sns.set(font_scale=1.1)
sns.set_style("whitegrid")

# Output directory
OUTPUT_DIR = "summary_stats_results"

def load_data(file_path):
    """Load and preprocess the dataset"""
    print(f"Loading data from {file_path}...")
    
    # Load with appropriate types
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert date columns
    for col in ['startDate', 'endDate', 'market_start_date', 'market_end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['brier_score', 'log_loss', 'prediction_correct', 'prediction_error',
                   'closing_price', 'price_volatility', 'trading_frequency', 'buy_sell_ratio']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def analyze_target(df, output_dir=OUTPUT_DIR):
    """Analyze and visualize the target variable"""
    if 'prediction_correct' not in df.columns:
        return None
        
    # Calculate statistics
    target_counts = df['prediction_correct'].value_counts().sort_index()
    target_pcts = df['prediction_correct'].value_counts(normalize=True).sort_index() * 100
    
    # Prepare data for visualization
    labels = ['Incorrect Prediction', 'Correct Prediction']
    values = [target_counts.get(0, 0), target_counts.get(1, 0)]
    pcts = [target_pcts.get(0, 0), target_pcts.get(1, 0)]
    
    # Create visualization with more explicit styling
    plt.figure(figsize=(10, 6))
    
    # Use basic bar plot instead of seaborn for more control
    x_pos = np.arange(len(labels))
    colors = ['#ff9999', '#66b3ff']
    bars = plt.bar(x_pos, values, color=colors)
    
    # Add text annotations with both counts and percentages
    for i, (val, pct) in enumerate(zip(values, pcts)):
        plt.text(x_pos[i], val/2, f"{val}\n({pct:.1f}%)", 
                ha='center', va='center', fontweight='bold', color='black', fontsize=12)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(x_pos[i], v + (max(values) * 0.02), f"{v}", 
                ha='center', fontweight='bold')
    
    plt.title('Distribution of Prediction Correctness', fontsize=14)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x_pos, labels, fontsize=11)
    plt.ylim(0, max(values) * 1.1)  # Add some space above the bars
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_distribution.png", dpi=300)
    plt.close()
    
    return {
        'counts': target_counts.to_dict(),
        'percentages': target_pcts.to_dict(),
        'imbalance_ratio': target_counts.max() / target_counts.min() if len(target_counts) > 1 and target_counts.min() > 0 else None
    }

def analyze_prediction_metrics(df, output_dir=OUTPUT_DIR):
    """Analyze prediction quality metrics (Brier score, log loss, and prediction error)"""
    metrics = {}
    metric_cols = ['brier_score', 'log_loss', 'prediction_error']
    
    # Filter to available metrics
    available_metrics = [col for col in metric_cols if col in df.columns]
    if not available_metrics:
        return metrics
        
    # Calculate and visualize each metric
    for col in available_metrics:
        valid_data = df[col].dropna()
        if len(valid_data) < 5:
            continue
            
        # Calculate statistics
        metrics[f"{col}_mean"] = valid_data.mean()
        metrics[f"{col}_median"] = valid_data.median()
        metrics[f"{col}_std"] = valid_data.std()
        metrics[f"{col}_min"] = valid_data.min()
        metrics[f"{col}_max"] = valid_data.max()
        
        # Calculate percentiles for analysis
        q1, q3 = valid_data.quantile([0.25, 0.75])
        metrics[f"{col}_q1"] = q1
        metrics[f"{col}_q3"] = q3
        
        # Create histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(valid_data, kde=True, color='steelblue')
        plt.axvline(valid_data.mean(), color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {valid_data.mean():.4f}')
        plt.axvline(valid_data.median(), color='green', linestyle='-.', alpha=0.7, 
                    label=f'Median: {valid_data.median():.4f}')
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_distribution.png", dpi=300)
        plt.close()
        
        # Create boxplot by correctness if we have the binary outcome
        if 'prediction_correct' in df.columns:
            comparison_df = df[[col, 'prediction_correct']].dropna()
            
            if len(comparison_df) > 5:
                plt.figure(figsize=(8, 5))
                # Use basic boxplot instead of seaborn for more control
                boxplot_data = [comparison_df[comparison_df['prediction_correct'] == 0][col],
                              comparison_df[comparison_df['prediction_correct'] == 1][col]]
                
                plt.boxplot(boxplot_data, labels=['Incorrect', 'Correct'], 
                           patch_artist=True, 
                           boxprops=dict(facecolor='#ff9999' if col in ['brier_score', 'log_loss'] else '#66b3ff'))
                
                plt.title(f'{col.replace("_", " ").title()} by Prediction Correctness')
                plt.ylabel(col.replace("_", " ").title())
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{col}_by_correctness.png", dpi=300)
                plt.close()
    
    # Create correlation matrix between metrics if we have multiple
    if len(available_metrics) >= 2 and 'prediction_correct' in df.columns:
        corr_cols = available_metrics + ['prediction_correct']
        corr_data = df[corr_cols].dropna()
        
        if len(corr_data) > 0:
            corr_matrix = corr_data.corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                      linewidths=0.5, square=True)
            plt.title('Correlation Between Prediction Metrics')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/prediction_metrics_correlation.png", dpi=300)
            plt.close()
    
    # Special analysis for Brier score and Log loss
    for metric in ['brier_score', 'log_loss']:
        if metric in df.columns and 'prediction_correct' in df.columns:
            analyze_prediction_metric_details(df, metric, output_dir)
    
    return metrics

def analyze_prediction_metric_details(df, metric_col, output_dir=OUTPUT_DIR):
    """Perform detailed analysis on Brier score or Log loss"""
    if metric_col not in df.columns or df[metric_col].isna().all():
        return
    
    # Prepare data
    analysis_df = df[[metric_col, 'prediction_correct']].dropna().copy()
    if len(analysis_df) < 10:
        return
    
    # Get metric values by prediction correctness
    correct_values = analysis_df[analysis_df['prediction_correct'] == 1][metric_col]
    incorrect_values = analysis_df[analysis_df['prediction_correct'] == 0][metric_col]
    
    # Calculate statistics
    stats = {
        'correct_mean': correct_values.mean() if len(correct_values) > 0 else None,
        'incorrect_mean': incorrect_values.mean() if len(incorrect_values) > 0 else None,
        'overall_mean': analysis_df[metric_col].mean(),
        'min': analysis_df[metric_col].min(),
        'max': analysis_df[metric_col].max()
    }
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create separate distributions for correct and incorrect predictions
    if len(correct_values) > 0:
        sns.kdeplot(correct_values, fill=True, color='#66b3ff', alpha=0.5, 
                  label=f'Correct Predictions (n={len(correct_values)})')
    
    if len(incorrect_values) > 0:
        sns.kdeplot(incorrect_values, fill=True, color='#ff9999', alpha=0.5, 
                   label=f'Incorrect Predictions (n={len(incorrect_values)})')
    
    # Add mean lines
    if stats['correct_mean'] is not None:
        plt.axvline(stats['correct_mean'], color='blue', linestyle='--', alpha=0.7,
                   label=f'Correct Mean: {stats["correct_mean"]:.4f}')
    
    if stats['incorrect_mean'] is not None:
        plt.axvline(stats['incorrect_mean'], color='red', linestyle='--', alpha=0.7,
                   label=f'Incorrect Mean: {stats["incorrect_mean"]:.4f}')
    
    plt.axvline(stats['overall_mean'], color='black', linestyle='-', alpha=0.7,
               label=f'Overall Mean: {stats["overall_mean"]:.4f}')
    
    # Add legend and labels
    plt.title(f'{metric_col.replace("_", " ").title()} by Prediction Outcome')
    plt.xlabel(metric_col.replace("_", " ").title())
    plt.ylabel('Density')
    plt.legend()
    
    # Add metric interpretation note
    metric_name = "Brier Score" if metric_col == "brier_score" else "Log Loss"
    interpretation = "Lower values indicate better predictions."
    plt.figtext(0.5, 0.01, f"Note: {metric_name} - {interpretation}", 
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_col}_detailed_analysis.png", dpi=300)
    plt.close()
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(f"{output_dir}/{metric_col}_statistics.csv", index=False)
    
    return stats

def feature_importance(df, output_dir=OUTPUT_DIR):
    """Analyze correlations between features and prediction outcomes"""
    if 'prediction_correct' not in df.columns:
        return pd.DataFrame()
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation with target
    target_corr = numeric_df.corr()['prediction_correct'].dropna().sort_values(ascending=False)
    target_corr = target_corr[target_corr.index != 'prediction_correct']
    
    # Get top correlations (both positive and negative)
    top_positive = target_corr.head(5)
    top_negative = target_corr.tail(5)
    top_corrs = pd.concat([top_positive, top_negative])
    
    # Create bar chart
    plt.figure(figsize=(10, 8))
    plot_df = pd.DataFrame({
        'feature': top_corrs.index,
        'correlation': top_corrs.values
    }).sort_values('correlation', ascending=False)
    
    # Ensure y-axis has proper labels
    y_pos = np.arange(len(plot_df['feature']))
    
    # Create horizontal bar chart
    bar_colors = ['#66b3ff' if x > 0 else '#ff9999' for x in plot_df['correlation']]
    plt.barh(y=y_pos, width=plot_df['correlation'], color=bar_colors)
    
    # Set feature names as y-tick labels
    plt.yticks(y_pos, plot_df['feature'])
    
    # Add text labels
    for i, v in enumerate(plot_df['correlation']):
        plt.text(v + 0.01 if v >= 0 else v - 0.08, i, f"{v:.3f}", 
                va='center', fontweight='bold', color='black' if v >= 0 else 'white')
    
    plt.title('Correlation with Prediction Correctness')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
    plt.close()
    
    # Return the correlations for reporting
    return top_corrs

def analyze_by_category(df, category_col, output_dir=OUTPUT_DIR):
    """Create table of market outcomes by category (election type, country)"""
    if category_col not in df.columns or 'prediction_correct' not in df.columns:
        return None
    
    # Filter out missing values
    analysis_df = df[[category_col, 'prediction_correct']].dropna()
    if len(analysis_df) == 0:
        return None
    
    # Group by category
    category_stats = analysis_df.groupby(category_col).agg({
        'prediction_correct': ['mean', 'count']
    })
    
    # Flatten multiindex
    category_stats.columns = ['accuracy', 'count']
    category_stats = category_stats.sort_values('count', ascending=False)
    
    # Filter to categories with enough samples and take top 15
    filtered_stats = category_stats[category_stats['count'] >= 5].head(15).copy()
    if filtered_stats.empty:
        return None
    
    # Format the percentages
    filtered_stats['accuracy_pct'] = (filtered_stats['accuracy'] * 100).round(1).astype(str) + '%'
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    categories = filtered_stats.index.tolist()
    counts = filtered_stats['count'].values
    accuracies = filtered_stats['accuracy'].values * 100
    
    # Create table style visualization
    table_data = []
    for cat, cnt, acc in zip(categories, counts, accuracies):
        table_data.append([str(cat), int(cnt), f"{acc:.1f}%"])
    
    # Add header
    col_labels = [category_col.replace('_', ' ').title(), "Count", "Accuracy"]
    
    # Create table
    table = plt.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=[0.5, 0.2, 0.3],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]  # Position the table in the figure
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)
    
    # Style header
    for i in range(len(col_labels)):
        cell = table[0, i]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Add colors to accuracy cells
    for i in range(len(table_data)):
        # Alternating row colors
        row_idx = i + 1  # +1 to account for header
        if i % 2 == 0:
            for j in range(len(col_labels)):
                table[row_idx, j].set_facecolor('#E6F0FF')
        
        # Color accuracy cells based on value
        acc_cell = table[row_idx, 2]
        acc_text = acc_cell.get_text().get_text()
        acc_value = float(acc_text.strip('%')) / 100
        
        if acc_value >= 0.9:
            acc_cell.set_facecolor('#C6EFCE')  # Green
        elif acc_value >= 0.8:
            acc_cell.set_facecolor('#FFEB9C')  # Yellow
        else:
            acc_cell.set_facecolor('#FFC7CE')  # Red
    
    # Turn off axis
    plt.axis('off')
    
    plt.title(f'Prediction Accuracy by {category_col.replace("_", " ").title()}', 
              fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_{category_col}.png", dpi=300)
    plt.close()
    
    # Save to CSV
    result_df = filtered_stats.reset_index()
    result_df.columns = [category_col, 'Accuracy', 'Count', 'Accuracy (%)']
    result_df.to_csv(f"{output_dir}/accuracy_by_{category_col}.csv", index=False)
    
    return result_df

def generate_feature_distributions(df, output_dir=OUTPUT_DIR):
    """Generate distributions for key features"""
    # Select important features
    key_features = [
        'brier_score', 'log_loss', 'price_volatility', 
        'trading_frequency', 'two_way_traders_ratio', 'late_stage_participation'
    ]
    
    # Filter to features that exist in the data
    available_features = [f for f in key_features if f in df.columns]
    if not available_features:
        return
    
    # Create grid of plots
    n_cols = min(3, len(available_features))
    n_rows = (len(available_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])  # Handle single plot case
    axes = axes.flatten()
    
    for i, col in enumerate(available_features):
        plot_data = df[col].dropna()
        if len(plot_data) < 5:
            continue
            
        # Filter extreme outliers
        q1, q3 = plot_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        plot_data = plot_data[(plot_data >= q1 - 3*iqr) & (plot_data <= q3 + 3*iqr)]
        
        # Plot histogram
        sns.histplot(plot_data, kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
        
        # Add mean and median
        mean_val = plot_data.mean()
        median_val = plot_data.median()
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_val:.2f}')
        axes[i].axvline(median_val, color='green', linestyle='-.', alpha=0.7, 
                       label=f'Median: {median_val:.2f}')
        axes[i].legend(fontsize='small')
    
    # Hide unused subplots
    for j in range(len(available_features), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300)
    plt.close()

def generate_basic_stats(df, output_dir=OUTPUT_DIR):
    """Generate basic statistics for numeric columns"""
    # Calculate statistics
    numeric_df = df.select_dtypes(include=['number'])
    stats = numeric_df.describe().T
    
    # Add missing values count and percentage
    stats['missing'] = df.shape[0] - numeric_df.count()
    stats['missing_pct'] = (stats['missing'] / df.shape[0] * 100).round(2)
    
    # Save to CSV
    stats.to_csv(f"{output_dir}/basic_statistics.csv")
    return stats

def create_summary_report(results, output_dir=OUTPUT_DIR):
    """Create a summary report with findings"""
    df = results.get('data')
    target_analysis = results.get('target_analysis')
    prediction_metrics = results.get('prediction_metrics')
    feature_corrs = results.get('feature_correlations')
    election_type_stats = results.get('election_type_stats')
    country_stats = results.get('country_stats')
    
    report_path = f"{output_dir}/summary_statistics_report.md"
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Polymarket Election Markets Analysis: Summary Statistics\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Dataset overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Markets:** {df.shape[0]}\n")
        f.write(f"- **Total Features:** {df.shape[1]}\n\n")
        
        # Target variable analysis
        if target_analysis:
            f.write("## Prediction Correctness\n\n")
            
            correct_count = target_analysis['counts'].get(1, 0)
            incorrect_count = target_analysis['counts'].get(0, 0)
            correct_pct = target_analysis['percentages'].get(1, 0)
            incorrect_pct = target_analysis['percentages'].get(0, 0)
            
            f.write(f"- **Correct Predictions:** {correct_count} ({correct_pct:.1f}%)\n")
            f.write(f"- **Incorrect Predictions:** {incorrect_count} ({incorrect_pct:.1f}%)\n\n")
            
            if target_analysis.get('imbalance_ratio'):
                f.write(f"- **Class Imbalance Ratio:** {target_analysis['imbalance_ratio']:.2f}\n\n")
            
            f.write("![Target Distribution](target_distribution.png)\n\n")
        
        # Prediction metrics
        if prediction_metrics:
            f.write("## Prediction Quality Metrics\n\n")
            
            if 'brier_score_mean' in prediction_metrics:
                brier_mean = prediction_metrics['brier_score_mean']
                brier_median = prediction_metrics['brier_score_median']
                f.write(f"- **Brier Score:** Mean: {brier_mean:.4f}, Median: {brier_median:.4f}\n")
                f.write("  *(Lower values indicate better predictive performance)*\n\n")
                f.write("![Brier Score Distribution](brier_score_distribution.png)\n\n")
                
                # Add detailed Brier score analysis if the file exists
                if os.path.exists(f"{output_dir}/brier_score_detailed_analysis.png"):
                    f.write("### Brier Score Analysis by Prediction Outcome\n\n")
                    f.write("![Brier Score Analysis](brier_score_detailed_analysis.png)\n\n")
            
            if 'log_loss_mean' in prediction_metrics:
                log_loss_mean = prediction_metrics['log_loss_mean']
                log_loss_median = prediction_metrics['log_loss_median']
                f.write(f"- **Log Loss:** Mean: {log_loss_mean:.4f}, Median: {log_loss_median:.4f}\n")
                f.write("  *(Lower values indicate better predictive performance, with higher penalty for confident but wrong predictions)*\n\n")
                f.write("![Log Loss Distribution](log_loss_distribution.png)\n\n")
                
                # Add detailed Log Loss analysis if the file exists
                if os.path.exists(f"{output_dir}/log_loss_detailed_analysis.png"):
                    f.write("### Log Loss Analysis by Prediction Outcome\n\n")
                    f.write("![Log Loss Analysis](log_loss_detailed_analysis.png)\n\n")
            
            # Add correlation image if it exists
            if os.path.exists(f"{output_dir}/prediction_metrics_correlation.png"):
                f.write("### Correlation Between Metrics\n\n")
                f.write("![Prediction Metrics Correlation](prediction_metrics_correlation.png)\n\n")
        
        # Feature importance
        if feature_corrs is not None and not feature_corrs.empty:
            f.write("## Feature Importance\n\n")
            f.write("Correlation between features and prediction correctness:\n\n")
            f.write("![Feature Importance](feature_importance.png)\n\n")
            
            # Add top positive correlations
            positive_corrs = feature_corrs[feature_corrs > 0]
            if not positive_corrs.empty:
                f.write("### Top Positive Correlations\n\n")
                for feature, corr in positive_corrs.items():
                    feature_name = feature.replace('_', ' ').title()
                    f.write(f"- **{feature_name}**: {corr:.3f}\n")
                f.write("\n")
            
            # Add top negative correlations
            negative_corrs = feature_corrs[feature_corrs < 0]
            if not negative_corrs.empty:
                f.write("### Top Negative Correlations\n\n")
                negative_corrs = negative_corrs.sort_values()  # Sort ascending for most negative first
                for feature, corr in negative_corrs.items():
                    feature_name = feature.replace('_', ' ').title()
                    f.write(f"- **{feature_name}**: {corr:.3f}\n")
                f.write("\n")
        
        # Election type analysis
        if election_type_stats is not None and not election_type_stats.empty:
            f.write("## Accuracy by Election Type\n\n")
            
            table_rows = []
            table_rows.append("| Election Type | Count | Accuracy |")
            table_rows.append("|--------------|-------|----------|")
            
            for _, row in election_type_stats.iterrows():
                election_type = row['event_electionType']
                count = row['Count']
                accuracy = row['Accuracy (%)']
                table_rows.append(f"| {election_type} | {count} | {accuracy} |")
            
            f.write("\n".join(table_rows) + "\n\n")
            f.write("![Accuracy by Election Type](accuracy_by_event_electionType.png)\n\n")
        
        # Country analysis
        if country_stats is not None and not country_stats.empty:
            f.write("## Accuracy by Country\n\n")
            
            table_rows = []
            table_rows.append("| Country | Count | Accuracy |")
            table_rows.append("|---------|-------|----------|")
            
            # Limit to top 10 countries by count
            for _, row in country_stats.head(10).iterrows():
                country = row['event_countryName']
                count = row['Count']
                accuracy = row['Accuracy (%)']
                table_rows.append(f"| {country} | {count} | {accuracy} |")
            
            f.write("\n".join(table_rows) + "\n\n")
            f.write("![Accuracy by Country](accuracy_by_event_countryName.png)\n\n")
        
        # Feature distributions
        f.write("## Feature Distributions\n\n")
        f.write("![Feature Distributions](feature_distributions.png)\n\n")
        
        # Recommendations for metric selection
        f.write("## Recommendations for Prediction Quality Metrics\n\n")
        
        has_brier = 'brier_score' in df.columns
        has_log_loss = 'log_loss' in df.columns
        
        if has_brier and has_log_loss:
            f.write("Based on the analysis, we recommend:\n\n")
            f.write("1. **Primary Metric: Brier Score** - Provides a balanced assessment of prediction accuracy\n")
            f.write("2. **Secondary Metric: Log Loss** - Offers additional insights by penalizing confident but wrong predictions\n")
            f.write("3. **Simplified Metric: Prediction Correctness** - When binary classification accuracy is needed\n")
        elif has_brier:
            f.write("**Brier Score is recommended** as the primary metric for evaluating prediction quality\n")
        elif has_log_loss:
            f.write("**Log Loss is recommended** as the primary metric for evaluating prediction quality\n")
        else:
            f.write("**Prediction Correctness** should be used as the primary metric given the available data\n")
        
        # Summary
        f.write("\n## Summary\n\n")
        if 'prediction_correct' in df.columns:
            correct_pct = df['prediction_correct'].mean() * 100
            f.write(f"The analyzed Polymarket election markets show an overall prediction accuracy of {correct_pct:.1f}%. ")
        f.write("The analysis identified key factors associated with prediction accuracy ")
        f.write("that can help understand which market characteristics contribute to successful predictions.\n")
    
    print(f"Summary report generated at {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate summary statistics for Polymarket data")
    parser.add_argument("--input", default="election_metrics_results.csv", help="Input CSV file")
    parser.add_argument("--output-dir", default="summary_stats_results", help="Output directory")
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        df = load_data(args.input)
        
        # Store all results for final report
        results = {'data': df}
        
        # Generate statistics
        print("Calculating basic statistics...")
        generate_basic_stats(df, output_dir)
        
        print("Analyzing target variable...")
        results['target_analysis'] = analyze_target(df, output_dir)
        
        print("Analyzing prediction metrics...")
        results['prediction_metrics'] = analyze_prediction_metrics(df, output_dir)
        
        print("Analyzing feature importance...")
        results['feature_correlations'] = feature_importance(df, output_dir)
        
        print("Generating feature distributions...")
        generate_feature_distributions(df, output_dir)
        
        # Category analysis
        if 'event_electionType' in df.columns:
            print("Analyzing by election type...")
            results['election_type_stats'] = analyze_by_category(df, 'event_electionType', output_dir)
        
        if 'event_countryName' in df.columns:
            print("Analyzing by country...")
            results['country_stats'] = analyze_by_category(df, 'event_countryName', output_dir)
        
        # Create report
        print("Generating summary report...")
        create_summary_report(results, output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}/")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()