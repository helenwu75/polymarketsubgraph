#!/usr/bin/env python3
"""
Summary Statistics for Polymarket Election Markets Analysis

This script generates comprehensive summary statistics for Polymarket election markets data,
including descriptive statistics, distributions, and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create output directory for results
output_dir = "summary_stats_results"
os.makedirs(output_dir, exist_ok=True)

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)

def load_data(file_path):
    """
    Load the dataset with appropriate data types to avoid warnings.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}...")
    
    # For large files, use chunksize to avoid memory issues
    if file_path.endswith(".csv"):
        # First check if the file is too large (more than 500MB)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        if file_size > 500:
            print(f"Large file detected ({file_size:.1f} MB). Using chunked loading...")
            
            # Read first chunk to get column names
            first_chunk = pd.read_csv(file_path, nrows=5, low_memory=False)
            dtypes = {}
            
            # Set appropriate types for known columns
            for col in first_chunk.columns:
                if col in ['prediction_correct', 'price_fluctuations']:
                    dtypes[col] = 'Int64'  # Nullable integer
                elif col in ['closing_price', 'price_volatility', 'trading_frequency']:
                    dtypes[col] = 'float64'
            
            # Sample the data instead of loading everything
            # This will load every nth row to get a representative sample
            sample_size = 100000
            total_rows = sum(1 for _ in open(file_path, 'r'))
            
            if total_rows > sample_size:
                skip_rows = sorted(set(range(1, total_rows)) - set(range(1, total_rows, total_rows // sample_size)))
                df = pd.read_csv(file_path, skiprows=skip_rows, low_memory=False, dtype=dtypes)
                print(f"Loaded sample of {len(df)} rows from {total_rows} total rows")
            else:
                df = pd.read_csv(file_path, low_memory=False, dtype=dtypes)
        else:
            # Regular load for smaller files
            df = pd.read_csv(file_path, low_memory=False)
    else:
        df = pd.read_csv(file_path, low_memory=False)
    
    # Convert date columns to datetime
    date_columns = ['startDate', 'endDate', 'market_start_date', 'market_end_date']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                print(f"Warning: Couldn't convert {col} to datetime")
    
    # Convert numeric columns to appropriate types
    numeric_columns = [
        'volumeNum', 'volumeClob', 'event_volume', 'event_commentCount',
        'yes_token_id', 'closing_price', 'price_2days_prior', 'pre_election_vwap_48h',
        'price_volatility', 'price_range', 'final_week_momentum', 'price_fluctuations',
        'last_trade_price', 'prediction_correct', 'prediction_error', 'prediction_confidence',
        'market_duration_days', 'trading_frequency', 'buy_sell_ratio', 'trading_continuity',
        'late_stage_participation', 'volume_acceleration', 'unique_traders_count',
        'trader_to_trade_ratio', 'two_way_traders_ratio', 'trader_concentration',
        'new_trader_influx', 'comment_per_vol', 'comment_per_trader'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert categorical columns to categorical type
    categorical_columns = ['event_ticker', 'event_slug', 'event_title', 
                          'event_countryName', 'event_electionType', 'correct_outcome']
    for col in categorical_columns:
        if col in df.columns and col in df:
            df[col] = df[col].astype('category')
    
    # Create additional features
    if 'startDate' in df.columns and 'endDate' in df.columns:
        try:
            # Double-check market duration if it isn't already calculated
            if 'market_duration_days' not in df.columns:
                df['market_duration_days'] = (df['endDate'] - df['startDate']).dt.days
        except:
            print("Warning: Couldn't calculate market_duration_days")
    
    # Verify target variable is properly formatted
    if 'prediction_correct' in df.columns:
        # Make sure it's binary (0 or 1)
        valid_values = df['prediction_correct'].dropna().unique()
        print(f"Unique values in prediction_correct: {valid_values}")
        
        if not all(v in [0, 1, 0.0, 1.0] for v in valid_values):
            print("Warning: prediction_correct contains non-binary values")
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def get_basic_stats(df):
    """
    Generate basic descriptive statistics for all numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pandas.DataFrame: Summary statistics
    """
    # Include all numeric types to capture Int64 and other numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'Int64', 'float64'])
    
    # Handle potential overflow issues by using safe calculations
    try:
        # Calculate basic statistics - use .astype(float) to prevent overflow
        with np.errstate(all='ignore'):  # Ignore numpy warnings
            stats = numeric_df.astype(float).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
        
        # Add additional statistics
        stats['missing'] = df.shape[0] - numeric_df.count()
        stats['missing_pct'] = (stats['missing'] / df.shape[0] * 100).round(2)
        
        # Calculate skew and kurtosis safely
        with np.errstate(all='ignore'):
            # Use ddof=0 to match pandas default
            stats['skew'] = numeric_df.astype(float).apply(lambda x: float(pd.Series(x.dropna()).skew()), axis=0)
            stats['kurtosis'] = numeric_df.astype(float).apply(lambda x: float(pd.Series(x.dropna()).kurtosis()), axis=0)
        
        # Reorder columns for better readability
        stats = stats[['count', 'missing', 'missing_pct', 'mean', 'std', 'min', 
                      '10%', '25%', '50%', '75%', '90%', 'max', 'skew', 'kurtosis']]
        
    except Exception as e:
        print(f"Warning: Error calculating statistics: {e}")
        # Fallback to simpler statistics if calculations fail
        stats = pd.DataFrame({
            'count': numeric_df.count(),
            'missing': df.shape[0] - numeric_df.count(),
            'missing_pct': ((df.shape[0] - numeric_df.count()) / df.shape[0] * 100).round(2),
            'mean': numeric_df.mean(),
            'median': numeric_df.median(),
            'min': numeric_df.min(),
            'max': numeric_df.max()
        }).T
    
    return stats

def target_variable_analysis(df, target_col='prediction_correct'):
    """
    Analyze the target variable distribution.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        dict: Target analysis results
    """
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found in dataframe")
        return {}
    
    # Make a clean copy of the target column
    target_data = df[target_col].dropna().copy()
    
    # Convert to binary if needed
    if not all(val in [0, 1, 0.0, 1.0] for val in target_data.unique()):
        print("Converting non-binary values in target variable to binary")
        target_data = target_data.apply(lambda x: 1 if x > 0.5 else 0)
    
    target_counts = target_data.value_counts()
    target_pcts = target_data.value_counts(normalize=True) * 100
    
    # Create a summary dictionary
    results = {
        'counts': target_counts.to_dict(),
        'percentages': target_pcts.to_dict(),
    }
    
    # Class imbalance ratio (majority/minority)
    if len(target_counts) >= 2:
        majority_class_count = target_counts.max()
        minority_class_count = target_counts.min()
        imbalance_ratio = majority_class_count / minority_class_count
        results['imbalance_ratio'] = imbalance_ratio
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    
    # Using updated seaborn syntax to avoid deprecation warnings
    ax = sns.countplot(data=df, x=target_col, hue=target_col, palette='viridis', legend=False)
    
    # Add percentages on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if height > 0:  # Only add text if the bar has height
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 5,
                    f'{target_pcts.iloc[i if i < len(target_pcts) else 0]:.1f}%',
                    ha="center")
    
    plt.title(f'Distribution of {target_col}')
    plt.ylabel('Count')
    
    # Set x-axis ticks with correct values
    plt.xticks(ticks=[0, 1], labels=['Incorrect (0)', 'Correct (1)'])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_distribution.png", dpi=300)
    plt.close()
    
    return results

def feature_distributions(df, n_features=12):
    """
    Plot distributions for the top numeric features.
    
    Args:
        df: Input DataFrame
        n_features: Number of features to plot
    """
    # Select numeric columns with the least missing values
    numeric_cols = df.select_dtypes(include=['int64', 'Int64', 'float64']).columns
    
    if len(numeric_cols) == 0:
        print("Warning: No numeric columns found for distribution plots")
        return
        
    # Get columns with fewer missing values
    missing_counts = df[numeric_cols].isnull().sum()
    
    # Prefer important features if they exist
    important_features = [
        'prediction_correct', 'closing_price', 'price_volatility', 
        'market_duration_days', 'trading_frequency', 'buy_sell_ratio',
        'unique_traders_count', 'trader_to_trade_ratio', 'two_way_traders_ratio',
        'late_stage_participation', 'prediction_error', 'prediction_confidence'
    ]
    
    # Filter to features that exist in the dataset
    available_important = [f for f in important_features if f in numeric_cols]
    
    # If we have enough important features, use those, otherwise use least missing
    if len(available_important) >= n_features:
        top_features = available_important[:n_features]
    else:
        # Fill remaining slots with least missing features
        remaining_features = [f for f in missing_counts.sort_values().index 
                              if f not in available_important]
        top_features = (available_important + 
                       remaining_features[:n_features-len(available_important)])
    
    # Determine grid size based on feature count
    n_features = min(n_features, len(top_features))
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Create histograms with proper grid size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, col in enumerate(top_features):
        if i < len(axes):
            try:
                # Use safe data for plotting
                plot_data = df[col].dropna()
                
                # Skip if not enough data
                if len(plot_data) < 5:
                    axes[i].text(0.5, 0.5, f"Insufficient data for {col}", 
                                ha='center', va='center')
                    axes[i].set_title(f'Distribution of {col}')
                    continue
                
                # Check for extreme outliers that might distort the plot
                q1, q3 = plot_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                # Filter extreme outliers for better visualization
                plot_data = plot_data[(plot_data >= lower_bound) & (plot_data <= upper_bound)]
                
                # Create the histogram
                sns.histplot(plot_data, kde=True, ax=axes[i], color='steelblue')
                axes[i].set_title(f'Distribution of {col}')
                
                # Add mean and median lines
                with np.errstate(all='ignore'):
                    mean_val = plot_data.mean()
                    median_val = plot_data.median()
                
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                               label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='-.', alpha=0.7, 
                                label=f'Median: {median_val:.2f}')
                axes[i].legend(fontsize='small')
                
            except Exception as e:
                print(f"Error plotting distribution for {col}: {e}")
                axes[i].text(0.5, 0.5, f"Error plotting {col}", ha='center', va='center')
                axes[i].set_title(f'Distribution of {col}')
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300)
    plt.close()

def correlation_analysis(df, threshold=0.3):
    """
    Analyze correlations between features.
    
    Args:
        df: Input DataFrame
        threshold: Minimum absolute correlation to report
        
    Returns:
        pandas.DataFrame: Strong correlations
    """
    # Select numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=['int64', 'Int64', 'float64'])
    
    # Check if we have enough data
    if numeric_df.shape[1] < 2:
        print("Warning: Not enough numeric columns for correlation analysis")
        return pd.DataFrame()
    
    # For large datasets, sample to make computation manageable
    if len(numeric_df) > 10000:
        print(f"Sampling {10000} rows for correlation analysis")
        numeric_df = numeric_df.sample(10000, random_state=42)
    
    # Calculate correlation matrix safely
    try:
        with np.errstate(all='ignore'):  # Ignore numpy warnings
            # Drop columns with all NaN values first
            non_empty_cols = [col for col in numeric_df.columns 
                              if numeric_df[col].notna().sum() > 0]
            numeric_df = numeric_df[non_empty_cols]
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr(method='pearson', min_periods=5)
            
            # Handle any NaN values in the correlation matrix
            corr_matrix = corr_matrix.fillna(0)
    except Exception as e:
        print(f"Warning: Error calculating correlation matrix: {e}")
        return pd.DataFrame()
    
    # Create heatmap
    try:
        # Keep only the 20 most important features for readability
        if len(corr_matrix) > 20:
            # If target exists, include it and other features with highest correlation to target
            if 'prediction_correct' in corr_matrix.columns:
                target_corrs = corr_matrix['prediction_correct'].abs().sort_values(ascending=False)
                top_features = list(target_corrs.index[:19])  # Top 19 + target = 20
                if 'prediction_correct' not in top_features:
                    top_features.append('prediction_correct')
            else:
                # Otherwise, find features with highest average absolute correlation
                mean_abs_corrs = corr_matrix.abs().mean().sort_values(ascending=False)
                top_features = list(mean_abs_corrs.index[:20])
            
            corr_matrix = corr_matrix.loc[top_features, top_features]
        
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False,
                   center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Error creating correlation heatmap: {e}")
    
    # Create correlation with target heatmap if target exists
    if 'prediction_correct' in corr_matrix.columns:
        try:
            target_corr = corr_matrix['prediction_correct'].sort_values(ascending=False)
            
            plt.figure(figsize=(12, 10))
            # Use updated seaborn syntax to avoid deprecation warnings
            sns.barplot(x=target_corr.values, y=target_corr.index, hue=target_corr.index, 
                       palette='coolwarm', legend=False)
            plt.title('Correlation with prediction_correct')
            plt.xlabel('Correlation Coefficient')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/target_correlation.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Error creating target correlation plot: {e}")
    
    # Find strong correlations
    strong_corrs = []
    try:
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2 and col1 < col2:  # Avoid duplicates and self-correlations
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) >= threshold:
                        strong_corrs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': corr
                        })
        
        # Create DataFrame of strong correlations
        strong_corrs_df = pd.DataFrame(strong_corrs)
        if not strong_corrs_df.empty:
            strong_corrs_df = strong_corrs_df.sort_values('correlation', key=abs, ascending=False)
        
        return strong_corrs_df
    except Exception as e:
        print(f"Warning: Error finding strong correlations: {e}")
        return pd.DataFrame()

def market_analysis_by_category(df, category_col, target_col='prediction_correct'):
    """
    Analyze market outcomes by category.
    
    Args:
        df: Input DataFrame
        category_col: Column name for categorization
        target_col: Target column
    """
    if category_col not in df.columns or target_col not in df.columns:
        print(f"Warning: Columns {category_col} or {target_col} not found in dataframe")
        return
    
    try:
        # Create a copy to avoid warnings
        analysis_df = df[[category_col, target_col]].copy()
        
        # Filter out rows with missing values in either column
        analysis_df = analysis_df.dropna(subset=[category_col, target_col])
        
        if len(analysis_df) == 0:
            print(f"Warning: No valid data for {category_col} analysis")
            return
        
        # Calculate accuracy by category - use observed=True to avoid FutureWarning
        category_accuracy = analysis_df.groupby(category_col, observed=True)[target_col].agg(['mean', 'count'])
        category_accuracy.columns = ['accuracy', 'count']
        category_accuracy = category_accuracy.sort_values('count', ascending=False)
        
        # Only plot categories with sufficient samples
        min_samples = 5
        plot_data = category_accuracy[category_accuracy['count'] >= min_samples].copy()
        
        if plot_data.empty:
            print(f"Warning: No categories with at least {min_samples} samples for {category_col}")
            return
            
        # Cap the number of categories for readability
        max_categories = 15
        if len(plot_data) > max_categories:
            plot_data = plot_data.head(max_categories)
            
        # Reset index to convert category to a column
        plot_data = plot_data.reset_index()
        
        plt.figure(figsize=(12, max(6, len(plot_data) * 0.4)))
        
        # Create a horizontal bar chart with updated syntax
        ax = sns.barplot(data=plot_data, x='accuracy', y=category_col, hue=category_col,
                         palette='viridis', legend=False)
        
        # Add count labels
        for i, row in enumerate(plot_data.itertuples()):
            ax.text(0.01, i, f"n={int(row.count)}", va='center')
        
        # Add percentage labels at the end of each bar
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            if width > 0:  # Only add text if the bar has width
                ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                        f'{width:.1%}', ha='left', va='center')
        
        plt.title(f'Prediction Accuracy by {category_col}')
        plt.xlabel('Accuracy')
        plt.xlim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_by_{category_col}.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error in market_analysis_by_category for {category_col}: {e}")

def feature_vs_target_plots(df, target_col='prediction_correct', n_features=6):
    """
    Create plots comparing top features vs target variable.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        n_features: Number of top features to plot
    """
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found in dataframe")
        return
    
    try:
        # Find features with the strongest correlation with the target
        numeric_df = df.select_dtypes(include=['int64', 'Int64', 'float64'])
        
        if target_col not in numeric_df.columns:
            print(f"Warning: Target column '{target_col}' not found in numeric columns")
            return
            
        # Sample if dataset is too large
        if len(numeric_df) > 10000:
            print(f"Sampling {10000} rows for correlation calculation")
            numeric_df = numeric_df.sample(10000, random_state=42)
            
        # Safe correlation calculation
        with np.errstate(all='ignore'):
            correlations = numeric_df.corr(method='pearson', min_periods=5)[target_col].abs().fillna(0)
            
        # Filter out target itself
        correlations = correlations[correlations.index != target_col]
        
        # Get top correlated features
        top_features = correlations.sort_values(ascending=False).head(n_features).index.tolist()
        
        if not top_features:
            print("Warning: No significant correlations found for boxplots")
            return
            
        # Calculate number of rows and columns for subplot grid
        n_cols = min(3, n_features)
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        # Create plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                try:
                    # Create a copy of the data with only needed columns
                    plot_df = df[[target_col, feature]].copy().dropna()
                    
                    # Skip if not enough data
                    if len(plot_df) < 10:
                        axes[i].text(0.5, 0.5, f"Insufficient data for {feature}", 
                                     ha='center', va='center')
                        axes[i].set_title(f'{feature} vs {target_col}')
                        continue
                    
                    # Make sure target is binary
                    plot_df[target_col] = plot_df[target_col].apply(lambda x: 1 if x > 0.5 else 0)
                    
                    # Filter extreme outliers for better visualization
                    q1, q3 = plot_df[feature].quantile([0.05, 0.95])
                    iqr = q3 - q1
                    filtered_df = plot_df[(plot_df[feature] >= q1 - 3*iqr) & 
                                         (plot_df[feature] <= q3 + 3*iqr)]
                    
                    # Use filtered data if we have enough, otherwise use original
                    if len(filtered_df) >= 10:
                        plot_df = filtered_df
                    
                    # Box plot with updated syntax
                    sns.boxplot(data=plot_df, x=target_col, y=feature, ax=axes[i], 
                               hue=target_col, palette='viridis', legend=False)
                    
                    axes[i].set_title(f'{feature} vs {target_col}')
                    axes[i].set_xlabel('Prediction Correct')
                    axes[i].set_ylabel(feature)
                    
                    # Set x-tick labels manually
                    axes[i].set_xticks([0, 1])
                    axes[i].set_xticklabels(['Incorrect (0)', 'Correct (1)'])
                    
                except Exception as e:
                    print(f"Error creating boxplot for {feature}: {e}")
                    axes[i].text(0.5, 0.5, f"Error plotting {feature}", ha='center', va='center')
                    axes[i].set_title(f'{feature} vs {target_col}')
        
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/features_vs_target.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error in feature_vs_target_plots: {e}")

def generate_report(df, stats_df, target_analysis, strong_correlations):
    """
    Generate a comprehensive summary report.
    
    Args:
        df: Input DataFrame
        stats_df: Statistics DataFrame
        target_analysis: Target variable analysis results
        strong_correlations: Strong correlations DataFrame
    """
    report_path = f"{output_dir}/summary_statistics_report.md"
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Polymarket Election Markets Analysis: Summary Statistics\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Dataset overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Markets:** {df.shape[0]}\n")
        f.write(f"- **Total Features:** {df.shape[1]}\n")
        
        # Missing data summary
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'count': missing_counts, 'percentage': missing_pct})
        missing_df = missing_df[missing_df['count'] > 0].sort_values('count', ascending=False)
        
        if not missing_df.empty:
            f.write("\n### Missing Data Summary\n\n")
            f.write("| Feature | Missing Count | Missing Percentage |\n")
            f.write("|---------|---------------|--------------------|\n")
            for feature, row in missing_df.iterrows():
                f.write(f"| {feature} | {int(row['count'])} | {row['percentage']}% |\n")
        
        # Target variable analysis
        if target_analysis:
            f.write("\n## Target Variable Analysis\n\n")
            f.write("### Prediction Correctness Distribution\n\n")
            f.write("| Outcome | Count | Percentage |\n")
            f.write("|---------|-------|------------|\n")
            
            for outcome, count in target_analysis['counts'].items():
                pct = target_analysis['percentages'][outcome]
                label = "Correct" if outcome == 1 else "Incorrect"
                f.write(f"| {label} | {count} | {pct:.2f}% |\n")
            
            if 'imbalance_ratio' in target_analysis:
                f.write(f"\nClass imbalance ratio (majority/minority): {target_analysis['imbalance_ratio']:.2f}\n")
            
            f.write("\n![Target Distribution](target_distribution.png)\n")
        
        # Categorical distribution for election types
        if 'event_electionType' in df.columns:
            f.write("\n## Election Type Distribution\n\n")
            election_counts = df['event_electionType'].value_counts()
            election_pcts = df['event_electionType'].value_counts(normalize=True) * 100
            
            f.write("| Election Type | Count | Percentage |\n")
            f.write("|---------------|-------|------------|\n")
            
            for election_type, count in election_counts.items():
                if pd.notna(election_type):
                    pct = election_pcts[election_type]
                    f.write(f"| {election_type} | {count} | {pct:.2f}% |\n")
            
            f.write("\n![Accuracy by Election Type](accuracy_by_event_electionType.png)\n")
        
        # Country distribution
        if 'event_countryName' in df.columns:
            country_counts = df['event_countryName'].value_counts().head(10)
            
            f.write("\n## Top Countries\n\n")
            f.write("| Country | Count |\n")
            f.write("|---------|-------|\n")
            
            for country, count in country_counts.items():
                if pd.notna(country):
                    f.write(f"| {country} | {count} |\n")
            
            f.write("\n![Accuracy by Country](accuracy_by_event_countryName.png)\n")
        
        # Key numeric statistics
        f.write("\n## Key Feature Statistics\n\n")
        
        key_metrics = [
            'prediction_correct', 'closing_price', 'price_volatility', 
            'market_duration_days', 'trading_frequency', 'buy_sell_ratio',
            'unique_traders_count', 'trader_to_trade_ratio', 'two_way_traders_ratio'
        ]
        
        available_metrics = [m for m in key_metrics if m in stats_df.index]
        
        if available_metrics:
            subset_stats = stats_df.loc[available_metrics, ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            
            f.write("| Feature | Count | Mean | Std | Min | 25% | Median | 75% | Max |\n")
            f.write("|---------|-------|------|-----|-----|-----|--------|-----|-----|\n")
            
            for feature, row in subset_stats.iterrows():
                f.write(f"| {feature} | {int(row['count'])} | {row['mean']:.3f} | {row['std']:.3f} | {row['min']:.3f} | {row['25%']:.3f} | {row['50%']:.3f} | {row['75%']:.3f} | {row['max']:.3f} |\n")
        
        # Correlations with target
        if 'prediction_correct' in df.columns:
            f.write("\n## Correlations with Prediction Correctness\n\n")
            
            # Calculate correlations with the target
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if 'prediction_correct' in numeric_df.columns:
                target_corr = numeric_df.corr()['prediction_correct'].sort_values(ascending=False)
                
                # Filter out the target itself and show top correlations
                target_corr = target_corr[target_corr.index != 'prediction_correct']
                top_corr = target_corr.head(10)
                bottom_corr = target_corr.tail(5)
                
                f.write("### Top Positive Correlations\n\n")
                f.write("| Feature | Correlation |\n")
                f.write("|---------|-------------|\n")
                
                for feature, corr in top_corr.items():
                    f.write(f"| {feature} | {corr:.4f} |\n")
                
                f.write("\n### Top Negative Correlations\n\n")
                f.write("| Feature | Correlation |\n")
                f.write("|---------|-------------|\n")
                
                for feature, corr in bottom_corr.items():
                    f.write(f"| {feature} | {corr:.4f} |\n")
                
                f.write("\n![Correlations with Target](target_correlation.png)\n")
        
        # Strong feature correlations
        if not strong_correlations.empty:
            f.write("\n## Strong Feature Correlations\n\n")
            f.write("| Feature 1 | Feature 2 | Correlation |\n")
            f.write("|-----------|-----------|-------------|\n")
            
            for _, row in strong_correlations.head(15).iterrows():
                f.write(f"| {row['feature1']} | {row['feature2']} | {row['correlation']:.4f} |\n")
            
            f.write("\n![Correlation Heatmap](correlation_heatmap.png)\n")
        
        # Feature distributions
        f.write("\n## Feature Distributions\n\n")
        f.write("![Feature Distributions](feature_distributions.png)\n")
        
        # Feature vs target plots
        if 'prediction_correct' in df.columns:
            f.write("\n## Features vs Target Variable\n\n")
            f.write("![Features vs Target](features_vs_target.png)\n")
        
        # Conclusion
        f.write("\n## Summary\n\n")
        f.write("This report provides a comprehensive overview of the Polymarket election markets dataset. ")
        
        if 'prediction_correct' in df.columns:
            correct_pct = df['prediction_correct'].mean() * 100
            f.write(f"The markets show an overall prediction accuracy of {correct_pct:.2f}%. ")
        
        f.write("Key metrics and their distributions reveal the characteristics of these prediction markets, ")
        f.write("while the correlation analysis highlights the relationships between different features and their impact on prediction accuracy.\n")
    
    print(f"Summary report generated at {report_path}")

def main(file_path):
    """
    Main function to run the summary statistics analysis.
    
    Args:
        file_path: Path to the CSV file
    """
    try:
        # Load the data
        df = load_data(file_path)
        
        if df.empty:
            print("Error: Empty dataframe. Please check your input file.")
            return
            
        # Make sure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Basic statistics - wrap each step in try-except to continue even if one fails
        try:
            print("Calculating basic statistics...")
            stats_df = get_basic_stats(df)
            stats_df.to_csv(f"{output_dir}/basic_statistics.csv")
        except Exception as e:
            print(f"Error calculating basic statistics: {e}")
            stats_df = pd.DataFrame()
        
        # Target variable analysis
        try:
            print("Analyzing target variable...")
            target_analysis = target_variable_analysis(df)
        except Exception as e:
            print(f"Error analyzing target variable: {e}")
            target_analysis = {}
        
        # Feature distributions
        try:
            print("Generating feature distribution plots...")
            feature_distributions(df)
        except Exception as e:
            print(f"Error generating feature distributions: {e}")
        
        # Correlation analysis
        try:
            print("Performing correlation analysis...")
            strong_correlations = correlation_analysis(df)
            if not strong_correlations.empty:
                strong_correlations.to_csv(f"{output_dir}/strong_correlations.csv", index=False)
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            strong_correlations = pd.DataFrame()
        
        # Analysis by categories
        if 'event_electionType' in df.columns:
            try:
                print("Analyzing by election type...")
                market_analysis_by_category(df, 'event_electionType')
            except Exception as e:
                print(f"Error analyzing by election type: {e}")
        
        if 'event_countryName' in df.columns:
            try:
                print("Analyzing by country...")
                market_analysis_by_category(df, 'event_countryName')
            except Exception as e:
                print(f"Error analyzing by country: {e}")
        
        # Feature vs target plots
        try:
            print("Creating feature vs target plots...")
            feature_vs_target_plots(df)
        except Exception as e:
            print(f"Error creating feature vs target plots: {e}")
        
        # Generate report
        try:
            print("Generating comprehensive report...")
            generate_report(df, stats_df, target_analysis, strong_correlations)
        except Exception as e:
            print(f"Error generating report: {e}")
        
        print(f"Summary statistics analysis complete. Results saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate summary statistics for Polymarket election markets data")
    parser.add_argument("--input", default="final_election_results.csv", help="Path to the input CSV file")
    args = parser.parse_args()
    
    main(args.input)