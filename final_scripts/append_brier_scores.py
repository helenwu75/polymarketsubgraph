"""
Inputes accuracy measures into the dataset to evaluate the model's performance
Use by running the script with the file path as an argument
The script will add the following columns to the dataset:
- brier_score: measures accuracy of probabilistic predictions
- log_loss: penalizes false confidence more heavily
The updated dataset will be saved back to the same file path
"""
import pandas as pd
import numpy as np

def add_key_prediction_metrics(file_path):
    # Load the dataset
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)  # Use the file_path parameter here
    
    # Check if required columns exist
    required_cols = ['closing_price', 'correct_outcome']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Convert correct_outcome to binary (1 for 'Yes', 0 for 'No')
    df['actual_outcome'] = df['correct_outcome'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 1. Brier Score - measures accuracy of probabilistic predictions
    df['brier_score'] = (df['closing_price'] - df['actual_outcome'])**2
    print(f"Average Brier Score: {df['brier_score'].mean():.4f}")
    
    # 2. Logarithmic Loss (Cross-Entropy) - penalizes false confidence more heavily
    # Clip probabilities to avoid log(0) issues
    epsilon = 1e-15
    closing_price_clipped = np.clip(df['closing_price'], epsilon, 1-epsilon)
    
    df['log_loss'] = df.apply(
        lambda row: -np.log(closing_price_clipped[row.name]) if row['actual_outcome'] == 1 
                   else -np.log(1 - closing_price_clipped[row.name]), 
        axis=1
    )
    print(f"Average Log Loss: {df['log_loss'].mean():.4f}")
    
    # Save the updated dataset back to the same file
    df.to_csv(file_path, index=False)
    print(f"Added Brier score and Log Loss metrics and updated {file_path}")
    
    return df

if __name__ == "__main__":
    # Update this file path as needed
    file_path = "mini_election_metrics_results_edit.csv"
    
    updated_df = add_key_prediction_metrics(file_path)