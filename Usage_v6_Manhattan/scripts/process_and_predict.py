import os
import pandas as pd
import numpy as np
import torch
import glob
from tqdm import tqdm
import sys
import time
from sklearn.preprocessing import StandardScaler
from scipy import signal

# Add the models path to system path to import the modules
script_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(os.path.dirname(script_dir), 'models')
sys.path.append(models_path)

try:
    from wrist_model_v6_3class import WristLSTM
    from wrist_data_loader_v6_3class import WristDataset
    from metrics_logger_3class import MetricsLogger
except ImportError as e:
    print(f"Error importing model modules: {e}")
    print(f"Looking for modules in: {models_path}")
    print("Make sure the model files exist in the correct location.")
    sys.exit(1)

def remove_datetime_column(input_file, output_file):
    """
    Read a CSV file, remove the DateTime column, and save the result to a new file.
    
    Args:
        input_file: Path to the input CSV file with DateTime column
        output_file: Path to save the output CSV file without DateTime column
    
    Returns:
        Pandas DataFrame of the processed data
    """
    try:
        # Read input file
        df = pd.read_csv(input_file)
        
        # Check if DateTime column exists
        if 'DateTime' not in df.columns:
            print(f"Warning: No DateTime column found in {input_file}")
            return df
        
        # Remove DateTime column
        df_without_datetime = df.drop('DateTime', axis=1)
        
        # Save to new file
        df_without_datetime.to_csv(output_file, index=False)
        print(f"Created {output_file}")
        
        return df_without_datetime
    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None

def prepare_data_for_model(df, file_path=None):
    """
    Prepare a DataFrame for model prediction by ensuring it has the required features.
    
    Args:
        df: Pandas DataFrame with participant data
        file_path: Optional path to the source file (used to extract subject ID if needed)
        
    Returns:
        Processed DataFrame ready for model input
    """
    # Required features
    required_features = ['ACC_x', 'ACC_y', 'ACC_z', 'EDA', 'TEMP', 'BVP',
                        'EDA_phasic', 'EDA_tonic']
    # Stats features should be included
    stats_features = [col for col in df.columns if any(x in col for x in 
                     ['_mean', '_std', '_min', '_max', '_slope'])]
    
    # Check if all required features exist
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        print(f"Warning: Missing required features: {missing_features}")
        return None
    
    # Ensure subject column exists
    if 'subject' not in df.columns:
        # Try to extract subject from index
        if 'Unnamed: 0' in df.columns and df['Unnamed: 0'].str.contains('P', case=False).any():
            # Extract subject ID from the index column
            df['subject'] = df['Unnamed: 0'].str.extract(r'P(\d+)').astype(int)
        else:
            # Try to extract from filename if available
            if file_path and os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                if filename.startswith('P') and '_' in filename:
                    subject_id = int(filename.split('_')[0][1:])
                    df['subject'] = subject_id
                else:
                    print("Warning: Could not determine subject ID")
                    df['subject'] = 0
            else:
                print("Warning: Could not determine subject ID")
                df['subject'] = 0
    
    # Create a placeholder label column for prediction (will be replaced)
    if 'label' not in df.columns:
        df['label'] = 0
    
    # Select only the features needed for the model
    all_features = required_features + stats_features + ['subject', 'label']
    df_filtered = df[all_features]
    
    # Normalize features
    scaler = StandardScaler()
    feature_cols = required_features + stats_features
    df_filtered[feature_cols] = scaler.fit_transform(df_filtered[feature_cols])
    
    return df_filtered

def load_model(model_path):
    """
    Load the trained model from the given path.
    
    Args:
        model_path: Path to the model file (.pth)
        
    Returns:
        Loaded model and device
    """
    try:
        # Check for CUDA availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model with updated architecture to match the saved model
        model = WristLSTM(
            input_size=33,  # Fixed size based on the model architecture
            hidden_size=192,  # Increased from 128 to 192 based on error message
            num_layers=2,
            num_classes=3,
            dropout=0.3,
            attention_dropout=0.3,
            num_heads=4,
            feature_expansion=5  # Set to 5 to match the 165 dimension (33 * 5 = 165)
        )
        
        # Load state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)  # Move model to GPU if available
        model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded model from {model_path}")
        print(f"Using device: {device}")
        
        return model, device
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def predict(model, df, device):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Loaded PyTorch model
        df: Processed DataFrame
        device: Device to run inference on (CPU or CUDA)
        
    Returns:
        DataFrame with predictions
    """
    # Create dataset
    dataset = WristDataset(df)
    
    # Create lists to store predictions
    predictions = []
    subjects = []
    
    # Use batch processing for faster inference
    batch_size = 1024  # Adjust based on your GPU memory
    n_samples = len(dataset)
    
    # Create batches
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Making predictions", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Process each example in the batch
            batch_features = []
            batch_subjects = []
            
            for i in range(start_idx, end_idx):
                features, _, subject = dataset[i]
                batch_features.append(features)
                batch_subjects.append(subject)
            
            # Stack features into a batch
            if batch_features:
                batch_features_tensor = torch.stack(batch_features).to(device)
                
                # Get predictions
                outputs = model(batch_features_tensor)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()
                
                # Collect results
                predictions.extend(batch_preds.tolist())
                subjects.extend([s.item() for s in batch_subjects])
    
    # Map predictions back to original labels
    # 0 → 1 (Baseline), 1 → 2 (Stress), 2 → 3/4 (Amusement/Meditation)
    label_mapping = {0: 1, 1: 2, 2: 3}  # Using 3 for Amusement/Meditation
    mapped_predictions = [label_mapping[pred] for pred in predictions]
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'subject': subjects,
        'predicted_label': mapped_predictions,
        'raw_prediction': predictions
    })
    
    return result_df

def process_participant(folder, model, device):
    """
    Process a single participant's data.
    
    Args:
        folder: Path to participant folder
        model: Loaded PyTorch model
        device: Device to run inference on
        
    Returns:
        True if successful, False otherwise
    """
    try:
        participant_id = os.path.basename(folder).split('#')[1]
        
        # Look for the prepared file with DateTime
        with_datetime_file = os.path.join(folder, f'P{participant_id}_Prepared_withDatetime.csv')
        
        if not os.path.exists(with_datetime_file):
            print(f"Warning: Could not find file {with_datetime_file}")
            return False
        
        # Output file path
        without_datetime_file = os.path.join(folder, f'P{participant_id}_Prepared.csv')
        
        # Process file - remove DateTime column
        df_processed = remove_datetime_column(with_datetime_file, without_datetime_file)
        if df_processed is None:
            return False
        
        # Prepare data for model
        df_model_ready = prepare_data_for_model(df_processed, without_datetime_file)
        if df_model_ready is None:
            return False
        
        # Make predictions
        prediction_df = predict(model, df_model_ready, device)
        
        # Save predictions
        prediction_file = os.path.join(folder, f'P{participant_id}_predictions.csv')
        prediction_df.to_csv(prediction_file, index=False)
        print(f"Saved predictions to {prediction_file}")
        
        # Create a summary of predictions
        pred_counts = prediction_df['predicted_label'].value_counts().sort_index()
        print(f"\nPrediction summary for Participant {participant_id}:")
        for label, count in pred_counts.items():
            label_name = {1: "Baseline", 2: "Stress", 3: "Amusement/Meditation"}[label]
            percentage = (count / len(prediction_df)) * 100
            print(f"  {label_name}: {count} samples ({percentage:.1f}%)")
        
        return True
    
    except Exception as e:
        print(f"Error processing participant: {str(e)}")
        return False

def main():
    start_time = time.time()
    
    # Base paths
    participant_data_path = r'E:\NYU\academic\Y2S2\CUSP-GX-7133\DataManhattan\ParticipantData'
    
    # 使用与导入模块相同的方法构建模型路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), 'models')
    model_path = os.path.join(models_dir, 'best_model_v6.pth')
    
    print(f"Looking for model at: {model_path}")
    
    # Load the model
    model, device = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Process all participant folders
    participant_folders = glob.glob(os.path.join(participant_data_path, 'Participant#*'))
    
    successful = 0
    total = len(participant_folders)
    
    for folder in tqdm(participant_folders, desc="Processing participants"):
        if process_participant(folder, model, device):
            successful += 1
    
    # Print summary statistics
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n===== Processing Summary =====")
    print(f"Successfully processed {successful}/{total} participants")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Average time per participant: {elapsed_time/max(successful, 1):.1f}s")
    print("==============================")

if __name__ == "__main__":
    main() 