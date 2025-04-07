import os
import pandas as pd
import glob
from tqdm import tqdm
import time

def add_labels_to_datetime_csv(prediction_file, datetime_file, output_file):
    """
    Add predicted labels from the predictions CSV to the withDatetime CSV file.
    
    Args:
        prediction_file: Path to the predictions CSV file
        datetime_file: Path to the CSV file with DateTime column
        output_file: Path to save the output CSV file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read both files
        pred_df = pd.read_csv(prediction_file)
        datetime_df = pd.read_csv(datetime_file)
        
        # Verify the prediction file has the required column
        if 'predicted_label' not in pred_df.columns:
            print(f"Error: 'predicted_label' column not found in {prediction_file}")
            return False
        
        # Check that the number of rows matches
        if len(pred_df) != len(datetime_df):
            print(f"Warning: Number of rows does not match between files:")
            print(f"  {prediction_file}: {len(pred_df)} rows")
            print(f"  {datetime_file}: {len(datetime_df)} rows")
            print(f"  Will align by index, but this might lead to misaligned labels")
        
        # Get the labels
        labels = pred_df['predicted_label'].values
        
        # Add labels to the withDatetime DataFrame
        datetime_df['predicted_label'] = labels[:len(datetime_df)] if len(labels) > len(datetime_df) else labels
        
        # Map numeric labels to descriptive text for easier interpretation
        label_map = {
            1: "Baseline",
            2: "Stress", 
            3: "Amusement/Meditation"
        }
        datetime_df['emotional_state'] = datetime_df['predicted_label'].map(label_map)
        
        # Save to new file
        datetime_df.to_csv(output_file, index=False)
        print(f"Created {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return False

def process_participant_data(folder):
    """
    Process a single participant's data - add predicted labels to withDatetime file.
    
    Args:
        folder: Path to participant folder
        
    Returns:
        True if successful, False otherwise
    """
    try:
        participant_id = os.path.basename(folder).split('#')[1]
        
        # Define file paths
        prediction_file = os.path.join(folder, f'P{participant_id}_predictions.csv')
        datetime_file = os.path.join(folder, f'P{participant_id}_Prepared_withDatetime.csv')
        output_file = os.path.join(folder, f'P{participant_id}_Prepared_withDatetime_labeled.csv')
        
        # Check if required files exist
        if not os.path.exists(prediction_file):
            print(f"Warning: Could not find prediction file {prediction_file}")
            return False
            
        if not os.path.exists(datetime_file):
            print(f"Warning: Could not find withDatetime file {datetime_file}")
            return False
        
        # Process files
        success = add_labels_to_datetime_csv(prediction_file, datetime_file, output_file)
        
        if success:
            print(f"Successfully added labels for Participant {participant_id}")
            return True
        else:
            print(f"Failed to add labels for Participant {participant_id}")
            return False
    
    except Exception as e:
        print(f"Error processing participant {participant_id}: {str(e)}")
        return False

def main():
    start_time = time.time()
    
    # Base path for participant data
    participant_data_path = r'E:\NYU\academic\Y2S2\CUSP-GX-7133\DataManhattan\ParticipantData'
    
    # Process all participant folders
    participant_folders = glob.glob(os.path.join(participant_data_path, 'Participant#*'))
    
    successful = 0
    total = len(participant_folders)
    
    for folder in tqdm(participant_folders, desc="Processing participants"):
        if process_participant_data(folder):
            successful += 1
    
    # Print summary statistics
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\n===== Processing Summary =====")
    print(f"Successfully processed {successful}/{total} participants")
    print(f"Total time: {int(minutes)}m {int(seconds)}s")
    print(f"Output files: P*_Prepared_withDatetime_labeled.csv")
    print("==============================")

if __name__ == "__main__":
    main() 