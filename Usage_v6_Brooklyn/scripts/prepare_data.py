import os
import pandas as pd
import glob
from tqdm import tqdm
import time
import sys

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

def process_participant_data(folder):
    """
    Process a single participant's data - remove DateTime column only.
    
    Args:
        folder: Path to participant folder
        
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
            
        print(f"Successfully processed data for Participant {participant_id}")
        return True
    
    except Exception as e:
        print(f"Error processing participant {participant_id}: {str(e)}")
        return False

def main():
    start_time = time.time()
    
    # Base path for participant data
    participant_data_path = r'C:\Users\xy2593\Desktop\CUSP-GX-9000 Guided Study\Data\ParticipantData'
    
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
    print(f"Average time per participant: {elapsed_time/max(successful, 1):.1f}s")
    print("==============================")

if __name__ == "__main__":
    main() 