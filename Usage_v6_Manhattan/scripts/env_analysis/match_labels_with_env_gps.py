import os
import pandas as pd
import glob
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from collections import Counter

def match_labels_with_env_data(labeled_data, env_data, participant_id):
    """
    Match emotion labels with environmental data based on time windows.
    Prioritizes stress (class 2) and relaxation (class 3) if they appear
    in at least 20% of the data points in a window.
    
    Args:
        labeled_data: DataFrame containing the labeled data (64Hz)
        env_data: DataFrame containing the environmental data (0.2Hz)
        participant_id: Participant ID for filtering env_data
        
    Returns:
        DataFrame with matched data
    """
    print(f"Matching data for Participant {participant_id}")
    
    # Convert participant_id to numeric format (remove any leading zeros)
    participant_id_numeric = int(participant_id)
    
    # Filter env_data for the current participant
    participant_env_data = env_data[env_data['Participant#'] == participant_id_numeric].copy()
    
    if len(participant_env_data) == 0:
        print(f"No environmental data found for Participant {participant_id} (numeric: {participant_id_numeric})")
        return None
    else:
        print(f"Found {len(participant_env_data)} environmental data points for Participant {participant_id}")
    
    # Ensure we have datetime objects for comparison
    if 'DateTime' not in labeled_data.columns:
        print(f"Error: DateTime column not found in labeled data")
        return None
    
    # Convert string datetime to datetime objects if needed
    if isinstance(labeled_data['DateTime'].iloc[0], str):
        labeled_data['DateTime'] = pd.to_datetime(labeled_data['DateTime'])
    
    # Create a datetime column in env_data by combining Date and Time
    if 'DateTime' not in participant_env_data.columns:
        participant_env_data['DateTime'] = pd.to_datetime(
            participant_env_data['Date'] + ' ' + participant_env_data['Time']
        )
    
    # Initialize lists to store the matched data
    matched_rows = []
    
    # Define threshold for prioritizing stress and relaxation classes
    # PRIORITY_THRESHOLD = 0.20  # 20%
    PRIORITY_THRESHOLD = 0.50  # 50%
    
    # For each env timestamp, find labels within ±2.5s window
    for _, env_row in tqdm(participant_env_data.iterrows(), 
                          total=len(participant_env_data),
                          desc=f"Processing P{participant_id} env data points"):
        
        env_time = env_row['DateTime']
        window_start = env_time - timedelta(seconds=2.5)
        window_end = env_time + timedelta(seconds=2.5)
        
        # Get labels in the window
        window_labels = labeled_data[
            (labeled_data['DateTime'] >= window_start) & 
            (labeled_data['DateTime'] <= window_end)
        ]
        
        # If no labels found in window, continue to next env timestamp
        if len(window_labels) == 0:
            continue
            
        # Count labels and calculate percentages
        if 'predicted_label' in window_labels.columns:
            label_counts = Counter(window_labels['predicted_label'])
            total_labels = sum(label_counts.values())
            
            if total_labels > 0:
                # Calculate percentages for each label
                label_percentages = {label: count/total_labels for label, count in label_counts.items()}
                
                # Initialize variables for decision logic
                selected_label = None
                decision_reason = ""
                
                # Check if stress (2) or relaxation (3) exceed threshold
                stress_percentage = label_percentages.get(2, 0)
                relax_percentage = label_percentages.get(3, 0)
                
                # Prioritization logic
                if stress_percentage >= PRIORITY_THRESHOLD and relax_percentage >= PRIORITY_THRESHOLD:
                    # Both exceed threshold, choose the one with higher percentage
                    if stress_percentage >= relax_percentage:
                        selected_label = 2
                        decision_reason = f"Both stress ({stress_percentage:.1%}) and relaxation ({relax_percentage:.1%}) exceed threshold, stress higher"
                    else:
                        selected_label = 3
                        decision_reason = f"Both stress ({stress_percentage:.1%}) and relaxation ({relax_percentage:.1%}) exceed threshold, relaxation higher"
                elif stress_percentage >= PRIORITY_THRESHOLD:
                    selected_label = 2
                    decision_reason = f"Stress ({stress_percentage:.1%}) exceeds threshold"
                elif relax_percentage >= PRIORITY_THRESHOLD:
                    selected_label = 3
                    decision_reason = f"Relaxation ({relax_percentage:.1%}) exceeds threshold"
                else:
                    # No priority class exceeds threshold, use most common
                    selected_label = label_counts.most_common(1)[0][0]
                    decision_reason = "No priority class exceeds threshold, using most common"
                
                # Map numeric labels to descriptive text
                label_map = {
                    1: "Baseline",
                    2: "Stress", 
                    3: "Amusement/Meditation"
                }
                emotional_state = label_map.get(selected_label, "Unknown")
                
                # Create a new row with env data and the selected label
                new_row = env_row.copy()
                new_row['BiLSTM_predicted_label'] = selected_label
                new_row['BiLSTM_emotional_state'] = emotional_state
                new_row['window_size'] = len(window_labels)
                new_row['decision_reason'] = decision_reason
                
                # Add percentages for all classes
                for label, percentage in label_percentages.items():
                    new_row[f'class_{label}_percentage'] = percentage
                
                matched_rows.append(new_row)
    
    # Create DataFrame from the matched rows
    if matched_rows:
        result_df = pd.DataFrame(matched_rows)
        print(f"Found {len(result_df)} matching windows for Participant {participant_id}")
        
        # Print statistics about selected labels
        if 'BiLSTM_predicted_label' in result_df.columns:
            label_counts = Counter(result_df['BiLSTM_predicted_label'])
            print(f"Label distribution after prioritization:")
            for label, count in label_counts.items():
                percentage = count / len(result_df) * 100
                print(f"  Class {label}: {count} ({percentage:.1f}%)")
        
        return result_df
    else:
        print(f"No matching windows found for Participant {participant_id}")
        return None

def process_all_participants():
    """
    Process all participants' data and create a matched dataset
    """
    start_time = time.time()
    
    # Base path for participant data
    participant_data_path = r'E:\NYU\academic\Y2S2\CUSP-GX-7133\DataManhattan\ParticipantData'
    
    # Path to the env+GPS data file
    env_gps_file = os.path.join(r'E:\NYU\academic\Y2S2\CUSP-GX-7133\DataManhattan', 
                               'All_Participant_Process', 
                               'Man_Participant_Labeled_BiLSTM_withEnvGPS.csv')
    
    # Check if env+GPS file exists
    if not os.path.exists(env_gps_file):
        print(f"Error: Could not find environmental data file {env_gps_file}")
        return
    
    # Read the env+GPS data
    print(f"Reading environmental and GPS data from {env_gps_file}")
    env_data = pd.read_csv(env_gps_file)
    
    # Print unique participant IDs in env data
    unique_participants = env_data['Participant#'].unique()
    print(f"Found {len(unique_participants)} unique participants in env data: {unique_participants}")
    
    # Get all participant folders
    participant_folders = glob.glob(os.path.join(participant_data_path, 'Participant#*'))
    
    # Lists to store results
    all_matched_data = []
    successful = 0
    total = len(participant_folders)
    
    # Process each participant
    for folder in tqdm(participant_folders, desc="Processing participants"):
        participant_id = os.path.basename(folder).split('#')[1]
        
        # Define the labeled data file path
        labeled_file = os.path.join(folder, f'P{participant_id}_Prepared_withDatetime_labeled.csv')
        
        # Check if labeled file exists
        if not os.path.exists(labeled_file):
            print(f"Warning: Could not find labeled file {labeled_file}")
            continue
        
        # Read labeled data
        labeled_data = pd.read_csv(labeled_file)
        
        # Match labels with env data
        matched_data = match_labels_with_env_data(labeled_data, env_data, participant_id)
        
        # If matching was successful, add to results
        if matched_data is not None and len(matched_data) > 0:
            all_matched_data.append(matched_data)
            successful += 1
    
    # Combine all matched data
    if all_matched_data:
        final_df = pd.concat(all_matched_data, ignore_index=True)
        
        # Print overall statistics about selected labels
        print("\nOverall label distribution after prioritization:")
        label_counts = Counter(final_df['BiLSTM_predicted_label'])
        for label, count in label_counts.items():
            percentage = count / len(final_df) * 100
            print(f"  Class {label}: {count} ({percentage:.1f}%)")
        
        # Save the final result
        output_file = os.path.join(r'E:\NYU\academic\Y2S2\CUSP-GX-7133\DataManhattan', 
                                  'All_Participant_Process', 
                                  'Man_All_Participant_Labeled_BiLSTM_withEnvGPS_prioritized_50.csv')
        
        final_df.to_csv(output_file, index=False)
        print(f"\nCreated combined file: {output_file}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("No matching data found")
    
    # Print summary statistics
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\n===== Processing Summary =====")
    print(f"Successfully processed {successful}/{total} participants")
    print(f"Total time: {int(minutes)}m {int(seconds)}s")
    print("==============================")

if __name__ == "__main__":
    process_all_participants() 