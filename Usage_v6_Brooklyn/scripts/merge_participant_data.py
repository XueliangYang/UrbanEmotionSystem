import os
import pandas as pd
import numpy as np
import glob
import datetime
from scipy import signal
from tqdm import tqdm
import traceback

def convert_timestamp(ts):
    """Convert timestamp to readable datetime format."""
    try:
        dt = datetime.datetime.fromtimestamp(ts / 1_000_000)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
    except (OSError, ValueError):
        return None

def decompose_eda(eda_signal):
    """Decompose EDA signal into phasic and tonic components using a moving average approach."""
    eda_series = pd.Series(eda_signal)
    window_size = 20  # Adjust based on your sampling rate
    
    # Calculate moving average for tonic component
    eda_tonic = eda_series.rolling(window=window_size, center=True).mean()
    # Calculate phasic component as the difference
    eda_phasic = eda_series - eda_tonic
    
    # Fill NaN values at the edges
    eda_tonic = eda_tonic.fillna(method='bfill').fillna(method='ffill')
    eda_phasic = eda_phasic.fillna(method='bfill').fillna(method='ffill')
    
    return eda_phasic.values, eda_tonic.values

def calculate_stats(signal_name, signal_data, window_size=20):
    """Calculate statistical features for a given signal."""
    signal_series = pd.Series(signal_data)
    return pd.DataFrame({
        f'{signal_name}_mean': signal_series.rolling(window=window_size, center=True).mean(),
        f'{signal_name}_std': signal_series.rolling(window=window_size, center=True).std(),
        f'{signal_name}_min': signal_series.rolling(window=window_size, center=True).min(),
        f'{signal_name}_max': signal_series.rolling(window=window_size, center=True).max(),
        f'{signal_name}_slope': (signal_series.diff(window_size) / window_size)
    })

def process_participant_data(participant_folder):
    """Process data for a single participant folder."""
    try:
        participant_id = os.path.basename(participant_folder).split('#')[1]
        print(f"Processing Participant {participant_id}...")
        
        # Find the rawData_to_csv folder (could be with or without date suffix)
        csv_folder = None
        for folder in os.listdir(participant_folder):
            if folder.startswith('rawData_to_csv'):
                csv_folder = os.path.join(participant_folder, folder)
                break
        
        if not csv_folder:
            print(f"No rawData_to_csv folder found for Participant {participant_id}")
            return None
        
        # Check for required CSV files
        acc_file = os.path.join(csv_folder, f'P{participant_id}_ACC.csv')
        bvp_file = os.path.join(csv_folder, f'P{participant_id}_BVP.csv')
        eda_file = os.path.join(csv_folder, f'P{participant_id}_EDA.csv')
        temp_file = os.path.join(csv_folder, f'P{participant_id}_TEMP.csv')
        
        required_files = [acc_file, bvp_file, eda_file, temp_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"Missing required files for Participant {participant_id}: {[os.path.basename(f) for f in missing_files]}")
            return None
        
        # Load the CSV files
        print(f"Loading CSV files for Participant {participant_id}...")
        acc_df = pd.read_csv(acc_file)
        bvp_df = pd.read_csv(bvp_file)
        eda_df = pd.read_csv(eda_file)
        temp_df = pd.read_csv(temp_file)
        
        # Detect and standardize column names
        acc_cols = acc_df.columns.tolist()
        bvp_cols = bvp_df.columns.tolist()
        eda_cols = eda_df.columns.tolist()
        temp_cols = temp_df.columns.tolist()
        
        # Map potential timestamp column names
        timestamp_cols = ['unix_timestamp', 'timestamp', 'time', 'ts']
        
        # Find timestamp column for each dataframe
        acc_ts_col = next((col for col in timestamp_cols if col in acc_cols), None)
        bvp_ts_col = next((col for col in timestamp_cols if col in bvp_cols), None)
        eda_ts_col = next((col for col in timestamp_cols if col in eda_cols), None)
        temp_ts_col = next((col for col in timestamp_cols if col in temp_cols), None)
        
        # Check if all timestamp columns were found
        if not all([acc_ts_col, bvp_ts_col, eda_ts_col, temp_ts_col]):
            missing_ts = []
            if not acc_ts_col: missing_ts.append('ACC')
            if not bvp_ts_col: missing_ts.append('BVP')
            if not eda_ts_col: missing_ts.append('EDA')
            if not temp_ts_col: missing_ts.append('TEMP')
            print(f"Could not find timestamp columns in {', '.join(missing_ts)} files for Participant {participant_id}")
            return None
        
        # Rename timestamp columns to a standard name
        acc_df.rename(columns={acc_ts_col: 'timestamp'}, inplace=True)
        bvp_df.rename(columns={bvp_ts_col: 'timestamp'}, inplace=True)
        eda_df.rename(columns={eda_ts_col: 'timestamp'}, inplace=True)
        temp_df.rename(columns={temp_ts_col: 'timestamp'}, inplace=True)
        
        # Map potential data column names
        acc_x_cols = ['x', 'acc_x', 'ACC_x', 'X']
        acc_y_cols = ['y', 'acc_y', 'ACC_y', 'Y']
        acc_z_cols = ['z', 'acc_z', 'ACC_z', 'Z']
        bvp_cols = ['bvp', 'BVP']
        eda_cols = ['eda', 'EDA']
        temp_cols = ['temperature', 'temp', 'TEMP']
        
        # Find data columns for each dataframe
        acc_x_col = next((col for col in acc_x_cols if col in acc_df.columns), None)
        acc_y_col = next((col for col in acc_y_cols if col in acc_df.columns), None)
        acc_z_col = next((col for col in acc_z_cols if col in acc_df.columns), None)
        bvp_col = next((col for col in bvp_cols if col in bvp_df.columns), None)
        eda_col = next((col for col in eda_cols if col in eda_df.columns), None)
        temp_col = next((col for col in temp_cols if col in temp_df.columns), None)
        
        # Check if all data columns were found
        missing_data_cols = []
        if not all([acc_x_col, acc_y_col, acc_z_col]): missing_data_cols.append('ACC')
        if not bvp_col: missing_data_cols.append('BVP')
        if not eda_col: missing_data_cols.append('EDA')
        if not temp_col: missing_data_cols.append('TEMP')
        
        if missing_data_cols:
            print(f"Could not find data columns in {', '.join(missing_data_cols)} files for Participant {participant_id}")
            return None
        
        # Ensure timestamps are sorted
        acc_df = acc_df.sort_values('timestamp')
        bvp_df = bvp_df.sort_values('timestamp')
        eda_df = eda_df.sort_values('timestamp')
        temp_df = temp_df.sort_values('timestamp')
        
        # Convert to numeric if needed
        for df, cols in [
            (acc_df, ['timestamp', acc_x_col, acc_y_col, acc_z_col]),
            (bvp_df, ['timestamp', bvp_col]),
            (eda_df, ['timestamp', eda_col]),
            (temp_df, ['timestamp', temp_col])
        ]:
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN values after conversion
            df.dropna(subset=cols, inplace=True)
            
            # Reset index after sorting and dropping NAs
            df.reset_index(drop=True, inplace=True)
        
        # Find common time range
        start_time = max(
            acc_df['timestamp'].min(),
            bvp_df['timestamp'].min(),
            eda_df['timestamp'].min(),
            temp_df['timestamp'].min()
        )
        
        end_time = min(
            acc_df['timestamp'].max(),
            bvp_df['timestamp'].max(),
            eda_df['timestamp'].max(),
            temp_df['timestamp'].max()
        )
        
        print(f"Common time range: {convert_timestamp(start_time)} to {convert_timestamp(end_time)}")
        
        # Create a common time base at the highest sampling rate (usually BVP at 64Hz)
        # Find the signal with the highest sampling rate by checking density
        signal_densities = {
            'ACC': len(acc_df) / max(1, (acc_df['timestamp'].max() - acc_df['timestamp'].min())),
            'BVP': len(bvp_df) / max(1, (bvp_df['timestamp'].max() - bvp_df['timestamp'].min())),
            'EDA': len(eda_df) / max(1, (eda_df['timestamp'].max() - eda_df['timestamp'].min())),
            'TEMP': len(temp_df) / max(1, (temp_df['timestamp'].max() - temp_df['timestamp'].min()))
        }
        
        highest_rate_signal = max(signal_densities, key=signal_densities.get)
        print(f"Highest sampling rate signal: {highest_rate_signal}")
        
        if highest_rate_signal == 'ACC':
            reference_df = acc_df
        elif highest_rate_signal == 'BVP':
            reference_df = bvp_df
        elif highest_rate_signal == 'EDA':
            reference_df = eda_df
        else:
            reference_df = temp_df
        
        # Filter reference dataframe to common time range
        reference_df = reference_df[(reference_df['timestamp'] >= start_time) & 
                                  (reference_df['timestamp'] <= end_time)]
        
        # If reference_df is empty after filtering, there's an issue with the time ranges
        if len(reference_df) == 0:
            print(f"Error: No overlapping time range for Participant {participant_id}")
            return None
        
        common_timestamps = reference_df['timestamp'].values
        
        # Create DateTime column from timestamps
        print(f"Creating DateTime values for {len(common_timestamps)} timestamps...")
        datetime_values = [convert_timestamp(ts) for ts in common_timestamps]
        
        # Prepare merged dataframe with common timestamps and DateTime
        merged_df = pd.DataFrame({
            'timestamp': common_timestamps,
            'DateTime': datetime_values
        })
        
        # Filter out rows where DateTime conversion failed
        merged_df = merged_df.dropna(subset=['DateTime'])
        
        # Interpolate ACC data to common timestamps
        print(f"Interpolating ACC data...")
        merged_df['ACC_x'] = np.interp(merged_df['timestamp'].values, acc_df['timestamp'].values, acc_df[acc_x_col].values)
        merged_df['ACC_y'] = np.interp(merged_df['timestamp'].values, acc_df['timestamp'].values, acc_df[acc_y_col].values)
        merged_df['ACC_z'] = np.interp(merged_df['timestamp'].values, acc_df['timestamp'].values, acc_df[acc_z_col].values)
        
        # Interpolate BVP data
        print(f"Interpolating BVP data...")
        merged_df['BVP'] = np.interp(merged_df['timestamp'].values, bvp_df['timestamp'].values, bvp_df[bvp_col].values)
        
        # Interpolate EDA data
        print(f"Interpolating EDA data...")
        merged_df['EDA'] = np.interp(merged_df['timestamp'].values, eda_df['timestamp'].values, eda_df[eda_col].values)
        
        # Interpolate TEMP data
        print(f"Interpolating TEMP data...")
        merged_df['TEMP'] = np.interp(merged_df['timestamp'].values, temp_df['timestamp'].values, temp_df[temp_col].values)
        
        # Decompose EDA into phasic and tonic components
        print(f"Decomposing EDA signal...")
        eda_phasic, eda_tonic = decompose_eda(merged_df['EDA'].values)
        merged_df['EDA_phasic'] = eda_phasic
        merged_df['EDA_tonic'] = eda_tonic
        
        # Calculate statistical features
        print("Calculating statistical features...")
        window_size = min(20, len(merged_df) // 2)  # Ensure window size is not too large for the data
        window_size = max(window_size, 3)  # Ensure window size is at least 3
        
        for signal_name in ['BVP', 'EDA', 'TEMP', 'EDA_phasic', 'EDA_tonic']:
            print(f"Processing {signal_name} statistics...")
            stats_df = calculate_stats(signal_name, merged_df[signal_name], window_size=window_size)
            for col in stats_df.columns:
                merged_df[col] = stats_df[col].values
        
        # Add subject ID
        merged_df['subject'] = participant_id
        
        # Handle NaN values
        merged_df = merged_df.fillna(method='bfill').fillna(method='ffill')
        
        # Make sure we have all required columns
        required_columns = [
            'DateTime', 'ACC_x', 'ACC_y', 'ACC_z', 'EDA', 'TEMP', 'BVP', 
            'EDA_phasic', 'EDA_tonic', 'BVP_mean', 'BVP_std', 'BVP_min', 
            'BVP_max', 'BVP_slope', 'EDA_mean', 'EDA_std', 'EDA_min', 
            'EDA_max', 'EDA_slope', 'TEMP_mean', 'TEMP_std', 'TEMP_min', 
            'TEMP_max', 'TEMP_slope', 'EDA_phasic_mean', 'EDA_phasic_std', 
            'EDA_phasic_min', 'EDA_phasic_max', 'EDA_phasic_slope', 
            'EDA_tonic_mean', 'EDA_tonic_std', 'EDA_tonic_min', 
            'EDA_tonic_max', 'EDA_tonic_slope', 'subject'
        ]
        
        missing_columns = [col for col in required_columns if col not in merged_df.columns]
        if missing_columns:
            print(f"Warning: Missing required columns for Participant {participant_id}: {missing_columns}")
        
        # Get columns in the requested order
        output_columns = [col for col in required_columns if col in merged_df.columns]
        
        # Drop timestamp column as it's not in the required list
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.drop(columns=['timestamp'])
        
        # Reorder columns to match required format
        merged_df = merged_df[output_columns]
        
        # Save the merged file
        output_file = os.path.join(participant_folder, f'P{participant_id}_Prepared_withDatetime.csv')
        merged_df.to_csv(output_file, index=False)
        print(f"Created merged file: {output_file} with {len(merged_df)} rows and {len(merged_df.columns)} columns")
        
        return output_file
    
    except Exception as e:
        print(f"Error processing participant {os.path.basename(participant_folder)}: {str(e)}")
        traceback.print_exc()
        return None

def main():
    # Define the base directory for the participant data
    base_dir = r'C:\Users\xy2593\Desktop\CUSP-GX-9000 Guided Study\Data\ParticipantData'
    
    # Get a list of all Participant folders
    participant_folders = glob.glob(os.path.join(base_dir, 'Participant#*'))
    
    print(f"Found {len(participant_folders)} participant folders")
    
    # Process each participant folder
    results = []
    for folder in tqdm(participant_folders):
        result = process_participant_data(folder)
        if result:
            results.append(result)
    
    print(f"Successfully processed {len(results)} out of {len(participant_folders)} participants")
    print("Merged files:")
    for result in results:
        print(f"  - {result}")

if __name__ == "__main__":
    main() 