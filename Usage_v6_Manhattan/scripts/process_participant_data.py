import os
import fastavro
import csv
import pandas as pd
import glob
import datetime

def convert_timestamp(ts, is_nanos=False):
    """Convert timestamp to readable datetime format.
    Args:
        ts: timestamp in microseconds or nanoseconds
        is_nanos: True if timestamp is in nanoseconds, False if in microseconds
    """
    try:
        divisor = 1_000_000_000 if is_nanos else 1_000_000
        dt = datetime.datetime.fromtimestamp(ts / divisor)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
    except (OSError, ValueError):
        return None

def process_avro_file(avro_file_path, output_dir):
    # Extract the prefix from the avro file name
    file_prefix = os.path.basename(avro_file_path).replace('.avro', '')

    with open(avro_file_path, 'rb') as f:
        reader = fastavro.reader(f)
        for data in reader:
            # Accelerometer
            acc = data["rawData"]["accelerometer"]
            timestamp = [round(acc["timestampStart"] + i * (1e6 / acc["samplingFrequency"]))
                         for i in range(len(acc["x"]))]
            delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
            delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
            x_g = [val * delta_physical / delta_digital for val in acc["x"]]
            y_g = [val * delta_physical / delta_digital for val in acc["y"]]
            z_g = [val * delta_physical / delta_digital for val in acc["z"]]
            with open(os.path.join(output_dir, f'{file_prefix}_accelerometer.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["unix_timestamp", "x", "y", "z"])
                writer.writerows([[ts, x, y, z] for ts, x, y, z in zip(timestamp, x_g, y_g, z_g)])

            # Gyroscope
            gyro = data["rawData"]["gyroscope"]
            timestamp = [round(gyro["timestampStart"] + i * (1e6 / gyro["samplingFrequency"]))
                         for i in range(len(gyro["x"]))]
            delta_physical = gyro["imuParams"]["physicalMax"] - gyro["imuParams"]["physicalMin"]
            delta_digital = gyro["imuParams"]["digitalMax"] - gyro["imuParams"]["digitalMin"]
            x_dps = [val * delta_physical / delta_digital for val in gyro["x"]]
            y_dps = [val * delta_physical / delta_digital for val in gyro["y"]]
            z_dps = [val * delta_physical / delta_digital for val in gyro["z"]]
            with open(os.path.join(output_dir, f'{file_prefix}_gyroscope.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["unix_timestamp", "x", "y", "z"])
                writer.writerows([[ts, x, y, z] for ts, x, y, z in zip(timestamp, x_dps, y_dps, z_dps)])

            # EDA
            eda = data["rawData"]["eda"]
            timestamp = [round(eda["timestampStart"] + i * (1e6 / eda["samplingFrequency"]))
                         for i in range(len(eda["values"]))]
            with open(os.path.join(output_dir, f'{file_prefix}_eda.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["unix_timestamp", "eda"])
                writer.writerows([[ts, eda_val] for ts, eda_val in zip(timestamp, eda["values"])])

            # Temperature
            tmp = data["rawData"]["temperature"]
            timestamp = [round(tmp["timestampStart"] + i * (1e6 / tmp["samplingFrequency"]))
                         for i in range(len(tmp["values"]))]
            with open(os.path.join(output_dir, f'{file_prefix}_temperature.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["unix_timestamp", "temperature"])
                writer.writerows([[ts, temp] for ts, temp in zip(timestamp, tmp["values"])])

            # Tags
            tags = data["rawData"]["tags"]
            with open(os.path.join(output_dir, f'{file_prefix}_tags.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["tags_timestamp"])
                writer.writerows([[tag] for tag in tags["tagsTimeMicros"]])

            # BVP
            bvp = data["rawData"]["bvp"]
            timestamp = [round(bvp["timestampStart"] + i * (1e6 / bvp["samplingFrequency"]))
                         for i in range(len(bvp["values"]))]
            with open(os.path.join(output_dir, f'{file_prefix}_bvp.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["unix_timestamp", "bvp"])
                writer.writerows([[ts, bvp_val] for ts, bvp_val in zip(timestamp, bvp["values"])])

            # Systolic peaks
            sps = data["rawData"]["systolicPeaks"]
            with open(os.path.join(output_dir, f'{file_prefix}_systolic_peaks.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["systolic_peak_timestamp"])
                writer.writerows([[sp] for sp in sps["peaksTimeNanos"]])

            # Steps
            steps = data["rawData"]["steps"]
            timestamp = [round(steps["timestampStart"] + i * (1e6 / steps["samplingFrequency"]))
                         for i in range(len(steps["values"]))]
            with open(os.path.join(output_dir, f'{file_prefix}_steps.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["unix_timestamp", "steps"])
                writer.writerows([[ts, step] for ts, step in zip(timestamp, steps["values"])])

def merge_signal_files(csv_folder, signal_type):
    """Merge all CSV files of a specific signal type in the folder."""
    pattern = os.path.join(csv_folder, f'*_{signal_type}.csv')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Define timestamp column name based on signal type
    timestamp_column = {
        'tags': 'tags_timestamp',
        'systolic_peaks': 'systolic_peak_timestamp'
    }.get(signal_type, 'unix_timestamp')
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        # Rename timestamp column to a common name for sorting
        if timestamp_column != 'unix_timestamp':
            df = df.rename(columns={timestamp_column: 'unix_timestamp'})
        dfs.append(df)
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.sort_values('unix_timestamp').drop_duplicates()
        
        # Add readable timestamp column
        is_nanos = signal_type == 'systolic_peaks'  # systolic peaks are in nanoseconds
        merged_df['datetime'] = merged_df['unix_timestamp'].apply(lambda x: convert_timestamp(x, is_nanos))
        
        # Remove rows where timestamp conversion failed
        merged_df = merged_df.dropna(subset=['datetime'])
        
        # Rename back to original column name if needed
        if timestamp_column != 'unix_timestamp':
            merged_df = merged_df.rename(columns={'unix_timestamp': timestamp_column})
        
        # Reorder columns to put datetime after timestamp
        timestamp_col = timestamp_column
        cols = merged_df.columns.tolist()
        cols.remove('datetime')
        timestamp_idx = cols.index(timestamp_col)
        cols.insert(timestamp_idx + 1, 'datetime')
        merged_df = merged_df[cols]
        
        return merged_df
    return None

def process_participant_folder(participant_folder):
    """Process a single participant folder."""
    raw_data_path = os.path.join(participant_folder, 'raw_data', 'v6')
    
    # Check if raw_data/v6 exists
    if not os.path.exists(raw_data_path):
        print(f"No raw_data/v6 folder found in {participant_folder}")
        return
    
    # Extract participant number
    participant_num = os.path.basename(participant_folder).split('#')[1]
    print(f"Processing participant {participant_num}")
    
    # Look for existing rawData_to_csv folders
    csv_folder = None
    for folder in os.listdir(participant_folder):
        if folder.startswith('rawData_to_csv'):
            csv_folder = os.path.join(participant_folder, folder)
            break
    
    # If no csv folder exists, create one with timestamp
    if not csv_folder:
        timestamp = datetime.datetime.now().strftime('%d%b')
        csv_folder = os.path.join(participant_folder, f'rawData_to_csv_{timestamp}')
        os.makedirs(csv_folder, exist_ok=True)
        
        # Process all avro files
        for file in os.listdir(raw_data_path):
            if file.endswith('.avro'):
                avro_path = os.path.join(raw_data_path, file)
                process_avro_file(avro_path, csv_folder)
    
    # Signal type mapping for file names
    signal_name_mapping = {
        'accelerometer': 'ACC',
        'gyroscope': 'GYRO',
        'eda': 'EDA',
        'temperature': 'TEMP',
        'tags': 'TAGS',
        'bvp': 'BVP',
        'systolic_peaks': 'PEAKS',
        'steps': 'STEPS'
    }
    
    # Required signals for model input
    required_signals = ['accelerometer', 'eda', 'temperature', 'bvp']
    
    # First check if all required signals are present
    missing_signals = []
    for signal_type in required_signals:
        pattern = os.path.join(csv_folder, f'*_{signal_type}.csv')
        if not glob.glob(pattern):
            missing_signals.append(signal_type)
    
    if missing_signals:
        print(f"Error: Missing required signals for participant {participant_num}: {', '.join(missing_signals)}")
        return
    
    # Merge signals
    signal_types = [
        'accelerometer', 
        'gyroscope', 
        'eda', 
        'temperature', 
        'tags',
        'bvp',
        'systolic_peaks',
        'steps'
    ]
    
    # Process required signals first
    for signal_type in required_signals:
        merged_df = merge_signal_files(csv_folder, signal_type)
        if merged_df is not None:
            # Create new filename format: P{participant_num}_{signal_name}.csv
            signal_name = signal_name_mapping[signal_type]
            output_filename = f'P{participant_num}_{signal_name}.csv'
            output_path = os.path.join(csv_folder, output_filename)
            merged_df.to_csv(output_path, index=False)
            print(f"Created merged file {output_filename} in {csv_folder}")
        else:
            print(f"Warning: Failed to merge {signal_type} data for participant {participant_num}")
            return
    
    # Process optional signals
    optional_signals = [s for s in signal_types if s not in required_signals]
    for signal_type in optional_signals:
        merged_df = merge_signal_files(csv_folder, signal_type)
        if merged_df is not None:
            signal_name = signal_name_mapping[signal_type]
            output_filename = f'P{participant_num}_{signal_name}.csv'
            output_path = os.path.join(csv_folder, output_filename)
            merged_df.to_csv(output_path, index=False)
            print(f"Created merged file {output_filename} in {csv_folder}")
    
    # Verify required files exist with correct format
    required_files = [f'P{participant_num}_{signal_name_mapping[s]}.csv' for s in required_signals]
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(csv_folder, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
            continue
        
        # Check file format
        df = pd.read_csv(filepath)
        if 'datetime' not in df.columns:
            print(f"Error: Missing datetime column in {filename}")
            return
        
        if signal_name_mapping[signal_type] == 'ACC':
            if not all(col in df.columns for col in ['x', 'y', 'z']):
                print(f"Error: Missing accelerometer components in {filename}")
                return
        elif signal_name_mapping[signal_type] in ['EDA', 'TEMP', 'BVP']:
            signal_col = signal_name_mapping[signal_type].lower()
            if signal_col not in df.columns:
                print(f"Error: Missing {signal_col} column in {filename}")
                return
    
    if missing_files:
        print(f"Error: Missing required files for participant {participant_num}: {', '.join(missing_files)}")
        return
    
    print(f"Successfully processed all data for participant {participant_num}")
    print("Files are ready for feature extraction using check_data_format.py")

def main():
    base_path = r'E:\NYU\academic\Y2S2\CUSP-GX-7133\DataManhattan\ParticipantData'
    
    # Process each participant folder
    for folder in os.listdir(base_path):
        if folder.startswith('Participant#'):
            participant_folder = os.path.join(base_path, folder)
            print(f"\nProcessing {folder}...")
            process_participant_folder(participant_folder)

if __name__ == "__main__":
    main() 