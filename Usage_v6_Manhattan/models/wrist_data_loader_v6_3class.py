import os
import pickle
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict

class WristDataset(Dataset):
    def __init__(self, df):
        # Update features list to include all features
        self.features = ['ACC_x', 'ACC_y', 'ACC_z', 'EDA', 'TEMP', 'BVP',
                        'EDA_phasic', 'EDA_tonic']
        # Add statistical features
        stats_features = [col for col in df.columns if any(x in col for x in 
                        ['_mean', '_std', '_min', '_max', '_slope'])]
        self.features.extend(stats_features)
        
        self.X = df[self.features].values
        self.y = df['label'].values
        self.subject = df['subject'].values
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Update reshape to match new feature dimension
        features = self.X[idx].reshape(1, -1)  # shape: (1, num_features)
        label = self.y[idx]
        subject = self.subject[idx]
        
        return (
            torch.FloatTensor(features),
            torch.LongTensor([label])[0],
            torch.LongTensor([subject])[0]
        )

class WristDataLoader_v6:
    def __init__(self, data_path):
        self.data_path = data_path
        self.fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
        self.features = ['ACC_x', 'ACC_y', 'ACC_z', 'EDA', 'TEMP', 'BVP', 
                        'EDA_phasic', 'EDA_tonic', 'subject']
        self.target_freq = 64  # Hz
    
    def process_labels(self, labels):
        """Process labels into 3 classes"""
        processed_labels = np.zeros_like(labels)
        
        # 修改标签映射（从0开始）
        processed_labels[labels == 1] = 0  # baseline
        processed_labels[labels == 2] = 1  # stress
        processed_labels[(labels == 3) | (labels == 4)] = 2  # amusement/meditation
        
        # 创建有效标签掩码
        valid_mask = ((labels == 1) |  # baseline
                      (labels == 2) |  # stress
                      (labels == 3) | 
                      (labels == 4))   # amusement/meditation
        
        return processed_labels, valid_mask

    def load_subject_data(self, subject_id):
        """Load single subject wrist data and labels"""
        subject_path = os.path.join(self.data_path, f'S{subject_id}', f'S{subject_id}.pkl')
        print(f"Loading data from: {subject_path}")
        
        if not os.path.exists(subject_path):
            print(f"File not found: {subject_path}")
            # Check if directory exists
            subject_dir = os.path.dirname(subject_path)
            if not os.path.exists(subject_dir):
                print(f"Subject directory not found: {subject_dir}")
            # List available files in the directory
            if os.path.exists(self.data_path):
                print(f"Available subjects in {self.data_path}:")
                print(os.listdir(self.data_path))
            return None, None
        
        try:
            with open(subject_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                print(f"Successfully loaded data for subject {subject_id}")
                return data['signal']['wrist'], data['label']
        except Exception as e:
            print(f"Error loading data for subject {subject_id}: {str(e)}")
            return None, None

    def decompose_eda(self, eda_signal):
        """Decompose EDA signal into phasic and tonic components"""
        # Convert numpy array to pandas Series to use rolling functions
        eda_series = pd.Series(eda_signal)
        window_size = 320  # 5 seconds at 64Hz
        
        # Calculate moving average for tonic component
        eda_tonic = eda_series.rolling(window=window_size, center=True).mean()
        # Calculate phasic component as the difference
        eda_phasic = eda_series - eda_tonic
        
        # Fill NaN values at the edges
        eda_tonic = eda_tonic.fillna(method='bfill').fillna(method='ffill')
        eda_phasic = eda_phasic.fillna(method='bfill').fillna(method='ffill')
        
        return eda_phasic.values, eda_tonic.values

    def preprocess_and_align_data(self, signals, labels, subject_id):
        """Preprocess signals and align with labels, with additional features"""
        print(f"\nPreprocessing data for subject {subject_id}...")
        
        # Create DataFrames for each signal
        eda_df = pd.DataFrame(signals['EDA'], columns=['EDA'])
        bvp_df = pd.DataFrame(signals['BVP'], columns=['BVP'])
        acc_df = pd.DataFrame(signals['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
        temp_df = pd.DataFrame(signals['TEMP'], columns=['TEMP'])
        
        # Print original signal shapes
        print(f"Original shapes:")
        print(f"EDA: {eda_df.shape}, ACC: {acc_df.shape}")
        print(f"TEMP: {temp_df.shape}, BVP: {bvp_df.shape}")
        
        # Calculate number of samples needed for each signal at 64Hz
        duration = len(labels) / 700  # Total duration in seconds (labels at 700Hz)
        target_samples = int(duration * 64)  # Number of samples needed at 64Hz
        
        # Create uniform time points for 64Hz
        time_64hz = np.linspace(0, duration, target_samples)
        
        # Create time points for each signal based on their original frequency
        time_eda = np.linspace(0, duration, len(eda_df))
        time_acc = np.linspace(0, duration, len(acc_df))
        time_temp = np.linspace(0, duration, len(temp_df))
        time_labels = np.linspace(0, duration, len(labels))
        
        # Interpolate signals to 64Hz
        eda_interp = np.interp(time_64hz, time_eda, eda_df['EDA'].values)
        temp_interp = np.interp(time_64hz, time_temp, temp_df['TEMP'].values)
        
        # Interpolate ACC (for each axis)
        acc_x_interp = np.interp(time_64hz, time_acc, acc_df['ACC_x'].values)
        acc_y_interp = np.interp(time_64hz, time_acc, acc_df['ACC_y'].values)
        acc_z_interp = np.interp(time_64hz, time_acc, acc_df['ACC_z'].values)
        
        # For BVP (already at 64Hz), just ensure it matches the length
        bvp_interp = bvp_df['BVP'].values[:target_samples]
        if len(bvp_interp) < target_samples:
            bvp_interp = np.pad(bvp_interp, (0, target_samples - len(bvp_interp)), 'edge')
        
        # For labels, use nearest neighbor interpolation
        label_indices = np.searchsorted(time_labels, time_64hz)
        label_indices = np.clip(label_indices, 0, len(labels) - 1)
        labels_interp = labels[label_indices]
        
        # EDA decomposition
        eda_phasic, eda_tonic = self.decompose_eda(eda_interp)
        
        # Calculate statistical features
        window_size = 320  # 5 seconds at 64Hz
        step_size = window_size // 4  # 75% overlap
        
        def calculate_stats(signal_name, signal_data):
            signal_series = pd.Series(signal_data)
            return pd.DataFrame({
                f'{signal_name}_mean': signal_series.rolling(window=window_size, center=True).mean(),
                f'{signal_name}_std': signal_series.rolling(window=window_size, center=True).std(),
                f'{signal_name}_min': signal_series.rolling(window=window_size, center=True).min(),
                f'{signal_name}_max': signal_series.rolling(window=window_size, center=True).max(),
                f'{signal_name}_slope': (signal_series.diff(window_size) / window_size)
            })
        
        # Create aligned DataFrame with all features
        aligned_df = pd.DataFrame({
            # Original signals
            'ACC_x': acc_x_interp,
            'ACC_y': acc_y_interp,
            'ACC_z': acc_z_interp,
            'EDA': eda_interp,
            'TEMP': temp_interp,
            'BVP': bvp_interp,
            # EDA components
            'EDA_phasic': eda_phasic,
            'EDA_tonic': eda_tonic,
            'subject': subject_id,
            'label': labels_interp
        })
        
        # Add statistical features
        for signal_name in ['BVP', 'EDA', 'TEMP', 'EDA_phasic', 'EDA_tonic']:
            stats_df = calculate_stats(signal_name, aligned_df[signal_name])
            aligned_df = pd.concat([aligned_df, stats_df], axis=1)
        
        # Handle NaN values from rolling calculations
        aligned_df = aligned_df.fillna(method='bfill').fillna(method='ffill')
        
        # Reorder columns to put subject and label at the end
        feature_cols = [col for col in aligned_df.columns if col not in ['subject', 'label']]
        final_cols = feature_cols + ['subject', 'label']
        aligned_df = aligned_df[final_cols]
        
        print(f"Aligned data shape for subject {subject_id}: {aligned_df.shape}")
        print("Features order:")
        print(aligned_df.columns.tolist())
        
        return aligned_df

    def get_aligned_data(self, subject_id):
        """Get aligned signals and labels for a subject"""
        signals, labels = self.load_subject_data(subject_id)
        aligned_data = self.preprocess_and_align_data(signals, labels, subject_id)
        return aligned_data

    def stratified_temporal_split(self, all_data, train_ratio=0.8):
        """Perform stratified temporal split based on subjects and labels"""
        train_indices = []
        test_indices = []
        
        print(f"\nPerforming stratified split with ratio {train_ratio}")
        print(f"Input data shape: {all_data.shape}")
        
        # Reset index to ensure proper indexing
        all_data = all_data.reset_index(drop=True)
        
        # Group by subject first to maintain subject independence
        for subject in all_data['subject'].unique():
            subject_data = all_data[all_data['subject'] == subject]
            print(f"\nProcessing subject {subject}")
            print(f"Subject data shape: {subject_data.shape}")
            
            # For each subject, group by label to maintain class distribution
            for label in subject_data['label'].unique():
                label_data = subject_data[subject_data['label'] == label]
                indices = label_data.index.tolist()
                
                if len(indices) > 0:
                    # Calculate split point
                    n_train = max(1, int(len(indices) * train_ratio))
                    print(f"Label {label}: total {len(indices)}, train {n_train}")
                    
                    # Split maintaining temporal order
                    train_indices.extend(indices[:n_train])
                    test_indices.extend(indices[n_train:])
        
        print(f"\nFinal split sizes - Train: {len(train_indices)}, Test: {len(test_indices)}")
        return train_indices, test_indices

    def get_data_loaders(self, subjects, train_ratio=0.8, valid_ratio=0.2, batch_size=32, window_size=320):
        all_data = []
        print("\nProcessing subjects:", subjects)
        
        for subject in subjects:
            try:
                print(f"\nProcessing subject {subject}...")
                signals, labels = self.load_subject_data(subject)
                
                if signals is None or labels is None:
                    print(f"Skipping subject {subject} due to loading error.")
                    continue
                
                df = self.preprocess_and_align_data(signals, labels, subject)
                
                # Create windows of data
                window_count = 0
                for i in range(0, len(df) - window_size, window_size//4):
                    window = df.iloc[i:i+window_size]
                    if len(window) == window_size:
                        # Get most common label in window
                        processed_labels, valid_mask = self.process_labels(window['label'].values)
                        if np.any(valid_mask):
                            valid_labels = processed_labels[valid_mask]
                            mode_label = np.bincount(valid_labels).argmax()
                            
                            # Calculate features for each window
                            features = {}
                            # 原始信号
                            for feature in ['ACC_x', 'ACC_y', 'ACC_z', 'EDA', 'TEMP', 'BVP']:
                                features[feature] = window[feature].mean()
                            
                            # EDA分解特征
                            features['EDA_phasic'] = window['EDA_phasic'].mean()
                            features['EDA_tonic'] = window['EDA_tonic'].mean()
                            
                            # 统计特征
                            for col in df.columns:
                                if any(x in col for x in ['_mean', '_std', '_min', '_max', '_slope']):
                                    features[col] = window[col].mean()
                            
                            # 添加subject和label
                            features['subject'] = window['subject'].iloc[0]
                            features['label'] = mode_label
                            all_data.append(features)
                            window_count += 1
                
                print(f"Created {window_count} windows for subject {subject}")
                
            except Exception as e:
                print(f"Error processing subject {subject}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        if len(df) == 0:
            raise ValueError("No data available after processing")
        
        # Print detailed statistics
        print("\nInitial data statistics:")
        print(f"Total samples: {len(df)}")
        print("Label distribution:")
        print(df['label'].value_counts().sort_index())
        print("\nSamples per subject:")
        print(df['subject'].value_counts().sort_index())
        
        try:
            # First split into train and test
            df = df.reset_index(drop=True)  # Reset index before splitting
            train_indices, test_indices = self.stratified_temporal_split(df, train_ratio)
            print(f"\nFirst split - Train size: {len(train_indices)}, Test size: {len(test_indices)}")
            
            train_valid_df = df.iloc[train_indices].reset_index(drop=True)  # Reset index again
            test_df = df.iloc[test_indices].reset_index(drop=True)
            
            # Then split train into train and validation
            valid_ratio_adjusted = valid_ratio / (1 - valid_ratio)  # Adjust ratio for second split
            train_indices_final, valid_indices = self.stratified_temporal_split(
                train_valid_df, 1.0 - valid_ratio_adjusted
            )
            
            valid_df = train_valid_df.iloc[valid_indices].reset_index(drop=True)
            train_df = train_valid_df.iloc[train_indices_final].reset_index(drop=True)
            
            # Standardize features
            scaler = StandardScaler()
            train_df[self.features] = scaler.fit_transform(train_df[self.features])
            valid_df[self.features] = scaler.transform(valid_df[self.features])
            test_df[self.features] = scaler.transform(test_df[self.features])
            
            # Create datasets
            train_dataset = WristDataset(train_df)
            valid_dataset = WristDataset(valid_df)
            test_dataset = WristDataset(test_df)
            
            # Print final statistics
            print("\nFinal Dataset Statistics:")
            print(f"Train set size: {len(train_dataset)}")
            print(f"Validation set size: {len(valid_dataset)}")
            print(f"Test set size: {len(test_dataset)}")
            
            print("\nLabel Distribution:")
            print("Train:", train_df['label'].value_counts().sort_index())
            print("Valid:", valid_df['label'].value_counts().sort_index())
            print("Test:", test_df['label'].value_counts().sort_index())
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            return train_loader, valid_loader, test_loader
            
        except Exception as e:
            print(f"\nError in data splitting: {str(e)}")
            print("Debug information:")
            print(f"DataFrame shape: {df.shape}")
            print("Sample of indices:")
            print(f"Train indices (first 5): {train_indices[:5] if len(train_indices) > 5 else train_indices}")
            raise

# Test the data loader
if __name__ == "__main__":
    print("Running start.")
    data_path = r"C:\Users\xy2593\Desktop\WESAD"  # Updated path
    print("Initializing data loader...")
    loader = WristDataLoader_v6(data_path)  # Updated class name
    
    try:
        # Test with one subject
        test_subject = 2  # Define the subject ID first
        print(f"\nProcessing subject S{test_subject}...")
        aligned_data = loader.get_aligned_data(test_subject)
        
        # Print data info
        print("\nDataset Info:")
        print(aligned_data.info())
        
        # Print first few rows
        print("\nFirst few rows of aligned data:")
        print(aligned_data.head())
        
        # Print label distribution
        print("\nLabel distribution:")
        print(aligned_data['label'].value_counts().sort_index())
        
        # Print signal statistics
        print("\nSignal statistics:")
        print(aligned_data.describe())
        
        print(f"\nData successfully processed and saved to 'sample_aligned_data_S{test_subject}.csv'")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check if the WESAD dataset path is correct and the data files exist.") 