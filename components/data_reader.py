import h5py
import numpy as np
import os
import glob
import json
import random

def get_sample_files(data_path):
    """Get list of available CSI data files"""
    try:
        pattern = os.path.join(data_path, "*.h5")
        files = glob.glob(pattern)
        return files
    except Exception as e:
        print(f"Error getting sample files: {e}")
        return []

def load_csi_data(file_path):
    """
    Load CSI data from h5 file
    
    Args:
        file_path: path to h5 file
        
    Returns:
        csi_data: processed CSI data array
        true_label: true activity label
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Extract CSI data
            if 'csi' in f.keys():
                csi_raw = f['csi'][:]
            elif 'CSI' in f.keys():
                csi_raw = f['CSI'][:]
            else:
                # Try to find the data key
                keys = list(f.keys())
                csi_raw = f[keys[0]][:]
            
            # Process CSI data
            csi_data = process_csi_data(csi_raw)
            
            # Extract true label from filename
            true_label = extract_label_from_filename(file_path)
            
        return csi_data, true_label
        
    except Exception as e:
        print(f"Error loading CSI data from {file_path}: {e}")
        # Return dummy data for demo
        return generate_dummy_csi_data(), random.randint(0, 4)

def process_csi_data(csi_raw):
    """
    Process raw CSI data to the format expected by the model
    
    Args:
        csi_raw: raw CSI data from h5 file
        
    Returns:
        processed_data: processed CSI data ready for model input
    """
    try:
        # Handle complex data
        if np.iscomplexobj(csi_raw):
            # Extract amplitude and phase
            amplitude = np.abs(csi_raw)
            phase = np.angle(csi_raw)
            
            # Combine amplitude and phase features
            csi_data = np.concatenate([amplitude, phase], axis=-1)
        else:
            csi_data = csi_raw
        
        # Ensure correct shape for model (batch, seq_len, features)
        if len(csi_data.shape) == 3:
            # If shape is (num_subcarriers, time, features), transpose
            if csi_data.shape[1] > csi_data.shape[0]:
                csi_data = np.transpose(csi_data, (1, 0, 2))
            # Flatten last two dimensions
            csi_data = csi_data.reshape(csi_data.shape[0], -1)
        
        # Ensure we have the right sequence length (500)
        target_seq_len = 500
        if csi_data.shape[0] > target_seq_len:
            # Truncate to target length
            csi_data = csi_data[:target_seq_len]
        elif csi_data.shape[0] < target_seq_len:
            # Pad to target length
            padding = np.zeros((target_seq_len - csi_data.shape[0], csi_data.shape[1]))
            csi_data = np.vstack([csi_data, padding])
        
        # Ensure feature dimension is 232
        target_features = 232
        if csi_data.shape[1] > target_features:
            csi_data = csi_data[:, :target_features]
        elif csi_data.shape[1] < target_features:
            padding = np.zeros((csi_data.shape[0], target_features - csi_data.shape[1]))
            csi_data = np.hstack([csi_data, padding])
        
        # Normalize data
        if np.std(csi_data) > 0:
            csi_data = (csi_data - np.mean(csi_data)) / np.std(csi_data)
        
        return csi_data
        
    except Exception as e:
        print(f"Error processing CSI data: {e}")
        return generate_dummy_csi_data()

def extract_label_from_filename(file_path):
    """
    Extract activity label from filename
    
    Args:
        file_path: path to CSI data file
        
    Returns:
        label: activity label index
    """
    try:
        filename = os.path.basename(file_path)
        
        # Load label mapping
        with open('data/metadata/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        # Extract activity from filename patterns
        if 'jumping' in filename or 'jump' in filename:
            return label_mapping['label_to_idx']['jumping']
        elif 'running' in filename or 'run' in filename:
            return label_mapping['label_to_idx']['running']
        elif 'walking' in filename or 'walk' in filename:
            return label_mapping['label_to_idx']['walking']
        elif 'waving' in filename or 'wave' in filename:
            return label_mapping['label_to_idx']['wavinghand']
        elif 'breathing' in filename or 'seated' in filename:
            return label_mapping['label_to_idx']['seated-breathing']
        else:
            # Default to random if pattern not found
            return random.randint(0, 4)
            
    except Exception as e:
        print(f"Error extracting label from filename: {e}")
        return random.randint(0, 4)

def generate_dummy_csi_data():
    """Generate dummy CSI data for demo purposes"""
    # Generate realistic-looking CSI data
    seq_len, features = 500, 232
    
    # Create base signal with some patterns
    t = np.linspace(0, 10, seq_len)
    base_signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
    
    # Create dummy CSI data with some structure
    csi_data = np.zeros((seq_len, features))
    for i in range(features):
        # Add some variation across subcarriers
        freq_response = np.sin(2 * np.pi * i / features * 4)
        csi_data[:, i] = base_signal * (1 + 0.3 * freq_response) + 0.1 * np.random.randn(seq_len)
    
    # Normalize
    csi_data = (csi_data - np.mean(csi_data)) / np.std(csi_data)
    
    return csi_data

def get_sample_metadata():
    """Get metadata about available samples"""
    try:
        sample_files = get_sample_files('data/sample_csi')
        
        metadata = {
            'total_samples': len(sample_files),
            'activities': {},
            'file_list': sample_files
        }
        
        # Count samples per activity
        for file_path in sample_files:
            label = extract_label_from_filename(file_path)
            if label not in metadata['activities']:
                metadata['activities'][label] = 0
            metadata['activities'][label] += 1
        
        return metadata
        
    except Exception as e:
        print(f"Error getting sample metadata: {e}")
        return {
            'total_samples': 0,
            'activities': {},
            'file_list': []
        } 