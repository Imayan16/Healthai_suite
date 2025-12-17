#!/usr/bin/env python3
"""
Download Model Files for HealthAI Suite

This script downloads all required pre-trained models from Google Drive.
Usage: python models/download_models.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Google Drive file IDs and target filenames
MODELS = {
    # ML Models
    'los_model.pkl': 'YOUR_GOOGLE_DRIVE_ID_1',
    'los_scaler.pkl': 'YOUR_GOOGLE_DRIVE_ID_2',
    'kmeans_cluster_model.pkl': 'YOUR_GOOGLE_DRIVE_ID_3',
    'cluster_scaler_final.pkl': 'YOUR_GOOGLE_DRIVE_ID_4',
    'xgboost_disease_model.json': 'YOUR_GOOGLE_DRIVE_ID_5',
    'association_rules.json': 'YOUR_GOOGLE_DRIVE_ID_6',
    # CNN Model for Imaging Diagnosis
    'pneumonia_cnn_model.h5': '1bAJgZRdRvXuHJA74ayHZzZOcNuNnwygb',
}

def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive using its file ID
    """
    url = f'https://drive.google.com/uc?id={file_id}'
    cmd = ['curl', '-L', '-o', destination, url]
    
    print(f'Downloading {destination}...')
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            print(f'✓ Downloaded: {destination}')
            return True
        else:
            print(f'✗ Failed to download: {destination}')
            return False
    except subprocess.CalledProcessError as e:
        print(f'✗ Error downloading {destination}: {e}')
        return False
    except Exception as e:
        print(f'✗ Unexpected error: {e}')
        return False

def main():
    """
    Main function to download all models
    """
    models_dir = Path(__file__).parent
    os.chdir(models_dir)
    
    print("="*60)
    print("HealthAI Suite - Model Downloader")
    print("="*60)
    print()
    
    successful = 0
    failed = 0
    failed_files = []
    
    for filename, file_id in MODELS.items():
        if file_id.startswith('YOUR_GOOGLE_DRIVE_ID'):
            print(f'⊘ Skipping {filename} (Google Drive ID not configured)')
            continue
            
        destination = models_dir / filename
        
        if destination.exists():
            print(f'⊘ Already exists: {filename} (skipping)')
            continue
            
        if download_file_from_google_drive(file_id, str(destination)):
            successful += 1
        else:
            failed += 1
            failed_files.append(filename)
    
    print()
    print("="*60)
    print("Download Summary")
    print("="*60)
    print(f'✓ Successful: {successful}')
    print(f'✗ Failed: {failed}')
    
    if failed_files:
        print(f'\nFailed to download:')
        for f in failed_files:
            print(f'  - {f}')
        return 1
    else:
        print('\n✓ All models downloaded successfully!')
        return 0

if __name__ == '__main__':
    sys.exit(main())
