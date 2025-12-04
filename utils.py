# Import required libraries
import numpy as np
import pydicom
from pathlib import Path

def get_dicom_files_with_slice_locations(directory_path):
    """
    Loop through all DICOM files in a directory and extract their paths and slice locations.
    
    Args:
        directory_path (str): Path to the directory containing DICOM files
        
    Returns:
        tuple: (list of file paths, list of corresponding slice locations)
    """
    dicom_paths = []
    slice_locations = []
    
    # Convert to Path object for easier handling
    dir_path = Path(directory_path)
    
    # Check if directory exists
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Loop through all files in directory (including subdirectories)
    for file_path in dir_path.rglob('*'):
        # Skip directories
        if file_path.is_dir():
            continue
            
        try:
            # Try to read as DICOM file
            dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
            
            # Check if SliceLocation exists in metadata
            if hasattr(dcm, 'Slice Location'):
                dicom_paths.append(str(file_path))
                slice_locations.append(float(dcm['Slice Location'].value))
            elif hasattr(dcm, 'ImagePositionPatient'):
                # Alternative: use Z coordinate from ImagePositionPatient if SliceLocation not available
                dicom_paths.append(str(file_path))
                slice_locations.append(float(dcm.ImagePositionPatient[2]))
        except Exception as e:
            # Skip files that are not valid DICOM files
            continue
    
    return dicom_paths, slice_locations

def order_slices(img_paths, slice_locations, reverse=False):
    sorted_ids = np.argsort(slice_locations)
    if reverse:
        sorted_ids = sorted_ids[::-1]
    sorted_img_paths = np.array(img_paths)[sorted_ids].tolist()
    sorted_slice_locs = np.sort(slice_locations).tolist()

    return sorted_img_paths, sorted_slice_locs