import numpy as np

# Reverse windowing function to reconstruct HU values from normalized image tensor for lung segmentor
def reverse_windowing(image, center, width, bit_size=16):
    """Reverse windowing function to transform normalized pixel values back to HU values.
    
    Reverses the windowing operation to recover original HU values from normalized [-1, 1] range.
    Assumes the input image has been normalized from [0, 2**bit_size - 1] to [-1, 1].
    
    Args:
        image (ndarray): Numpy image array with values in [-1, 1] range
        center (float): Window center (or level) used in original windowing
        width (float): Window width used in original windowing
        bit_size (int): Max bit size of pixel used in original windowing
    
    Returns:
        ndarray: Numpy array of HU values
    """
    image = image.copy()
    
    # First, convert from [-1, 1] back to [0, y_max]
    y_min = 0
    y_max = 2**bit_size - 1
    y_range = y_max - y_min
    
    # Map from [-1, 1] to [0, y_max]
    windowed_image = (image + 1) / 2 * y_range + y_min
    
    # Now reverse the windowing operation
    c = center - 0.5
    w = width - 1
    
    # Create output array for HU values
    hu_values = np.zeros_like(windowed_image)
    
    # Pixels that were set to y_min (black) -> values below window
    below = windowed_image <= y_min
    hu_values[below] = c - w / 2
    
    # Pixels that were set to y_max (white) -> values above window
    above = windowed_image >= y_max
    hu_values[above] = c + w / 2
    
    # Pixels that were in the window range -> reverse the linear mapping
    between = np.logical_and(~below, ~above)
    if between.any():
        # Reverse: y = ((x - c) / w + 0.5) * y_range + y_min
        # Solve for x: (y - y_min) / y_range = (x - c) / w + 0.5
        #              (y - y_min) / y_range - 0.5 = (x - c) / w
        #              x = ((y - y_min) / y_range - 0.5) * w + c
        hu_values[between] = ((windowed_image[between] - y_min) / y_range - 0.5) * w + c
    
    return hu_values